"""Implement user code."""

from collections import namedtuple
import copy
from itertools import chain
import logging
from typing import Dict, Optional, Callable, List
import weakref

import numpy as np
import torch

from .data import construct_dataloader

log = logging.getLogger(__name__)


def construct_user(model, loss_fn, cfg_case, setup, need_all_data=False):
    """Interface function."""
    if cfg_case.user.user_type == "local_gradient":
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=cfg_case.user.user_idx, return_full_dataset=need_all_data)
        # The user will deepcopy this model template to have their own
        user = UserSingleStep(model, loss_fn, dataloader, setup, idx=cfg_case.user.user_idx, cfg_user=cfg_case.user)
    elif cfg_case.user.user_type == "local_update":
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=cfg_case.user.user_idx, return_full_dataset=need_all_data)
        user = UserMultiStep(model, loss_fn, dataloader, setup, idx=cfg_case.user.user_idx, cfg_user=cfg_case.user)
    elif cfg_case.user.user_type == "multiuser_aggregate":
        dataloaders, indices = [], []
        for idx in range(*cfg_case.user.user_range):
            dataloaders += [construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=idx, return_full_dataset=need_all_data)]
            indices += [idx]
        user = MultiUserAggregate(model, loss_fn, dataloaders, setup, cfg_case.user, user_indices=indices)
    return user


class UserSingleStep(torch.nn.Module):
    """A user who computes a single local update step."""

    def __init__(self, model, loss, dataloader, setup, idx, cfg_user):
        """Initialize from cfg_user dict which contains atleast all keys in the matching .yaml :>"""
        super().__init__()
        self.cfg = cfg_user
        self.num_data_points = cfg_user.num_data_points

        self.provide_labels = cfg_user.provide_labels
        self.provide_num_data_points = cfg_user.provide_num_data_points
        self.provide_buffers = cfg_user.provide_buffers

        self.user_idx = idx
        self.setup = setup

        self.model = copy.deepcopy(model)
        self.model.to(**setup)

        self.defense_repr = []
        self._initialize_local_privacy_measures(cfg_user.local_diff_privacy)

        self.dataloader = dataloader
        self.loss = copy.deepcopy(loss)  # Just in case the loss contains state

        self.counted_queries = 0  # Count queries to this user

    def __repr__(self):
        n = "\n"
        return f"""User (of type {self.__class__.__name__}) with settings:
    Number of data points: {self.num_data_points}

    Threat model:
    User provides labels: {self.provide_labels}
    User provides buffers: {self.provide_buffers}
    User provides number of data points: {self.provide_num_data_points}

    Data:
    Dataset: {self.dataloader.name}
    user: {self.user_idx}
    {n.join(self.defense_repr)}
        """

    def _initialize_local_privacy_measures(self, local_diff_privacy):
        """Initialize generators for noise in either gradient or input."""
        # Noises for gradient.
        if local_diff_privacy["gradient_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **self.setup)
            scale = torch.as_tensor(local_diff_privacy["gradient_noise"], **self.setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} gradient noise with strength {scale.item()}.'
            )
        else:
            self.generator = None
        
        # Noises for input data.
        if local_diff_privacy["input_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **self.setup)
            scale = torch.as_tensor(local_diff_privacy["input_noise"], **self.setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator_input = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator_input = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"].lower() == "dp-sgd":
                budget = local_diff_privacy.get("privacy_budget", 1.0)
                slack_term = local_diff_privacy.get("privacy_slack_term", 1e-5)
                clip_bound = local_diff_privacy.get("per_example_clipping", 1.0)

                def noise_scale(budget, clip_bound, slack_term=1e-5):
                    # \sigma^2\geq\frac{2\mathrm{ln}(1.25/\delta)}{\epsilon^2}
                    multiple = np.sqrt(2 * np.log(1.25 / slack_term) / budget ** 2)
                    return clip_bound * multiple

                scale = noise_scale(budget=budget, clip_bound=clip_bound, slack_term=slack_term)
                self.generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} input noise with strength {scale.item()}.'
            )
        else:
            self.generator_input = None
        
        # configs for gradient clipping
        self.clip_value = local_diff_privacy.get("per_example_clipping", 0.0)
        if self.clip_value > 0:
            self.defense_repr.append(f"Defense: Gradient clipping to maximum of {self.clip_value}.")
        
        # configs for gradient pruning
        self.pruning_type = local_diff_privacy.get("pruning_type", "naive")
        self.pruning_ratio = local_diff_privacy.get("pruning_ratio", 0.0)
        self.pruning_ratio = min(max(self.pruning_ratio, 0.0), 1.0)
        self.outpost_noise_perturb_ratio = local_diff_privacy.get("outpost_noise_perturb_ratio", 0.4)
        self.outpost_fail_preserve_rate_beta = local_diff_privacy.get("outpost_fail_preserve_rate_beta", 0.1)
        self.outpost_simulating_epoch = local_diff_privacy.get("outpost_simulating_epoch", 0)
        self.outpost_noise_scale_lambda = local_diff_privacy.get("outpost_noise_scale_lambda", 0.8)
        if self.pruning_ratio > 0:
            self.defense_repr.append(f"Defense: Gradient pruning with type {self.pruning_type}, ratio {self.pruning_ratio}.")

    def compute_local_updates(self, server_payload, custom_data=None):
        """Compute local updates to the given model based on server payload.

        Batchnorm behavior:
        If public buffers are sent by the server, then the user will be set into evaluation mode
        Otherwise the user is in training mode and sends back buffer based on .provide_buffers.

        Shared labels are canonically sorted for simplicity.

        Optionally custom data can be directly inserted here, superseding actual user data.
        Use this behavior only for demonstrations.
        """
        self.counted_queries += 1
        data = self._load_data()
        if custom_data is not None:
            data = custom_data
        B = data["labels"].shape[0]

        # reset model's param and buffer
        self._reset_model_with_payload(payload=server_payload)
        log.info(
            f"Computing user update on user {self.user_idx} in model mode: {'training' if self.model.training else 'eval'}."
        )

        # Compute local updates
        shared_grads = []
        shared_buffers = []
        self_ref = weakref.ref(self)
        def _compute_batch_gradient(data):
            # avoid circular reference
            if self_ref():
                setup = self_ref().setup
                data[self.data_key] = (
                    data[self.data_key].to(**setup) + self.generator_input.sample(data[self.data_key].shape).to(**setup)
                    if self.generator_input is not None
                    else data[self.data_key].to(**setup)
                )
            else:
                data[self.data_key] = (
                    data[self.data_key] + self.generator_input.sample(data[self.data_key].shape)
                    if self.generator_input is not None
                    else data[self.data_key]
                )
            
            labels = data["labels"]
            outputs = self.model(**data)
            loss = self.loss(outputs, labels)
            return torch.autograd.grad(loss, self.model.parameters())

        if self.clip_value > 0:  # Compute per-example gradients and clip them in this case
            shared_grads = [torch.zeros_like(p) for p in self.model.parameters()]
            for data_idx in range(B):
                data_point = {key: val[data_idx : data_idx + 1] for key, val in data.items()}
                per_example_grads = _compute_batch_gradient(data_point)
                self._clip_list_of_grad_(per_example_grads)
                torch._foreach_add_(shared_grads, per_example_grads)
            torch._foreach_div_(shared_grads, B)
        else:
            # Compute the forward pass
            shared_grads = _compute_batch_gradient(data)
        
        self._apply_pruning(shared_grads, data)
        self._apply_differential_noise(shared_grads)

        # compute loss of original model, updated model
        LossInfo = namedtuple('LossInfo', ['payload', 'loss'])
        loss_infos: [LossInfo] = [LossInfo(payload=server_payload, loss=self.compute_loss(data=data, is_detached=True))]
        if hasattr(self.cfg, "local_learning_rate") and self.cfg.local_learning_rate is not None:
            optimized_payload = self.compute_sgd_updated_payload(initial_payload=server_payload, data=data, lr=self.cfg.local_learning_rate)
            optimized_loss = self.compute_loss(payload=optimized_payload, data=data, is_detached=True)
            loss_infos.append(LossInfo(payload=optimized_payload, loss=optimized_loss))

        # shared buffer if needed.
        if server_payload["buffers"] is not None:
            shared_buffers = None
        else:
            shared_buffers = [b.clone().detach() for b in self.model.buffers()]

        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=data["labels"].sort()[0] if self.provide_labels else None,
            local_hyperparams=None,
            loss_infos=loss_infos,
            loss_fn=copy.deepcopy(self.loss),
            case_cfg=self.cfg
        )
        shared_data = dict(
            gradients=shared_grads,
            buffers=shared_buffers if self.provide_buffers else None,
            metadata=metadata
        )
        true_user_data = dict(data=data[self.data_key], labels=data["labels"], buffers=shared_buffers)

        return shared_data, true_user_data

    def _reset_model_with_payload(self, payload, model: torch.nn.Module = None):
        if model is None:
            model = self.model

        parameters = payload["parameters"]
        buffers = payload["buffers"]
        with torch.no_grad():
            for param, server_state in zip(model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                model.eval()
            else:
                for module in model.modules():
                    if hasattr(module, "momentum"):
                        module.momentum = None  # Force recovery without division
                model.train()

    def _clip_list_of_grad_(self, grads):
        """Apply differential privacy component per-example clipping."""
        grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
        if grad_norm > self.clip_value:
            [g.mul_(self.clip_value / (grad_norm + 1e-6)) for g in grads]

    def _apply_differential_noise(self, grads):
        """Apply differential privacy component gradient noise."""
        if self.generator is not None:
            for grad in grads:
                grad += self.generator.sample(grad.shape)

    def _apply_pruning(self, grads, data):
        if self.pruning_ratio <= 0:
            return
        
        method_by_type: Dict[str: Callable] = {
            "naive": self._apply_naive_pruning,
            "soteria": self._apply_soteria_pruning,
            "outpost": self._apply_outpost_pruning
        }
        if self.pruning_type.lower() not in method_by_type.keys():
            log.info(f"User Defense: Pruning with {self.pruning_type.lower()} is not supported.")
            return 
        
        method = method_by_type[self.pruning_type.lower()]
        return method(grads, data)

    def _apply_naive_pruning(self, grads: List[torch.Tensor], data: Dict[str, torch.Tensor]):
        "https://github.com/MiLab-HITSZ/2023YeHFGradInv/blob/main/breaching/cases/users.py#L250"
        if self.pruning_ratio <= 0:
            return
        
        pruning_rate = self.pruning_ratio
        for grad in grads:
            n = int(pruning_rate * grad.numel())
            tmp_grad = torch.abs(grad)
            sort_data = torch.sort(tmp_grad.flatten())[0]
            cut = sort_data[n-1]
            mask = tmp_grad > cut
            grad.mul_(mask)

    def _apply_soteria_pruning(self, grads: List[torch.Tensor], data: Dict[str, torch.Tensor]):
        """Gradient pruning with Soteria(CVPR'21).
        @Reference:
            [1] Sun, Jingwei, et al. "Soteria: Provable defense against privacy leakage in federated
                learning from representation perspective." Proceedings of the IEEE/CVF conference on 
                computer vision and pattern recognition. 2021.
            [2] https://github.com/jeremy313/Soteria
            [3] https://discuss.pytorch.org/t/using-forward-hooks-vs-modifying-the-forward-function/129923
        """

        # 1. Clone Data and Model, set data.require_grad = True
        from copy import deepcopy
        tmp_model: torch.nn.Module = deepcopy(self.model)
        tmp_model.load_state_dict(self.model.state_dict())
        tmp_model.to(self.setup["device"])
        tmp_model.zero_grad()
        tmp_model.train()

        tmp_data: Dict[str: torch.Tensor] = {
            key: value.detach().clone()
            for key, value in data.items()
        }
        tmp_data[self.data_key].requires_grad = True

        # 2. Hook model forwarding, listen to the input of first continuous Linear layers (at the end of model). 
        children: List[torch.nn.Module] = _retrieve_forwarded_modules_with_order(model=tmp_model, input=tmp_data)

        if not isinstance(children[-1], torch.nn.Linear):
            return

        i = len(children) - 1
        layer: torch.nn.Module = children[i]
        while i >= 0:
            if not isinstance(children[i], torch.nn.Linear):
                i = i + 1
                break
            else:
                i = i - 1

        representations = []
        def make_hook():
            def set_representations(module, input, output):
                representations.append(input)
            return set_representations
        handler = layer.register_forward_hook(make_hook())

        tmp_model(**tmp_data)
        handler.remove() 
        if len(representations) == 0:
            return

        # 3. Calculate mask of gradients, as what Soteria did.
        representation = representations[0][0]
        deviation_f1_target = torch.zeros_like(representation)
        deviation_f1_x_norm = torch.zeros_like(representation)
        for f in range(deviation_f1_x_norm.size(1)):
            deviation_f1_target[:,f] = 1
            representation.backward(deviation_f1_target, retain_graph=True)
            deviation_f1_x = tmp_data[self.data_key].grad.data
            deviation_f1_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/(representation.data[:,f])
            tmp_model.zero_grad()
            tmp_data[self.data_key].grad.data.zero_()
            deviation_f1_target[:,f] = 0

        import numpy as np

        deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
        thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), self.pruning_ratio * 100)
        mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

        # 4. Seek for the gradient of this layer's weight.
        #    Here we make a simplified design, match the first one with same shapes, and truncate it.
        i = len(grads) - 1
        while i >= 0:
            grad = grads[i]
            if not hasattr(layer, "weight"):
                continue
            if grad.size() == layer.weight.size():
                grads[i].mul_(torch.Tensor(mask).to(**self.setup))
                break
            i = i - 1

    def _apply_outpost_pruning(self, grads: List[torch.Tensor], data: Dict[str, torch.Tensor]):
        """Gradient pruning with Outpost(INFOCOM'23, best paper).
        @Reference:
            [1] Wang, Fei, Ethan Hugh, and Baochun Li. "More than Enough is Too Much: Adaptive Defenses against Gradient Leakage 
                in Production Federated Learning." IEEE INFOCOM 2023-IEEE Conference on Computer Communications. IEEE, 2023.
            [2] https://github.com/TL-System/plato/blob/main/examples/gradient_leakage_attacks/defense/Outpost/perturb.py#L10
        """

        # 0. Abort defenses randomly
        if torch.rand((1,)).item() > 1 / (1 + self.outpost_simulating_epoch * self.outpost_fail_preserve_rate_beta):
            print("[Debug] Tried to apply outpost, but randomly aborted.")
            return

        # 1. Calculate empirical FIM for each gradients
        each_grad_fims: List[torch.Tensor] = [grad.detach().clone().pow(2) for grad in grads]
        flattened_each_grad_fims = torch.cat([f.view(-1) for f in each_grad_fims])
        fim_thresh = np.percentile(flattened_each_grad_fims.detach().clone().cpu().numpy(), 100 - self.outpost_noise_perturb_ratio * 100)

        # 2. Calculate Gaussian Noise will apply to each layer: scale = {layer_param_var} * self.outpost_noise_scale_lambda
        param_vars = [p.var().clone().detach() for _, p in self.model.named_parameters()]
        scales: List[torch.Tensor] = [v * self.outpost_noise_scale_lambda for v in param_vars]

        for grad, fim, scale in zip(grads, each_grad_fims, scales):
            # 3. Pruning gradients
            n = int(self.pruning_ratio * grad.numel())
            tmp_grad = torch.abs(grad)
            sort_data = torch.sort(tmp_grad.flatten())[0]
            cut = sort_data[n-1]
            pruning_mask = (tmp_grad > cut).float()
            grad.mul_(pruning_mask.float())

            # 4. Perturbing with Gaussian noises $$N\sim(0, scale)$$
            perturbing_mask = (fim > fim_thresh).float()
            perturbing_mask = perturbing_mask * (torch.randn_like(perturbing_mask.float()) * scale)
            grad.add_(perturbing_mask)     

    def _load_data(self, setup=None):
        """Generate data from dataloader, truncated by self.num_data_points"""
        # Select data
        data_blocks = []
        num_samples = 0

        if setup is None:
            setup = self.setup

        for idx, data_block in enumerate(self.dataloader):
            data_blocks += [data_block]
            num_samples += data_block["labels"].shape[0]
            if num_samples > self.num_data_points:
                break

        if num_samples < self.num_data_points:
            raise ValueError(
                f"This user does not have the requested {self.num_data_points} samples,"
                f"they only own {num_samples} samples."
            )

        data = dict()
        for key in data_blocks[0]:
            data[key] = torch.cat([d[key] for d in data_blocks], dim=0)[: self.num_data_points].to(
                device=setup["device"]
            )
        self.data_key = "input_ids" if "input_ids" in data.keys() else "inputs"
        return data

    def print(self, user_data, **kwargs):
        """Print decoded user data to output."""
        tokenizer = self.dataloader.dataset.tokenizer
        decoded_tokens = tokenizer.batch_decode(user_data["data"], clean_up_tokenization_spaces=True)
        for line in decoded_tokens:
            print(line)

    def print_with_confidence(self, user_data, **kwargs):
        """Print decoded user data to output."""
        tokenizer = self.dataloader.dataset.tokenizer
        colors = [160, 166, 172, 178, 184, 190]
        thresholds = torch.as_tensor([0, 0.5, 0.75, 0.95, 0.99, 0.9999])

        def bg_color(text, confidence_score):
            threshold = ((confidence_score > thresholds) + torch.arange(0, len(colors)) / 100).argmax()
            return "\33[48;5;" + str(colors[threshold]) + "m" + text + "\33[0m"

        for sequence, sequence_confidence in zip(user_data["data"], user_data["confidence"]):
            for token, c in zip(sequence, sequence_confidence):
                decoded_token = tokenizer.decode(token)
                print(bg_color(decoded_token + " ", c), end="")
            print("\n")

    def print_and_mark_correct(self, user_data, true_user_data, **kwargs):
        """Print decoded user data to output."""
        tokenizer = self.dataloader.dataset.tokenizer

        def bg_color(text, correct):
            if correct:
                return "\33[48;5;190m" + text + "\33[0m"
            else:
                return "\33[48;5;160m" + text + "\33[0m"

        for sequence, gt_sequence in zip(user_data["data"], true_user_data["data"]):
            for token, gt_token in zip(sequence, gt_sequence):
                decoded_token = tokenizer.decode(token)
                print(bg_color(decoded_token + " ", token == gt_token), end="")
            print("\n")

    def plot(self, user_data, scale=False, print_labels=False):
        """Plot user data to output. Probably best called from a jupyter notebook."""
        import matplotlib.pyplot as plt  # lazily import this here

        dm = torch.as_tensor(self.dataloader.dataset.mean, **self.setup)[None, :, None, None]
        ds = torch.as_tensor(self.dataloader.dataset.std, **self.setup)[None, :, None, None]
        classes = self.dataloader.dataset.classes

        data = user_data["data"].clone().detach()
        labels = user_data["labels"].clone().detach() if user_data["labels"] is not None else None
        if labels is None:
            print_labels = False

        if scale:
            min_val, max_val = data.amin(dim=[2, 3], keepdim=True), data.amax(dim=[2, 3], keepdim=True)
            # print(f'min_val: {min_val} | max_val: {max_val}')
            data = (data - min_val) / (max_val - min_val)
        else:
            data.mul_(ds).add_(dm).clamp_(0, 1)
        data = data.to(dtype=torch.float32)

        if data.shape[0] == 1:
            plt.axis("off")
            plt.imshow(data[0].permute(1, 2, 0).cpu())
            if print_labels:
                plt.title(f"Data with label {classes[labels]}")
        else:
            grid_shape = int(torch.as_tensor(data.shape[0]).sqrt().ceil())
            s = 24 if data.shape[3] > 150 else 6
            fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
            label_classes = []
            for i, (im, axis) in enumerate(zip(data, axes.flatten())):
                axis.imshow(im.permute(1, 2, 0).cpu())
                if labels is not None and print_labels:
                    label_classes.append(classes[labels[i]])
                axis.axis("off")
            if print_labels:
                print(label_classes)

        plt.show()

    def compute_loss(self, payload=None, data=None, is_detached=False) -> torch.Tensor:
        """compute loss w.r.t. given payload and data.
            Notes: params and buffers in `self.model` will not be affected by calling this function.
        """
        if data is None:
            data = self._load_data()

        outputs = self.compute_outputs(payload=payload, data=data)
        loss = self.loss(outputs, data["labels"])
        if is_detached:
            loss = loss.detach()
        return loss

    def compute_outputs(self, payload=None, data=None, is_detached=False) -> torch.Tensor:
        """compute output w.r.t. given payload and data.
            Notes: params and buffers in `self.model` will not be affected by calling this function.
        """
        if data is None:
            data = self._load_data()

        model = self.model
        if payload is not None:
            model = copy.deepcopy(self.model)
            self._reset_model_with_payload(payload=payload, model=model)

        try:
            data[self.data_key] = (
                data[self.data_key].to(**self.setup) + self.generator_input.sample(data[self.data_key].shape).to(**self.setup)
                if self.generator_input is not None
                else data[self.data_key].to(**self.setup)
            )
        except BaseException:
            pass

        outputs = model(**data)
        if is_detached:
            outputs = outputs.detach()
        return outputs

    def payload(self, model: Optional[torch.nn.Module] = None) -> Dict:
        """Return a copied params and buffers in model with the format of payload.
        Notes:
            if `model` is None, we will return the payload of `self.model`.
        """

        if model is None:
            model = self.model

        return dict(
            parameters=[p.clone() for p in model.parameters()],
            buffers=[b.clone() for b in model.buffers()] if self.provide_buffers else None,
            metadata=None
        )

    def compute_sgd_updated_payload(self, initial_payload=None, data=None, lr=0.001) -> Dict:
        """Compute payload which has been updated by one SGD step.
        Notes:
            (1) The reason why we define such function, rather than computing `payload = payload - lr * gradient`
                directly, is because gradients of data may have a very small value (e.g. less than 1e-6). It will
                lead to a floating-point precision error when you directly subtract two of them. We have also
                found some similar deduction on this problem, please refer to the linkage below for more details.
            (2) This function can be seen as a functional tool, which means it will not affect params and buffers
                in `self.model`.
        Reference:
            [Losses not matching when training using hand-written SGD and torch.optim.SGD](https://discuss.pytorch.org/t/losses-not-matching-when-training-using-hand-written-sgd-and-torch-optim-sgd/3467)
        """
        if initial_payload is None:
            initial_payload = self.payload()
        if data is None:
            data = self._load_data()

        model = copy.deepcopy(self.model)
        loss_fn = copy.deepcopy(self.loss)
        self._reset_model_with_payload(payload=initial_payload, model=model)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer.zero_grad()
        data[self.data_key] = (
            data[self.data_key] + self.generator_input.sample(data[self.data_key].shape)
            if self.generator_input is not None
            else data[self.data_key]
        )
        outputs = model(**data)
        loss = loss_fn(outputs, data["labels"])
        loss.backward()
        optimizer.step()
        return self.payload(model=model)


class UserMultiStep(UserSingleStep):
    """A user who computes multiple local update steps as in a FedAVG scenario."""

    def __init__(self, model, loss, dataloader, setup, idx, cfg_user):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloader, setup, idx, cfg_user)

        self.num_local_updates = cfg_user.num_local_updates
        self.num_data_per_local_update_step = cfg_user.num_data_per_local_update_step
        self.local_learning_rate = cfg_user.local_learning_rate
        self.provide_local_hyperparams = cfg_user.provide_local_hyperparams

    def __repr__(self):
        n = "\n"
        return (
            super().__repr__()
            + n
            + f"""    Local FL Setup:
        Number of local update steps: {self.num_local_updates}
        Data per local update step: {self.num_data_per_local_update_step}
        Local learning rate: {self.local_learning_rate}

        Threat model:
        Share these hyperparams to server: {self.provide_local_hyperparams}

        """
        )

    def compute_local_updates(self, server_payload, custom_data=None):
        """Compute local updates to the given model based on server payload."""
        self.counted_queries += 1
        user_data = self._load_data()

        # Compute local updates
        parameters = server_payload["parameters"]
        buffers = server_payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(self.model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                self.model.eval()
            else:
                self.model.train()
        log.info(
            f"Computing user update on user {self.user_idx} in model mode: {'training' if self.model.training else 'eval'}."
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate)
        seen_data_idx = 0
        label_list = []
        for step in range(self.num_local_updates):
            data = {
                k: v[seen_data_idx : seen_data_idx + self.num_data_per_local_update_step] for k, v in user_data.items()
            }
            seen_data_idx += self.num_data_per_local_update_step
            seen_data_idx = seen_data_idx % self.num_data_points
            label_list.append(data["labels"].sort()[0])

            optimizer.zero_grad()
            # Compute the forward pass
            data[self.data_key] = (
                data[self.data_key] + self.generator_input.sample(data[self.data_key].shape)
                if self.generator_input is not None
                else data[self.data_key]
            )
            outputs = self.model(**data)
            loss = self.loss(outputs, data["labels"])
            loss.backward()

            grads_ref = [p.grad for p in self.model.parameters()]
            if self.clip_value > 0:
                self._clip_list_of_grad_(grads_ref)
            self._apply_differential_noise(grads_ref)
            optimizer.step()

        # compute loss
        LossInfo = namedtuple('LossInfo', ['payload', 'loss'])
        loss_infos: [LossInfo] = [
            LossInfo(payload=server_payload, loss=self.compute_loss(payload=server_payload, is_detached=True)),
            LossInfo(payload=self.payload(), loss=self.compute_loss(is_detached=True))
        ]

        # Share differential to server version:
        # This is equivalent to sending the new stuff and letting the server do it, but in line
        # with the gradients sent in UserSingleStep
        shared_grads = [
            (p_local - p_server.to(**self.setup)).clone().detach()
            for (p_local, p_server) in zip(self.model.parameters(), parameters)
        ]

        shared_buffers = [b.clone().detach() for b in self.model.buffers()]
        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=user_data["labels"] if self.provide_labels else None,
            local_hyperparams=dict(
                lr=self.local_learning_rate,
                steps=self.num_local_updates,
                data_per_step=self.num_data_per_local_update_step,
                labels=label_list,
            )
            if self.provide_local_hyperparams
            else None,
            data_key=self.data_key,
            loss_infos=loss_infos,
            loss_fn=copy.deepcopy(self.loss)
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=user_data[self.data_key], labels=user_data["labels"], buffers=shared_buffers)

        return shared_data, true_user_data


class ChainedDataloader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return chain(*self.dataloaders)

    def __len__(self):
        return torch.as_tensor([len(loader) for loader in self.dataloaders]).sum().item()

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataloaders[0], name)


class MultiUserAggregate(UserMultiStep):
    """A silo of users who compute local updates as in a fedSGD or fedAVG scenario and aggregate their results.

    For an unaggregated single silo refer to SingleUser classes as above.
    This aggregration is assumed to be safe (e.g. via secure aggregation) and the attacker and server only gain
    access to the aggregated local updates.

    self.dataloader of this class is actually quite unwieldy, due to its possible size.
    """

    def __init__(self, model, loss, dataloaders, setup, cfg_user, user_indices):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloaders, setup, None, cfg_user)

        self.num_users = len(dataloaders)

        self.users = []
        self.user_setup = dict(dtype=setup["dtype"], device=torch.device("cpu"))  # Simulate on CPU
        for idx in user_indices:
            if self.num_local_updates > 1:
                self.users.append(UserMultiStep(model, loss, dataloaders[idx], self.user_setup, idx, cfg_user))
            else:
                self.users.append(UserSingleStep(model, loss, dataloaders[idx], self.user_setup, idx, cfg_user))

        self.dataloader = ChainedDataloader(dataloaders)
        self.user_idx = f"{user_indices[0]}_{user_indices[-1]}"  # Only for printout identification

    def __repr__(self):
        n = "\n"
        return self.users[0].__repr__() + n + f"""    Number of aggregated users: {self.num_users}"""

    def compute_local_updates(self, server_payload, return_true_user_data=False):
        """Compute local updates to the given model based on server payload.

        Collecting and returning a tensor containing all input data (as for the other users) is disabled by default.
        (to save your RAM).
        """
        self.counted_queries += 1
        # Compute local updates

        server_payload["parameters"] = [p.to(**self.setup) for p in server_payload["parameters"]]
        if server_payload["buffers"] is not None:
            server_payload["buffers"] = [b.to(**self.setup) for b in server_payload["buffers"]]

        aggregate_updates = [torch.zeros_like(p) for p in server_payload["parameters"]]
        aggregate_buffers = [torch.zeros_like(b, dtype=torch.float) for b in server_payload["buffers"]]
        aggregate_labels = []
        aggregate_label_lists = []  # Only ever used in rare sanity checks. List of labels per local update step
        aggregate_true_user_data = []
        aggregate_true_user_labels = []

        for user in self.users:
            user.to(**self.setup)
            user.setup = self.setup
            user_data, true_user_data = user.compute_local_updates(server_payload)
            user.to(**self.user_setup)
            user.setup = self.user_setup

            if return_true_user_data:
                aggregate_true_user_data += [true_user_data["data"].cpu()]
                if true_user_data["labels"] is not None:
                    aggregate_true_user_labels += [true_user_data["labels"].cpu()]
            torch._foreach_sub_(user_data["gradients"], aggregate_updates)
            torch._foreach_add_(aggregate_updates, user_data["gradients"], alpha=1 / self.num_users)

            if user_data["buffers"] is not None:
                torch._foreach_sub_(user_data["buffers"], aggregate_buffers)
                torch._foreach_add_(aggregate_buffers, buffer_to_server, alpha=1 / self.num_users)
            if user_data["metadata"]["labels"] is not None:
                aggregate_labels.append(user_data["metadata"]["labels"])
            if params := user_data["metadata"]["local_hyperparams"] is not None:
                if params["labels"] is not None:
                    aggregate_label_lists += [l.cpu() for l in user_data["metadata"]["local_hyperparams"]["labels"]]

        shared_data = dict(
            gradients=aggregate_updates,
            buffers=aggregate_buffers if self.provide_buffers else None,
            metadata=dict(
                num_data_points=self.num_data_points * len(self.users) if self.provide_num_data_points else None,
                labels=torch.cat(aggregate_labels).sort()[0] if self.provide_labels else None,
                num_users=self.num_users,
                local_hyperparams=dict(
                    lr=self.local_learning_rate,
                    steps=self.num_local_updates,
                    data_per_step=self.num_data_per_local_update_step,
                    labels=aggregate_label_lists,
                )
                if self.provide_local_hyperparams
                else None,
            ),
        )

        if return_true_user_data:
            aggregate_true_user_data = torch.cat(aggregate_true_user_data, dim=0)
            aggregate_true_user_labels = (
                torch.cat(aggregate_true_user_labels) if len(aggregate_true_user_labels) > 0 else None
            )

            true_user_data = dict(
                data=aggregate_true_user_data, labels=aggregate_true_user_labels, buffers=aggregate_buffers
            )

        return shared_data, true_user_data


def _retrieve_forwarded_modules_with_order(model: torch.nn.Module, input):
    forwarded_modules: List[torch.nn.Module] = []
    handlers = []

    def make_forward_hook():
        def forward_hook(module, input, output):
            forwarded_modules.append(module)
        return forward_hook

    def register_hook(module: torch.nn.Module):
        if len(list(module.parameters(recurse=False))) > 0:
            handler = module.register_forward_hook(make_forward_hook())
            handlers.append(handler)

        children = list(module.children())
        for child in children:
            register_hook(child)

    register_hook(model)

    if isinstance(input, dict):
        model(**input)
    else:
        model(input)

    [h.remove() for h in handlers]
    return forwarded_modules