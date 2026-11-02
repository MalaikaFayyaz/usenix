"""Various regularizers that can be re-used for multiple attacks."""

import copy
from collections import namedtuple
from typing import Optional, Dict, List
import warnings

import torch


from .deepinversion import DeepInversionFeatureHook


class _LinearFeatureHook:
    """Hook to retrieve input to given module."""

    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        input_features = input[0]
        self.features = input_features

    def close(self):
        self.hook.remove()


class FeatureRegularization(torch.nn.Module):
    """Feature regularization implemented for the last linear layer at the end."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, shared_data, labels, *args, **kwargs):
        self.measured_features = []
        for user_data in shared_data:
            # Assume last two gradient vector entries are weight and bias:
            weights = user_data["gradients"][-2]
            bias = user_data["gradients"][-1]
            grads_fc_debiased = weights / bias[:, None]
            features_per_label = []
            for label in labels:
                if bias[label] != 0:
                    features_per_label.append(grads_fc_debiased[label])
                else:
                    features_per_label.append(torch.zeros_like(grads_fc_debiased[0]))
            self.measured_features.append(torch.stack(features_per_label))

        self.refs = [None for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                # Keep only the last linear layer here:
                if isinstance(module, torch.nn.Linear):
                    self.refs[idx] = _LinearFeatureHook(module)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for ref, measured_val in zip(self.refs, self.measured_features):
            regularization_value += (ref.features - measured_val).pow(2).mean()
        return regularization_value * self.scale

    def __repr__(self):
        return f"Feature space regularization, scale={self.scale}"


class LinearLayerRegularization(torch.nn.Module):
    """Linear layer regularization implemented for arbitrary linear layers. WIP Example."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, gradient_data, *args, **kwargs):
        self.measured_features = []
        self.refs = [list() for model in models]

        for idx, (model, user_data) in enumerate(zip(models, shared_data)):
            # 1) Find linear layers:
            linear_layers = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_layers.append(name)
                    self.refs[idx].append(_LinearFeatureHook(module))
            named_grads = {name: g for (g, (name, param)) in zip(user_data["gradients"], model.named_parameters())}

            # 2) Check features
            features = []
            for linear_layer in linear_layers:
                weights = named_grads[linear_layer + ".weight"]
                bias = named_grads[linear_layer + ".bias"]
                grads_fc_debiased = (weights / bias[:, None]).mean(dim=0)  # At some point todo: Make this smarter
                features.append(grads_fc_debiased)
            self.measured_features.append(features)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for model_ref, data_ref in zip(self.refs, self.measured_features):
            for linear_layer, data in zip(model_ref, data_ref):
                regularization_value += (linear_layer.features.mean(dim=0) - data).pow(2).sum()

    def __repr__(self):
        return f"Feature space regularization, scale={self.scale}"


class TotalVariation(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions.
    Optionally also Color TV based on its last three dimensions.

    The value of this regularization is scaled by 1/sqrt(M*N) times the given scale."""

    def __init__(self, setup, scale=0.1, inner_exp=1, outer_exp=1, double_opponents=False, eps=1e-8, start_iter=0, default_value=100.0):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
        Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.inner_exp = inner_exp
        self.outer_exp = outer_exp
        self.eps = eps
        self.double_opponents = double_opponents
        self.default_value = default_value
        
        # Code Adaption for `High-Fidelity Gradient Inversion`.
        # Z. Ye, W. Luo, Q. Zhou, and Y. Tang, “High-Fidelity Gradient Inversion in Distributed Learning,” in AAAI Conference on Artificial Intelligence, 2024, pp. 19983–19991.
        self.start_iter = start_iter
        self.iter = 0

        grad_weight = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], **setup).unsqueeze(0).unsqueeze(1)
        grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
        self.groups = 6 if self.double_opponents else 3
        grad_weight = torch.cat([grad_weight] * self.groups, 0)

        self.register_buffer("weight", grad_weight)

    def initialize(self, models, *args, **kwargs):
        # Code Adaption for `High-Fidelity Gradient Inversion`
        self.iter = 0

    def forward(self, tensor, *args, **kwargs):
        """Use a convolution-based approach."""
        # Code Adaption for `High-Fidelity Gradient Inversion`.
        # Z. Ye, W. Luo, Q. Zhou, and Y. Tang, “High-Fidelity Gradient Inversion in Distributed Learning,” in AAAI Conference on Artificial Intelligence, 2024, pp. 19983–19991.
        self.iter = self.iter + 1
        if self.iter < self.start_iter:
            return tensor.new_tensor(self.default_value)
        
        if self.double_opponents:
            tensor = torch.cat(
                [
                    tensor,
                    tensor[:, 0:1, :, :] - tensor[:, 1:2, :, :],
                    tensor[:, 0:1, :, :] - tensor[:, 2:3, :, :],
                    tensor[:, 1:2, :, :] - tensor[:, 2:3, :, :],
                ],
                dim=1,
            )
        diffs = torch.nn.functional.conv2d(
            tensor, self.weight, None, stride=1, padding=1, dilation=1, groups=self.groups
        )
        squares = (diffs.abs() + self.eps).pow(self.inner_exp)
        squared_sums = (squares[:, 0::2] + squares[:, 1::2]).pow(self.outer_exp)
        return squared_sums.mean() * self.scale

    def __repr__(self):
        return (
            f"Total Variation, scale={self.scale}. p={self.inner_exp} q={self.outer_exp}. "
            f"{'Color TV: double oppponents' if self.double_opponents else ''}"
        )


class OrthogonalityRegularization(torch.nn.Module):
    """This is the orthogonality regularizer described Qian et al.,

    "MINIMAL CONDITIONS ANALYSIS OF GRADIENT-BASED RECONSTRUCTION IN FEDERATED LEARNING"
    """

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        if tensor.shape[0] == 1:
            return 0
        else:
            B = tensor.shape[0]
            full_products = (tensor.unsqueeze(0) * tensor.unsqueeze(1)).pow(2).view(B, B, -1).mean(dim=2)
            idx = torch.arange(0, B, out=torch.LongTensor())
            full_products[idx, idx] = 0
            return full_products.sum()

    def __repr__(self):
        return f"Input Orthogonality, scale={self.scale}"


class NormRegularization(torch.nn.Module):
    """Implement basic norm-based regularization, e.g. an L2 penalty."""

    def __init__(self, setup, scale=0.1, pnorm=2.0):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.pnorm = pnorm

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        return 1 / self.pnorm * tensor.pow(self.pnorm).mean() * self.scale

    def __repr__(self):
        return f"Input L^p norm regularization, scale={self.scale}, p={self.pnorm}"


class DeepInversion(torch.nn.Module):
    """Implement a DeepInversion based regularization as proposed in DeepInversion as used for reconstruction in
    Yin et al, "See through Gradients: Image Batch Recovery via GradInversion".
    """

    def __init__(self, setup, scale=0.1, first_bn_multiplier=10):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.first_bn_multiplier = first_bn_multiplier

    def initialize(self, models, *args, **kwargs):
        """Initialize forward hooks."""
        self.losses = [list() for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.losses[idx].append(DeepInversionFeatureHook(module))

    def forward(self, tensor, *args, **kwargs):
        rescale = [self.first_bn_multiplier] + [1.0 for _ in range(len(self.losses[0]) - 1)]
        feature_reg = 0
        for loss in self.losses:
            feature_reg += sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss)])
        return self.scale * feature_reg

    def __repr__(self):
        return f"Deep Inversion Regularization (matching batch norms), scale={self.scale}, first-bn-mult={self.first_bn_multiplier}"


class LossRegularization(torch.nn.Module):
    """Implementation of Loss Regularization, used by Hybrid Inversion"""

    def __init__(self, setup, scale=0.1, is_joint_optimize=False):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.is_joint_optimize = is_joint_optimize

        # Essential properties for regularization forwarding.
        # Notes:
        #   We do not use hook like FeatureRegularization to implement this feature. Because
        #   LossRegularization not only need to regularize model used for optimizing dummy input,
        #   but also need to regularize model that it can retrieve.
        self.models: [torch.nn.Module] = []
        self.losses: [torch.Tensor] = []
        self.loss_fn: Optional[torch.nn.Module] = None
        self.labels: Optional[torch.Tensor] = None

    def initialize(self, models, shared_datas, labels, *args, **kwargs):
        # reset state
        self.models.clear()
        self.losses.clear()
        self.loss_fn = None
        self.labels = None

        # record labels, raise Warning if not given.
        if labels is not None:
            self.labels = labels
        else:
            warnings.warn(f"Warning: `{self.__class__.__name__}.initialize`, labels is None, it will "
                          f"lead to ValueError raising when `{self.__class__.__name__}.forward` is called.")

        # record loss function, models, losses
        for model, shared_data in zip(models, shared_datas):
            metadata: dict = shared_data["metadata"]
            if "loss_fn" in metadata.keys() and metadata["loss_fn"] is not None:
                self.loss_fn = metadata["loss_fn"]

            if "loss_infos" not in metadata.keys():
                continue
            LossInfo = namedtuple('LossInfo', ['payload', 'loss'])
            loss_infos: [LossInfo] = metadata["loss_infos"]
            if not self.is_joint_optimize:
                loss_infos = loss_infos[0:1]

            for payload, loss in loss_infos:
                model = copy.deepcopy(model)
                parameters = payload["parameters"]
                buffers = payload["buffers"]
                with torch.no_grad():
                    for param, new_value in zip(model.parameters(), parameters):
                        param.copy_(new_value.to(**self.setup))
                    if buffers is not None:
                        for buffer, new_value in zip(model.buffers(), buffers):
                            buffer.copy_(new_value.to(**self.setup))
                        model.eval()
                    else:
                        for module in model.modules():
                            if hasattr(module, "momentum"):
                                module.momentum = None  # Force recovery without division
                        model.train()

                self.models.append(model)
                self.losses.append(loss)

        if self.loss_fn is None:
            warnings.warn(f"Warning: `{self.__class__.__name__}.initialize`, loss_fn is not given, it will "
                          f"lead to ValueError raising when `{self.__class__.__name__}.forward` is called.")

    def forward(self, tensor, *args, **kwargs):
        # Notes:
        #   Following operation are mainly performed with torch operator, which is mainly to avoid
        #   the precision loss for floating-point calculation.
        self._assert_enable_forwarding(tensor=tensor)
        if len(self.losses) == 0:
            return 0
        if self.scale == 0:
            return 0

        losses_for_each_model: [torch.Tensor] = [
            self.loss_fn(model(tensor), self.labels)
            for model in self.models
        ]
        loss_diff_for_each_model: [torch.Tensor] = [
            torch.abs(candidate_loss.sub_(target_loss))
            for candidate_loss, target_loss in zip(losses_for_each_model, self.losses)
        ]
        loss = torch.sum(torch.stack(loss_diff_for_each_model), dim=0)
        return loss.mul_(self.scale)


    def _assert_enable_forwarding(self, tensor):
        if self.labels is None:
            raise ValueError(f"{self.__class__.__name__}._assert_enable_forwarding, fail to forward, "
                             f"because self.labels is None")

        if self.loss_fn is None:
            raise ValueError(f"{self.__class__.__name__}._assert_enable_forwarding, fail to forward, "
                             f"because self.loss_fn is None")

        if tensor.size(0) != self.labels.size(0):
            input_amount = tensor.size(0)
            label_amount = self.labels.size(0)
            raise ValueError(f"{self.__class__.__name__}._assert_enable_forwarding, fail to forward, "
                             f"because input amount ({input_amount}) not equal to labels amount ({label_amount})")

    def __repr__(self):
        return f"Loss regularization, scale={self.scale}, is_joint_optimize={self.is_joint_optimize}"


class NormalizationRegularization(torch.nn.Module):
    # [WIP] Wait for Checking
    def __init__(self, setup, scale=0.1, rank=2.0, lower_bound=0.0, upper_bound=1.0, need_denormalized=False):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.rank = rank
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.need_denormalized = need_denormalized
        
        self.mean_tensor: Optional[torch.Tensor] = None
        self.std_tensor: Optional[torch.Tensor] = None
        
    def initialize(self, models, shared_data, labels, *args, **kwargs):
        if not self.need_denormalized:
            return
        
        self.normalize_value_by_key.clear()
        data_cfg = shared_data["metadata"]["case_cfg"]["data"]
        self.mean = torch.as_tensor(list(data_cfg["mean"])).to(**self.setup)
        self.std = torch.as_tensor(list(data_cfg["std"])).to(**self.setup)

    def forward(self, tensor, *args, **kwargs):
        if self.need_denormalized:
            tensor = torch.add(torch.mul(tensor, self.mean), self.std)
        
        losses = torch.div(torch.subtract(tensor, self.lower_bound), torch.subtract(self.upper_bound, self.lower_bound))
        return torch.max(losses.view(-1)).pow(self.rank)
        
    def __repr__(self):
        return f"NormalizationRegularization, scale={self.scale}, rank={self.rank} lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, need_denormalized={self.need_denormalized}"


regularizer_lookup = dict(
    total_variation=TotalVariation,
    orthogonality=OrthogonalityRegularization,
    norm=NormRegularization,
    deep_inversion=DeepInversion,
    features=FeatureRegularization,
    loss=LossRegularization,
    normalization=NormalizationRegularization,
)
