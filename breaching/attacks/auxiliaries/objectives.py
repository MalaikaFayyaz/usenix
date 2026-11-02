"""Various objective functions that can be re-used for multiple attacks."""

from functools import reduce
import logging
import math
import re
from typing import List, Dict, Optional
import weakref

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn

from .make_functional import make_functional_with_buffers


class GradientLoss(torch.nn.Module):
    """Super-class to simplify gradient-based objectives."""

    def __init__(self, force_single_steps=False):
        super().__init__()
        self.task_regularization = 0
        self.force_single_steps = force_single_steps # AGIC: Approximate Gradient Inversion Attack on Federated Learning(SRDC'22)
        
        # Weak reference to observe param passed in `self.forward`, enable subclasses to make a further introducing.
        self._model_ref: Optional[weakref.ReferenceType] = None
        self._candidate_ref: Optional[weakref.ReferenceType] = None
        self._labels_ref: Optional[weakref.ReferenceType] = None

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None, rec_model: Optional[nn.Module]=None, true_grad: Optional[List[torch.Tensor]]=None):
        self.loss_fn = loss_fn
        self.local_hyperparams = local_hyperparams
        if self.local_hyperparams is None:
            self._grad_fn = self._grad_fn_single_step
        else:
            self._grad_fn = self._grad_fn_multi_step

        self.cfg_impl = cfg_impl

    def forward(self, model, gradient_data, candidate, labels):
        # Reference for parameters, enabling further utilizing for sub-class in `self.gradient_based_loss`
        self._model_ref = weakref.ref(model)
        self._candidate_ref = weakref.ref(candidate)
        self._labels_ref = weakref.ref(labels)
        
        gradient, task_loss = self._grad_fn(model, candidate, labels)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            objective = self.gradient_based_loss(gradient, gradient_data)
        if self.task_regularization != 0:
            objective += self.task_regularization * task_loss
        
        # Release reference when finish forwarding, avoiding potentially misusing.
        self._model_ref = None
        self._candidate_ref = None
        self._labels_ref = None
        return objective, task_loss.detach()

    def gradient_based_loss(self, gradient_rec, gradient_data):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def _grad_fn_single_step(self, model, candidate, labels):
        """Compute a single gradient."""
        model.zero_grad()
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        gradient = torch.autograd.grad(task_loss, model.parameters(), create_graph=True)
        return gradient, task_loss

    def _grad_fn_multi_step(self, model, candidate, labels):
        """Compute the full graph for multiple local update steps."""
        model.zero_grad()
        func_model, params, buffers = make_functional_with_buffers(model)
        initial_params = [p.clone() for p in params]

        seen_data_idx = 0
        for i in range(self.local_hyperparams["steps"]):
            data = candidate[seen_data_idx: seen_data_idx + self.local_hyperparams["data_per_step"]]
            seen_data_idx += self.local_hyperparams["data_per_step"]
            seen_data_idx = seen_data_idx % candidate.shape[0]
            labels = self.local_hyperparams["labels"][i]
            with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
                task_loss = self.loss_fn(func_model(params, buffers, data), labels)

            step_gradient = torch.autograd.grad(task_loss, params, create_graph=True)

            # Update parameters in graph:
            params = [param - self.local_hyperparams["lr"] * grad for param, grad in zip(params, step_gradient)]
            if self.force_single_steps:
                break

        # Finally return differentiable difference in state:
        gradient = [p_local - p_server for p_local, p_server in zip(params, initial_params)]

        # Return last loss as the "best" task loss
        return gradient, task_loss


class WeightedMixIn():
    """MinIn for Weighted Calculation.
    @Notes:
        Here is indices indication it has supported:
            1. any(['default', 'interleave', 'concat', 'all']): all params.
            2. any(["layer", "layer-wise"]): all params but grouped them with layer-wise.
            3. any(["each5", "each10", ..., "each{k}"]): all params but grouped them continuous-k index.
            
            4. any(["batch5", "batch10", "batch18", ..., "batch{k}"]): randomly choose k params, and just match their gradients.
            5. any(["first5", "first10", ..., "first{k}"]): choose first-k params, and just match their gradients.
            6. any(["last5", "last10", ..., "last{k}"]): choose last-k params, and just match their gradients.
            7. any(["top5", "top10", ..., "top{k}"]): choose top-k params w.r.t. their norm, and just match their gradients.
        
        Here is weights indiction it has supported:
            1. any(['equal', 'equals', 'same']): same weights
            2. "sapag": used by SAPAG attack, "Variance of the Gradient Also Matters: Privacy Leakage from Gradients".
            3. any(["preserver_linear-50", "preserver_linear-2", ..., "preserver_linear-{k}"]), used by AGIC attack, "AGIC: Approximate Gradient Inversion Attack on Federated Learning". 
        
    @Reference:
        J. Xu, C. Hong, J. Huang, L. Y. Chen and J. Decouchant, "AGIC: Approximate Gradient Inversion Attack on Federated Learning," 
        2022 41st International Symposium on Reliable Distributed Systems (SRDS), Vienna, Austria, 2022, pp. 12-22.
    """
    
    __last_regex = "^last[1-9]\d*$"     # only compute last-k params' gradient w.r.t. their order
    __last_not_split_regex = "^last_not_split[1-9]\d*$"     # only compute last-k params' gradient w.r.t. their order, but not split them. 
    __first_regex = "^first[1-9]\d*$"   # only compute first-k params' gradient w.r.t. their order
    __top_regex = "^top[1-9]\d*$"       # only compute top-k params' gradient w.r.t. param's norm
    __each_regex = "^each[1-9]\d*$"     # compute all gradients, but each-k would be in same group.
    __key_regex = "^key:.*$"            # compute only gradients with key matched, you could specified key after ':'.
    __keys_regex = "^keys:.*$"          # compute only gradients any keys matched, you could specified key after ':', and split each key by ', '.
    __batch_regex = "^batch[1-9]\d*$"   # only compute k params with randomly chosen
    
    __preserve_linear_regex = "^preserve_linear-[1-9]\d*$" # used by "AGIC: Approximate Gradient Inversion Attack on Federated Learning"
    
    def make_weighted_reprs(self, weighted_config: Dict) -> str:
        style = weighted_config["style"]
        indices = weighted_config["indices"]
        need_average = weighted_config["need_average"]
        need_proportion_rescale = weighted_config["need_proportion_rescale"]
        need_large_proportion_rescale = weighted_config["need_large_proportion_rescale"]
        need_normalized = weighted_config["need_normalized"]
        need_inverse = weighted_config["need_inverse"]
        return f"[Weighted Required] Status: style={style}, indices={indices}, need_average={need_average}, proportion_rescale={need_proportion_rescale}, need_large_proportion_rescale={need_large_proportion_rescale}, need_normalized={need_normalized}, need_inverse={need_inverse}"
        
    def map_weight_config_to_dict(self, config: Optional[DictConfig]) -> Optional[Dict]:
        if not config:
            return None

        option_by_key: Dict = OmegaConf.to_container(config)
        # set default value to avoid runtime error
        if "need_average" not in option_by_key.keys():
            option_by_key["need_average"] = False
        if "need_proportion_rescale" not in option_by_key.keys():
            option_by_key["need_proportion_rescale"] = False
        if "need_normalized" not in option_by_key.keys():
            option_by_key["need_normalized"] = False
        if "need_large_proportion_rescale" not in option_by_key.keys():
            option_by_key["need_large_proportion_rescale"] = False
        if "need_inverse" not in option_by_key.keys():
            option_by_key["need_inverse"] = False
        if "need_normalized_to_one" not in option_by_key.keys():
            option_by_key["need_normalized_to_one"] = False
        return option_by_key
    
    def _get_nested_indices_to_calculate(self, rec_model: List[nn.Module], indice_style: str) -> List[List[int]]:
        """
        @Notes:
            Indices returned by this function would be arranged as a two-dimension nested array, a.k.a. List[List[int]].
            - First Dimension: indices groups, weights will be assigned for each group.
            - Second Dimension: indices in same group, distance between gradients would be calculated w.r.t. the combination of elem in each group.
            
            The reason why make such nested array is because, param in one single layer may contain "weights" and "bias", and they would be separated
            when calling `model.named_parameters()` and `torch.autograd()`, while some papers would calculate calculate distance between gradients in
            same layer, and assign different weights for different layers.
        """
        
        # [TODO]: Model given by attacker is a list, now we just take the first one.
        rec_model = rec_model[0]
        
        params = [param for _, param in rec_model.named_parameters()]
        indice_style = indice_style.lower()
        
        if indice_style in ['default', 'interleave', 'concat', 'all']:
            indices = torch.arange(len(params)).numpy().tolist()
            indices = [[x] for x in indices]
            return indices
        
        elif indice_style in ["layer", "layer-wise"] and rec_model:
            from ordered_set import OrderedSet
            param_names = [name for name, _ in rec_model.named_parameters()]
            layer_names = []
            for name in param_names:
                identifiers = name.split(".")
                layer_name = reduce(lambda x, y: x + "." + y, identifiers[:-1], identifiers[0])
                layer_names.append(layer_name)
                
            layer_set = OrderedSet(layer_names)
            indices = [[] for _ in range(len(layer_set))]
            for i, name in enumerate(layer_names):
                group_index = layer_set.index(name)
                indices[group_index].append(i)
            return indices
        
        elif re.search(self.__key_regex, indice_style): # "key:layer", "key:attention", ...
            specified_key = indice_style[4:]
            param_names = [name for name, _ in rec_model.named_parameters()]
            chosen_indices = [i for i, name in enumerate(param_names) if specified_key in name]
            return [chosen_indices]

        elif re.search(self.__keys_regex, indice_style): # "keys:layer1, layer2, layer3", "keys:attention1, attention2", ...
            specified_keys = indice_style[5:].split(", ")
            param_names = [name for name, _ in rec_model.named_parameters()]
            chosen_indices = []
            for i, name in enumerate(param_names):
                for key in specified_keys:
                    if key in name:
                        chosen_indices.append(i)
                        break
            return [chosen_indices]

        elif re.search(self.__batch_regex, indice_style): # "batch1", "batch8", ...
            amount = int(indice_style[5:])
            indices = torch.randperm(len(params))[:amount].numpy().tolist()
            indices = [[x] for x in indices]
            return indices
        
        elif re.search(self.__top_regex, indice_style): # "top1", "top10", ...
            _, indices = torch.topk(torch.stack([p.norm() for p in params], dim=0), int(indice_style[3:]))
            indices = [[x] for x in indices.numpy().tolist()]
            return indices
        
        elif re.search(self.__first_regex, indice_style): # "first1", "first10", ...
            amount = int(indice_style[5:])
            indices = torch.arange(len(params))[0: amount]
            indices = [[x] for x in indices.numpy().tolist()]
            return indices
        
        elif re.search(self.__last_regex, indice_style): # "last1", "last10", ...
            amount = int(indice_style[4:])
            indices = torch.arange(len(params))[-1 * amount:]
            indices = [[x] for x in indices.numpy().tolist()]
            return indices
        
        elif re.search(self.__last_not_split_regex, indice_style):  # "last_not_split1", "last_not_split10", ...
            amount = int(indice_style[14:])
            indices = torch.arange(len(params))[-1 * amount:].numpy().tolist()
            return [indices]
        
        elif re.search(self.__each_regex, indice_style): # "each1", "each3", "each10" ...
            amount = int(indice_style[4:])
            group_amount = len(params) // amount
            if amount * group_amount < len(params):
                group_amount = group_amount + 1
            
            indices = [[] for _ in range(group_amount)]
            for i in range(len(params)):
                indices[i // amount].append(i)
            return indices
        
        else:
            logging.warning(f"[Fallback]Fail to match indice style: \"{indice_style}\", return \"all\" by default")
            indices = torch.arange(len(params)).numpy().tolist()
            indices = [[x] for x in indices]
            return indices

    def _get_weights_to_calculate(self, rec_model: List[nn.Module], nested_indices: List[List[int]], weighted_config: Dict, true_grad: Optional[List[torch.Tensor]]=None) -> torch.Tensor:
         # [TODO]: Model given by attacker is a list, now we just take the first one.
        rec_model = rec_model[0]
        
        # retrieve key in weighted_config
        weight_style = weighted_config["style"]
        need_proportion_rescale = weighted_config["need_proportion_rescale"]
        need_normalized = weighted_config["need_normalized"]
        need_large_proportion_rescale = weighted_config["need_large_proportion_rescale"]
        need_inverse = weighted_config["need_inverse"]
        need_normalized_to_one = weighted_config["need_normalized_to_one"]
        
        if len(nested_indices) == 0:
            return torch.Tensor([])
        
        params = [param for _, param in rec_model.named_parameters()]
        weights: torch.Tensor = params[0].new_ones(len(nested_indices))
        weight_style = weight_style.lower()
        if weight_style in ['equal', 'equals', 'same']:
            pass
        
        elif weight_style == "sapag": # Used by SAPAG attack, "Variance of the Gradient Also Matters: Privacy Leakage from Gradients".
            weights = params[0].new_tensor(range(1, len(nested_indices) + 1))
            weights = torch.nn.Tanh()(2. / weights) + 1
        
        elif re.search(self.__preserve_linear_regex, weight_style): # Used by AGIC, like "preserver_linear-50", "preserver_linear-2"
            scale = float(weight_style.split('-')[-1])
            n = len(weights)
            for k in range(1, n):
                weights[k] = 1 + (scale - 1) / (n - 1) * k
            weights[-1] = torch.mean(weights[:-1])
        
        elif weight_style == "gradient_sparsity_positive_correlative":
            sparsities: torch.Tensor = params[0].new_zeros(len(nested_indices))
            for layer, param_indices in enumerate(nested_indices):
                flatten_grad = torch.cat([true_grad[i].view(-1) for i in param_indices])
                is_zero = flatten_grad == 0
                sparsities[layer] = torch.sum(is_zero) / flatten_grad.size()[0]
            
            # calculate for weights, and normalize, ensuring max(weights) == 1.0
            weights = weights + sparsities
            
        elif weight_style == "gradient_sparsity_negative_correlative":
            sparsities: torch.Tensor = params[0].new_zeros(len(nested_indices))
            for layer, param_indices in enumerate(nested_indices):
                flatten_grad = torch.cat([true_grad[i].view(-1) for i in param_indices])
                is_zero = flatten_grad == 0
                sparsities[layer] = torch.sum(is_zero) / flatten_grad.size()[0]
            
            # calculate for weights, and normalize, ensuring max(weights) == 1.0
            weights = weights + (1.0 / (1.0 + sparsities))
        
        else:
            logging.warning(f"Fail to match weight style: \"{weight_style}\", return \"equal\" by default")
        
        if need_proportion_rescale:
            each_layer_param_amounts = [reduce(lambda x, y: x * y, p.size()[1:], p.size()[0]) for p in params]
            param_amount = reduce(lambda x, y: x + y, each_layer_param_amounts[1:], each_layer_param_amounts[0])
            
            for i, (weight, indices) in enumerate(zip(weights, nested_indices)):
                group_params = [params[j] for j in indices]
                group_param_amounts = [reduce(lambda x, y: x * y, p.size()[1:], p.size()[0]) for p in group_params]
                group_param_amount = reduce(lambda x, y: x + y, group_param_amounts[1:], group_param_amounts[0])
                
                scale = group_param_amount / param_amount
                weights[i] = torch.mul(weight, scale)
                
        if need_large_proportion_rescale:
            each_layer_param_amounts = [reduce(lambda x, y: x * y, p.size()[1:], p.size()[0]) for p in params]
            each_group_param_amounts = [
                sum([each_layer_param_amounts[index] for index in indices])
                for indices in nested_indices
            ]
            param_amount = reduce(lambda x, y: x + y, each_layer_param_amounts[1:], each_layer_param_amounts[0])
            proportions = weights.new_tensor([1. + amount / param_amount for amount in each_group_param_amounts])
            weights = torch.mul(weights, proportions)
        
        if need_normalized:
            weights = torch.nn.Softmax()(weights)

        if need_normalized_to_one:
            max_weight = torch.max(weights).item()
            weights = weights / max_weight

        if need_inverse:
            weights = torch.flip(weights, dims=(0,))

        return weights
        

class Euclidean(GradientLoss, WeightedMixIn):
    """Gradient matching based on the euclidean distance of two gradient vectors."""

    def __init__(self, force_single_steps=False, scale=1.0, task_regularization=0.0, weighted_config: Optional[DictConfig] = None, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.task_regularization = task_regularization
        
        # setup weighted config and specify loss function, to speed up the function `graident_based_loss`.
        self.weighted_config: Optional[Dict] = self.map_weight_config_to_dict(weighted_config)
        self.__gradient_fn = self._weighted_euclidean if self.need_weighted else self._euclidean
        self._nested_indices: List[List[int]] = []
        self._weights: torch.Tensor = torch.Tensor([])
        
    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None, rec_model: Optional[nn.Module]=None, true_grad: Optional[List[torch.Tensor]]=None):
        super().initialize(loss_fn, cfg_impl, local_hyperparams, rec_model)
        
        if not self.need_weighted or not rec_model:
            return
        self._nested_indices = self._get_nested_indices_to_calculate(
            rec_model=rec_model,
            indice_style=self.weighted_config["indices"]
        )
        self._weights = self._get_weights_to_calculate(
            rec_model=rec_model,
            nested_indices=self._nested_indices,
            weighted_config=self.weighted_config,
            true_grad=true_grad
        )

    @property
    def need_weighted(self) -> bool:
        return self.weighted_config is not None

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self.__gradient_fn(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        repr_str = f"Euclidean loss with scale={self.scale} and task reg={self.task_regularization}"
        if self.need_weighted:
            repr_str += "\n\t\t" + self.make_weighted_reprs(self.weighted_config) + ","
        return repr_str

    @staticmethod
    @torch.jit.script
    def _euclidean(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).pow(2).sum()
        return 0.5 * objective
    
    def _weighted_euclidean(self, gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]) -> torch.Tensor:
        if not self.need_weighted:
            return self._euclidean(gradient_rec, gradient_data)
        
        losses = []
        for indices, weight in zip(self._nested_indices, self._weights):
            fake_grads = [gradient_rec[i] for i in indices]
            true_grads = [gradient_data[i] for i in indices]
            loss = self._euclidean(fake_grads, true_grads)
            losses.append(torch.mul(loss, weight))
        
        losses = torch.hstack(losses)
        if self.weighted_config["need_average"]:
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class EuclideanTag(GradientLoss):
    """Gradient matching based on the euclidean distance of two gradient vectors plus TAG regularizer

    from Deng et al., "TAG: Gradient Attack on Transformer-based Language Models"
    How to scale each layer is unclear to me based on the paper, so I am recycling decay schemes from
    the InvertingGradients repo.
    """

    def __init__(self, force_single_steps=False, scale=1.0, task_regularization=0.0, tag_scale=0.1, scale_scheme="linear", **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.task_regularization = task_regularization
        self.tag_scale = tag_scale
        self.scale_scheme = scale_scheme
        self.weights = None

    def gradient_based_loss(self, gradient_rec, gradient_data):
        if self.weights is None:
            setup = dict(dtype=gradient_rec[0].dtype, device=gradient_rec[0].device)
            if self.scale_scheme == "linear":
                weights = torch.arange(len(gradient_rec), 0, -1, **setup) / len(gradient_rec)
            elif self.scale_scheme == "exp":
                weights = torch.arange(len(gradient_rec), 0, -1, **setup)
                weights = weights.softmax(dim=0)
                weights = weights / weights[0]
            else:
                weights = gradient_rec[0].new_ones(len(gradient_rec))
        return self._weighted_euclidean_l1(gradient_rec, gradient_data, weights, self.tag_scale) * self.scale

    def __repr__(self):
        return (
            f"Tag loss with scale={self.scale}, weight scheme {self.scale_scheme}, L1 scale {self.tag_scale} "
            f"and task reg={self.task_regularization}"
        )

    @staticmethod
    @torch.jit.script
    def _weighted_euclidean_l1(
        gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], weights: torch.Tensor, tag_scale: float,
    ):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data, weight in zip(gradient_rec, gradient_data, weights):
            objective += (rec - data).pow(2).sum() + tag_scale * weight * (rec - data).abs().sum()
        return 0.5 * objective


class L1Loss(GradientLoss):
    """Gradient matching based on the L1 distance of two gradient vectors."""

    def __init__(self, force_single_steps=False, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._l1loss(gradient_rec, gradient_data) * self.scale

    def __repr__(self):
        return f"L1 loss with scale={self.scale} and task reg={self.task_regularization}"

    @staticmethod
    @torch.jit.script
    def _l1loss(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        objective = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).abs().sum()
        return 0.5 * objective


class CosineSimilarity(GradientLoss, WeightedMixIn):
    """Gradient matching based on cosine similarity of two gradient vectors."""

    def __init__(self, force_single_steps=False, scale=1.0, task_regularization=0.0, dropout=0.0, weighted_config: Optional[DictConfig] = None, need_zero_matching = True,**kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.task_regularization = task_regularization
        self.dropout: float = max(0.0, min(dropout, 1.0))
        self.need_zero_matching = need_zero_matching
        
        # setup weighted config and specify loss function, to speed up the function `graident_based_loss`.
        self.weighted_config: Optional[Dict] = self.map_weight_config_to_dict(weighted_config)
        self.__gradient_fn = self._weighted_cosine_sim if self.need_weighted else self._cosine_sim
        self._nested_indices: List[List[int]] = []
        self._weights: torch.Tensor = torch.Tensor([])

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None, rec_model: Optional[nn.Module]=None, true_grad: Optional[List[torch.Tensor]]=None):
        super().initialize(loss_fn, cfg_impl, local_hyperparams, rec_model)
        
        if not self.need_weighted or not rec_model:
            return
        self._nested_indices = self._get_nested_indices_to_calculate(
            rec_model=rec_model,
            indice_style=self.weighted_config["indices"]
        )
        self._weights = self._get_weights_to_calculate(
            rec_model=rec_model,
            nested_indices=self._nested_indices,
            weighted_config=self.weighted_config,
            true_grad=true_grad
        )

    @property
    def need_weighted(self) -> bool:
        return self.weighted_config is not None

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self.__gradient_fn(gradient_rec, gradient_data, self.dropout) * self.scale
    
    def __repr__(self):
        repr_str = f"Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}."
        if self.need_weighted:
            repr_str += "\n\t\t" + self.make_weighted_reprs(self.weighted_config) + ","
        return repr_str

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], dropout: float, need_zero_matching: bool = True):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        for rec, data in zip(gradient_rec, gradient_data):
            if dropout > 0.0:
                dropout = max(0.0, min(dropout, 1.0))
                dropout_mask = torch.rand_like(rec) >= dropout
                rec = torch.mul(rec, dropout_mask)
                data = torch.mul(data, dropout_mask)
            if not need_zero_matching:
                not_zero_mask = torch.rand_like(data) != 0
                rec = torch.mul(rec, not_zero_mask)
                data = torch.mul(data, not_zero_mask)
            
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum()
            data_norm += data.pow(2).sum()

        objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        return objective
    
    def _weighted_cosine_sim(self, gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], dropout: float, need_zero_matching: bool = True) -> torch.Tensor:
        if not self.need_weighted:
            return self._cosine_sim(gradient_rec, gradient_data)
        
        losses = []
        for indices, weight in zip(self._nested_indices, self._weights):
            fake_grads = [gradient_rec[i] for i in indices]
            true_grads = [gradient_data[i] for i in indices]
            loss = self._cosine_sim(fake_grads, true_grads, dropout=dropout, need_zero_matching=need_zero_matching)
            losses.append(torch.mul(loss, weight))
        
        losses = torch.hstack(losses)
        if self.weighted_config["need_average"]:
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class AngularSimilarity(CosineSimilarity):
    """Gradient matching based on angular similarity of two gradient vectors.

    This is basically a more linear cosine similarity."""

    def __init__(self, force_single_steps=False, scale=1.0, task_regularization=0.0, fudge_factor=1e-7, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.task_regularization = task_regularization
        self.fudge_factor = 1e-7

    def gradient_based_loss(self, gradient_rec, gradient_data):
        cosine = 1 - self._cosine_sim(gradient_rec, gradient_data)
        angle = torch.acos(cosine.clamp(min=-1 + self.fudge_factor, max=1 - self.fudge_factor))

        return angle / torch.pi * self.scale

    def __repr__(self):
        return f"Angular Similarity with scale={self.scale} and task reg={self.task_regularization}"


class MaskedCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors.
    All positions that are zero in the data gradient are masked.
    """

    def __init__(self, force_single_steps=False, scale=1.0, mask_value=1e-6, task_regularization=0.0, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.mask_value = 1e-6
        self.task_regularization = task_regularization

    def __repr__(self):
        return f"Masked Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}. Mask val={self.mask_value}"

    def gradient_based_loss(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            mask = data.abs() > self.mask_value
            scalar_product += (rec * data * mask).sum()
            rec_norm += (rec * mask).pow(2).sum()
            data_norm += (data * mask).pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class FastCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors.
    No gradient flows through the normalization."""

    def __init__(self, force_single_steps=False, scale=1.0, task_regularization=0.0, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.task_regularization = task_regularization

    def gradient_based_loss(self, gradient_rec, gradient_data):
        return self._cosine_sim(gradient_rec, gradient_data) * self.scale

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)

        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.detach().pow(2).sum()
            data_norm += data.detach().pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective

    def __repr__(self):
        return f"Fast Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}"


class PearlmutterEuclidean(torch.nn.Module):
    """Use a first-order approximation of \nabla_x \nabla_g instead of the correct autograd value."""

    def __init__(
        self,
        scale=1.0,
        eps=1e-3,
        level_gradients=False,
        fudge_factor=1e-6,
        task_regularization=0.0,
        implementation="forward",
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.task_regularization = task_regularization

        self.eps = eps
        self.level_gradients = level_gradients
        self.fudge_factor = fudge_factor
        self.implementation = implementation

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None, rec_model: Optional[nn.Module]=None, true_grad: Optional[List[torch.Tensor]]=None):
        self.loss_fn = loss_fn
        self.local_hyperparams = local_hyperparams
        if self.local_hyperparams is not None:
            raise ValueError("This loss is only implemented for local gradients so far.")

        self.cfg_impl = cfg_impl
        if self.implementation == "forward":
            self._forward_impl = self._forward_differences
        elif self.implementation == "backward":
            self._forward_impl = self._backward_differences
        elif self.implementation == "central":
            self._forward_impl = self._central_differences
        elif self.implementation == "upwind":
            self._forward_impl = self._upwind_differences
        else:
            raise ValueError(f"Invalid finite difference implementation {self.implementation} given.")

    def __repr__(self):
        return (
            f"Pearlmutter-type Finite Differences Loss with scale={self.scale} and task reg={self.task_regularization}."
            f"Finite Difference Eps: {self.eps}. Level gradients: {self.level_gradients}. "
            f"{f'Fudge-factor: {self.fudge_factor}' if self.level_gradients else ''}"
        )

    def forward(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        model.zero_grad()
        # Keep original parameters
        original_parameters = [p.detach().clone() for p in model.parameters()]
        # Compute derivative and accumulate into candidate.grad
        objective_value, task_loss = self._forward_impl(model, gradient_data, candidate, labels)

        # Unpatch model:# Want to do this faster, somehow
        for param, original_param in zip(model.parameters(), original_parameters):
            param.data.copy_(original_param)
        return objective_value, task_loss

    def _forward_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)
        if self.level_gradients:
            # Only a good idea if normalize_gradients=True
            grad_norm = torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
            torch._foreach_div_(gradients, max(grad_norm, self.fudge_factor))

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
        # Patch model and compute loss at offset vector:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_task_loss = self.loss_fn(model(candidate), labels)
        (dLv_dx,) = torch.autograd.grad(offset_task_loss, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLv_dx - dLdx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _backward_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)
        if self.level_gradients:
            # Only a good idea if normalize_gradients=True
            grad_norm = torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
            torch._foreach_div_(gradients, max(grad_norm, self.fudge_factor))

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
        # Patch model and compute loss at offset vector:
        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_task_loss = self.loss_fn(model(candidate), labels)
        (dLv_dx,) = torch.autograd.grad(offset_task_loss, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLdx - dLv_dx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _central_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()

        # Patch model and compute loss at offset vectors:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=0.5 * eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_plus = self.loss_fn(model(candidate), labels)
        (dLvp_dx,) = torch.autograd.grad(offset_plus, (candidate,), create_graph=False)

        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_minus = self.loss_fn(model(candidate), labels)
        (dLvm_dx,) = torch.autograd.grad(offset_minus, (candidate,), create_graph=False)

        # Compute finite difference approximation
        candidate.grad += (dLvp_dx - dLvm_dx) / eps_n * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _upwind_differences(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)

        # Compute first-order gradients of objective
        objective_value, first_order_grad = self._compute_objective_and_first_order(candidate, gradients, gradient_data)
        # Adapts eps to different block strengths
        eps_n = self.eps / torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()

        # Patch model and compute loss at offset vectors:
        torch._foreach_add_(list(model.parameters()), first_order_grad, alpha=0.5 * eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_plus = self.loss_fn(model(candidate), labels)
        (dLvp_dx,) = torch.autograd.grad(offset_plus, (candidate,), create_graph=False)

        torch._foreach_sub_(list(model.parameters()), first_order_grad, alpha=eps_n)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_minus = self.loss_fn(model(candidate), labels)
        (dLvm_dx,) = torch.autograd.grad(offset_minus, (candidate,), create_graph=False)

        # Compute both finite differences
        Dp = (dLvp_dx - dLdx) / eps_n
        Dm = (dLdx - dLvm_dx) / eps_n
        # Upwind based on dLdx
        # Compute finite difference approximation
        candidate.grad += (torch.max(dLdx, 0)[0] * Dm + torch.min(dLdx, 0)[0] * Dp) * self.scale
        # Add task loss
        candidate.grad += self.task_regularization * dLdx

        return objective_value, task_loss

    def _compute_objective_and_first_order(self, candidate, gradients, gradient_data):
        residuals = torch._foreach_sub(gradients, gradient_data)  # Save one copy of the gradient list here ?
        # Gradients have already been populated. Make sure not to kill them later on.
        with torch.no_grad():
            with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
                objective_value = 0.5 * self.scale * torch.stack([r.detach().pow(2).sum() for r in residuals]).sum()
        return objective_value, residuals


class PearlmutterCosine(PearlmutterEuclidean):
    """Use a first-order approximation of \nabla_x \nabla_g instead of the correct autograd value."""

    def _compute_objective_and_first_order(self, candidate, gradients, gradient_data):
        with torch.no_grad():
            scalar_product, grad_norm, data_norm = self._cosine_sim_components(gradients, gradient_data)

        first_order_cosine = torch._foreach_div(gradient_data, -grad_norm * data_norm)
        torch._foreach_sub_(first_order_cosine, gradients, alpha=-scalar_product / (grad_norm.pow(3) * data_norm))

        objective_value = self.scale * (1 - scalar_product / (grad_norm * data_norm))
        return objective_value, first_order_cosine

    @staticmethod
    @torch.jit.script
    def _cosine_sim_components(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        scalar_product = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)
        rec_norm = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)
        data_norm = torch.tensor(0, device=gradient_rec[0].device, dtype=gradient_rec[0].dtype)

        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.detach().pow(2).sum()
            data_norm += data.detach().pow(2).sum()

        return scalar_product, rec_norm.sqrt(), data_norm.sqrt()


class Wasserstein(GradientLoss):
    """Wasserstein Distance.
    @Reference:
        [1] H. Ren, et. al., "GRNN: Generative Regression Neural Network—A Data Leakage Attack for Federated Learning", ACM Trans. Intell. Syst. Technol., May 2022.
        [2] https://github.com/Rand2AI/GRNN/blob/main/WassersteinDistance/wd.py#L9
    """
    def __init__(self, force_single_steps=False, scale=1.0, rank=2, task_regularization=0.0, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.rank = rank
        self.task_regularization = task_regularization

    def __repr__(self):
        return f"Wasserstein Distance with scale={self.scale}, rank={self.rank}, task reg={self.task_regularization}"

    def gradient_based_loss(self, gradient_rec, gradient_data):
        "https://github.com/Rand2AI/GRNN/blob/main/WassersteinDistance/wd.py#L9"
        fake_grad = torch.cat([grad.view(-1) for grad in gradient_rec])
        true_grad = torch.cat([grad.view(-1) for grad in gradient_data])
        value = torch.abs(torch.sub(fake_grad, true_grad))
        value = torch.pow(torch.sum(torch.pow(value, self.rank)), 1. / self.rank)
        return torch.mul(value, self.scale)


class KernelScalingNorm(GradientLoss, WeightedMixIn):
    """Norm Scaling by Kernel Function, reproduced version.
    @Reference:
        Y. Wang et al., “Variance of the Gradient Also Matters: Privacy Leakage from Gradients,” in International 
        Joint Conference on Neural Networks, Jul. 2022, pp. 1–8. doi: 10.1109/IJCNN55064.2022.9892665.
    """
    def __init__(self, force_single_steps=False, scale=1.0, rank=2, task_regularization=0.0, need_var_scale=True, weighted_config: Optional[DictConfig] = None, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.rank = rank
        self.task_regularization = task_regularization
        self.need_var_scale = need_var_scale
        
        # setup weighted config and specify loss function, to speed up the function `graident_based_loss`.
        self.weighted_config: Optional[Dict] = self.map_weight_config_to_dict(weighted_config)
        self.__gradient_fn = self.weighted_loss if self.need_weighted else self.loss
        self._nested_indices: List[List[int]] = []
        self._weights: torch.Tensor = torch.Tensor([])
        
    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None, rec_model: Optional[nn.Module]=None, true_grad: Optional[List[torch.Tensor]]=None):
        super().initialize(loss_fn, cfg_impl, local_hyperparams, rec_model)
        
        if not self.need_weighted or not rec_model:
            return
        self._nested_indices = self._get_nested_indices_to_calculate(
            rec_model=rec_model,
            indice_style=self.weighted_config["indices"]
        )
        self._weights = self._get_weights_to_calculate(
            rec_model=rec_model,
            nested_indices=self._nested_indices,
            weighted_config=self.weighted_config,
            true_grad=true_grad
        )

    @property
    def need_weighted(self) -> bool:
        return self.weighted_config is not None

    def __repr__(self):
        repr_str = f"Kernel Scaling Norm with scale={self.scale}, rank={self.rank}, task reg={self.task_regularization}, need_var_scale={self.need_var_scale}"
        if self.need_weighted:
            repr_str += "\n\t\t" + self.make_weighted_reprs(self.weighted_config) + ","
        return repr_str

    def gradient_based_loss(self, gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        loss = self.__gradient_fn(
            gradient_rec=gradient_rec, 
            gradient_data=gradient_data,
            rank=self.rank,
            need_var_scale=self.need_var_scale
        )
        
        return torch.mul(loss, self.scale)
    
    @staticmethod
    @torch.jit.script
    def loss(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], rank: float, need_var_scale: bool) -> torch.Tensor:
        fake_grad = torch.cat([grad.view(-1) for grad in gradient_rec])
        true_grad = torch.cat([grad.view(-1) for grad in gradient_data])
        value = torch.abs(torch.sub(fake_grad, true_grad))
        value = torch.pow(torch.sum(torch.pow(value, rank)), 1. / rank)
        
        if need_var_scale:
            true_grad_var = torch.var(true_grad)
            value = torch.div(value, true_grad_var)
            
        return torch.sub(1., torch.exp(torch.mul(-1, value)))
    
    def weighted_loss(self, gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], rank: float, need_var_scale: bool) -> torch.Tensor:
        if not self.need_weighted:
            return self.loss(gradient_rec, gradient_data, rank, need_var_scale)
        
        losses = []
        for indices, weight in zip(self._nested_indices, self._weights):
            fake_grads = [gradient_rec[i] for i in indices]
            true_grads = [gradient_data[i] for i in indices]
            loss = self.loss(fake_grads, true_grads, rank, need_var_scale)
            losses.append(torch.mul(loss, weight))
        
        losses = torch.hstack(losses)
        if self.weighted_config["need_average"]:
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class HybridInversion(GradientLoss, WeightedMixIn):
    "Hybrid Inversion: Norm Scaling by Gradient Value, and Cosine Similarity."
    def __init__(self, force_single_steps=False, scale=1.0, rank=2, task_regularization=0.0, dropout=0.0, is_norm_mean=True, weighted_config: Optional[DictConfig] = None, is_matching_zero = True, is_cosine_matching_zero = True, is_norm_weight_adapted = False, max_norm_weight_ratio=0.1, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.rank = rank
        self.task_regularization = task_regularization
        self.dropout = dropout
        
        # measure setting
        self.is_matching_zero = is_matching_zero
        self.is_cosine_matching_zero = is_cosine_matching_zero
        
        # norm weight setting
        self.is_norm_mean = is_norm_mean
        self.is_norm_weight_adapted = is_norm_weight_adapted
        self.max_norm_weight_ratio = max_norm_weight_ratio
        
        # setup weighted config and specify loss function, to speed up the function `graident_based_loss`.
        self.weighted_config: Optional[Dict] = self.map_weight_config_to_dict(weighted_config)
        self.__gradient_fn = self.weighted_loss if self.need_weighted else self.loss
        self._nested_indices: List[List[int]] = []
        self._weights: torch.Tensor = torch.Tensor([])
        
    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None, rec_model: Optional[nn.Module]=None, true_grad: Optional[List[torch.Tensor]]=None):
        super().initialize(loss_fn, cfg_impl, local_hyperparams, rec_model)
        
        if not self.need_weighted or not rec_model:
            return
        self._nested_indices = self._get_nested_indices_to_calculate(
            rec_model=rec_model,
            indice_style=self.weighted_config["indices"]
        )
        self._weights = self._get_weights_to_calculate(
            rec_model=rec_model,
            nested_indices=self._nested_indices,
            weighted_config=self.weighted_config,
            true_grad=true_grad
        )

    @property
    def need_weighted(self) -> bool:
        return self.weighted_config is not None

    def __repr__(self):
        repr_str = f"Hybrid Inversion: Cosine Distance with Gradient Value(GV) Scaling Norm, scale={self.scale}, rank={self.rank}, task reg={self.task_regularization}, dropout={self.dropout}, is_norm_matching_zero={self.is_matching_zero}, is_cosine_matching_zero={self.is_cosine_matching_zero}, is_norm_mean={self.is_norm_mean}, is_norm_weight_adapted={self.is_norm_weight_adapted}, max_norm_weight_ratio={self.max_norm_weight_ratio}"
        if self.need_weighted:
            repr_str += "\n\t\t" + self.make_weighted_reprs(self.weighted_config) + ","
        return repr_str

    def gradient_based_loss(self, gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor]):
        loss = self.__gradient_fn(
            gradient_rec=gradient_rec, 
            gradient_data=gradient_data,
            rank=self.rank,
            dropout=self.dropout,
            is_norm_mean=self.is_norm_mean,
            is_matching_zero=self.is_matching_zero,
            is_cosine_matching_zero=self.is_cosine_matching_zero,
            is_norm_weight_adapted=self.is_norm_weight_adapted,
            max_norm_weight_ratio=self.max_norm_weight_ratio
        )
        
        return torch.mul(loss, self.scale)
    
    @staticmethod
    @torch.jit.script
    def loss(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], rank: float, dropout: float, is_norm_mean: bool, is_matching_zero: bool, is_cosine_matching_zero: bool, is_norm_weight_adapted: bool, max_norm_weight_ratio: float, norm_weight: float = 1.0) -> torch.Tensor:
        # 1. Convert two gradients into 1D-tensor
        fake_grad = torch.cat([grad.view(-1) for grad in gradient_rec])
        true_grad = torch.cat([grad.view(-1) for grad in gradient_data])
        
        # 2. Introduce dropout to tensor with masking
        # @Ref: Z. Ye, W. Luo, Q. Zhou, and Y. Tang, “High-Fidelity Gradient Inversion in Distributed Learning,” in AAAI Conference on Artificial Intelligence, 2024, pp. 19983–19991.
        dropout = max(0.0, min(dropout, 1.0))
        mask = torch.rand_like(fake_grad) >= dropout
        fake_grad = torch.mul(fake_grad, mask)
        true_grad = torch.mul(true_grad, mask)
        
        # 3. Calculate Gradient Value Normalized Norm
        # @Notes: Add bias 1.0 to avoid overflow as well as the case that grad < 1.0, which will lead to a controversial effect, and also Division with amount of Gradients.
        tmp_fake_g = fake_grad.clone()
        tmp_true_g = true_grad.clone()
        if not is_matching_zero:
            not_zero_mask = true_grad.clone().detach() != 0
            tmp_true_g = torch.mul(tmp_true_g, not_zero_mask)
            tmp_fake_g = torch.mul(tmp_fake_g, not_zero_mask)
            norm_grad_amount = torch.sum(not_zero_mask)
        else:
            norm_grad_amount = torch.as_tensor(tmp_fake_g.size()[0])
            
        abs_diff = torch.abs(torch.sub(tmp_fake_g, tmp_true_g))
        norm_value = torch.div(abs_diff, torch.abs(tmp_true_g) + 1.0)
        norm_value = torch.pow(torch.sum(torch.pow(norm_value, rank)), 1. / rank)
            
        # 4. Calculate Cosine Similarity
        scalar_product, rec_norm, data_norm = torch.as_tensor(0.0).to(fake_grad.device), torch.as_tensor(0.0).to(fake_grad.device), torch.as_tensor(0.0).to(fake_grad.device)
        for rec, data in zip(gradient_rec, gradient_data):
            if not is_cosine_matching_zero:
                not_zero_mask = data.clone().detach() != 0
                rec = torch.mul(rec, not_zero_mask)
                data = torch.mul(data, not_zero_mask)
            
            layer_mask = torch.rand_like(data) >= dropout
            scalar_product += (rec * data * layer_mask).sum()
            rec_norm += (rec * layer_mask).pow(2).sum()
            data_norm += (data * layer_mask).pow(2).sum()
        
        cosine_value = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()
        
        loss = cosine_value
        if is_norm_mean:
            norm_value = torch.div(norm_value, norm_grad_amount)
        elif is_norm_weight_adapted:
            while torch.div(norm_value, cosine_value).item() > max_norm_weight_ratio:
                norm_value = torch.mul(norm_value, 0.1)
        loss = cosine_value + norm_value * norm_weight
        return loss
    
    def weighted_loss(self, gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], rank: float, dropout: float, is_norm_mean: bool, is_matching_zero: bool, is_cosine_matching_zero: bool, is_norm_weight_adapted: bool, max_norm_weight_ratio: float) -> torch.Tensor:
        if not self.need_weighted:
            return self.loss(gradient_rec, gradient_data, rank, dropout)
        
        losses = []
        for indices, weight in zip(self._nested_indices, self._weights):
            fake_grads = [gradient_rec[i] for i in indices]
            true_grads = [gradient_data[i] for i in indices]
            loss = self.loss(
                fake_grads, 
                true_grads, 
                rank, 
                dropout, 
                is_norm_mean, 
                is_matching_zero, 
                is_cosine_matching_zero, 
                is_norm_weight_adapted,
                max_norm_weight_ratio,
                norm_weight=weight
            )
            losses.append(loss)
        
        losses = torch.hstack(losses)
        if self.weighted_config["need_average"]:
            return torch.mean(losses)
        else:
            return torch.sum(losses)


class DynaLayerRandPickedCosineSimilarity(GradientLoss):
    "Z. Ye, W. Luo, Q. Zhou, and Y. Tang, “High-Fidelity Gradient Inversion in Distributed Learning,” in AAAI Conference on Artificial Intelligence, 2024, pp. 19983–19991."

    def __init__(self, force_single_steps=False, scale=1.0, start=0, min_start=0, broken_tail=0, peroid_Add10=1000, add10=10, min_weight_scale=0, max_weight_scale=1, task_regularization=0.0, dropout=0.3, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.start = start
        self.min_start = min_start
        self.broken_tail = broken_tail
        self.peroid_Add10 = peroid_Add10
        self.add10 = add10
        self.min_weight_scale = min_weight_scale
        self.max_weight_scale = max_weight_scale
        self.dropout = max(0.0, min(dropout, 1.0))
        
        self.iter = 0
        

    def __repr__(self):
        return f"High Fidenlity Inversion: Masked Cosine Similarity with scale={self.scale} and task reg={self.task_regularization}. Start layer={self.start} and broken tail={self.broken_tail}"

    def gradient_based_loss(self, gradient_rec, gradient_data):
        # ResNet-18:
        #   start: 50
        #   min_start: 15
        #   peroid_Add10: 500
        # 
        # 1. Weight will be an ascent arithmetic progression when partially matching.
        # 2. Then weight will be equal to one when matching all params.
        # 3. According to config, it will just match params in [15:]
        
        len_layer = len(gradient_rec)
        final = len_layer - self.broken_tail
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        # flag = False
        if (self.iter+1) % self.peroid_Add10 == 0:
            self.start = self.start-self.add10 if (self.start-self.add10) > self.min_start else self.min_start
        self.iter += 1
        if self.iter >= self.peroid_Add10+1:
            # flag = True
            self.min_weight_scale = 1
            self.max_weight_scale = 1
        for count, (rec, data) in enumerate(zip(gradient_rec, gradient_data)):
            if self.start <= count+1 <= final:
                mask = torch.rand_like(data) > self.dropout
                weight = (count+1-self.start)/(final-self.start)*(self.max_weight_scale-self.min_weight_scale)+self.min_weight_scale
                # if count+1 < 153 and self.iter < 200:
                #     weight = weight*0.1
                scalar_product += (rec * data * mask).sum()*weight
                rec_norm += (rec * mask).pow(2).sum()*weight
                data_norm += (data * mask).pow(2).sum()*weight

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class NonZeroGradCosineSimilarity(GradientLoss, WeightedMixIn):
    """Gradient matching based on cosine similarity of two gradient vectors, but focusing more on Non-zero Gradient"""

    def __init__(self, force_single_steps=False, scale=1.0, task_regularization=0.0, weighted_config: Optional[DictConfig]=None, non_zero_weight=0.3, start_epoch=0, inversed=False, **kwargs):
        super().__init__(force_single_steps=force_single_steps)
        self.scale = scale
        self.task_regularization = task_regularization
        self.non_zero_weight: float = non_zero_weight
        self.start_epoch: int = max(0, int(start_epoch))
        self.epoch = 0
        self.inversed = False # boolean to indicate which is the major part - False(Full), True(Non Zero)
        
        # setup weighted config and specify loss function, to speed up the function `graident_based_loss`.
        self.weighted_config: Optional[Dict] = self.map_weight_config_to_dict(weighted_config)
        self.__gradient_fn = self._weighted_cosine_sim if self.need_weighted else self._cosine_sim
        self._nested_indices: List[List[int]] = []
        self._weights: torch.Tensor = torch.Tensor([])

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None, rec_model: Optional[nn.Module]=None, true_grad: Optional[List[torch.Tensor]]=None):
        super().initialize(loss_fn, cfg_impl, local_hyperparams, rec_model)
        self.epoch = 0
        
        if not self.need_weighted or not rec_model:
            return
        self._nested_indices = self._get_nested_indices_to_calculate(
            rec_model=rec_model,
            indice_style=self.weighted_config["indices"]
        )
        self._weights = self._get_weights_to_calculate(
            rec_model=rec_model,
            nested_indices=self._nested_indices,
            weighted_config=self.weighted_config,
            true_grad=true_grad
        )

    @property
    def need_weighted(self) -> bool:
        return self.weighted_config is not None

    def gradient_based_loss(self, gradient_rec, gradient_data):
        loss = self.__gradient_fn(gradient_rec, gradient_data, self.non_zero_weight, self.start_epoch, self.epoch, self.inversed) * self.scale
        self.epoch = self.epoch + 1
        return loss
    
    def __repr__(self):
        repr_str = f"Not Zero Cosine Similarity with scale={self.scale}, task reg={self.task_regularization}, non_zero_weight={self.non_zero_weight}, and hybrid matching will start at {self.start_epoch} epoch."
        if self.need_weighted:
            repr_str += "\n\t\t" + self.make_weighted_reprs(self.weighted_config) + ","
        return repr_str

    @staticmethod
    @torch.jit.script
    def _cosine_sim(gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], non_zero_weight: float = 0.3, start_epoch: int = 0, epoch: int = 0, inversed: bool = False):
        scalar_product = gradient_rec[0].new_zeros(1,)
        rec_norm = gradient_rec[0].new_zeros(1,)
        data_norm = gradient_rec[0].new_zeros(1,)
        for rec, data in zip(gradient_rec, gradient_data):
            if inversed:
                not_zero_mask = torch.rand_like(data) != 0
                rec = torch.mul(rec, not_zero_mask)
                data = torch.mul(data, not_zero_mask)
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum()
            data_norm += data.pow(2).sum()

        objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
        
        if start_epoch <= epoch:
            scalar_product = gradient_rec[0].new_zeros(1,)
            rec_norm = gradient_rec[0].new_zeros(1,)
            data_norm = gradient_rec[0].new_zeros(1,)
            for rec, data in zip(gradient_rec, gradient_data):
                if not inversed:
                    not_zero_mask = torch.rand_like(data) != 0
                    rec = torch.mul(rec, not_zero_mask)
                    data = torch.mul(data, not_zero_mask)
                
                scalar_product += (rec * data).sum()
                rec_norm += rec.pow(2).sum()
                data_norm += data.pow(2).sum()
                
            non_zero_objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
            objective = objective + torch.mul(non_zero_objective, non_zero_weight)
                
        return objective
    
    def _weighted_cosine_sim(self, gradient_rec: List[torch.Tensor], gradient_data: List[torch.Tensor], non_zero_weight=0.3, start_epoch=0, epoch=0, inversed=False) -> torch.Tensor:
        if not self.need_weighted:
            return self._cosine_sim(gradient_rec, gradient_data)
        
        losses = []
        for indices, weight in zip(self._nested_indices, self._weights):
            fake_grads = [gradient_rec[i] for i in indices]
            true_grads = [gradient_data[i] for i in indices]
            loss = self._cosine_sim(fake_grads, true_grads, non_zero_weight=non_zero_weight, start_epoch=start_epoch, epoch=epoch, inversed=inversed)
            losses.append(torch.mul(loss, weight))
        
        losses = torch.hstack(losses)
        if self.weighted_config["need_average"]:
            return torch.mean(losses)
        else:
            return torch.sum(losses)


objective_lookup = {
    "euclidean": Euclidean,
    "cosine-similarity": CosineSimilarity,
    "masked-cosine-similarity": MaskedCosineSimilarity,
    "fast-cosine-similarity": FastCosineSimilarity,
    "angular": AngularSimilarity,
    "l1": L1Loss,
    "pearlmutter-loss": PearlmutterEuclidean,
    "pearlmutter-cosine": PearlmutterCosine,
    "tag-euclidean": EuclideanTag,
    "wasserstein": Wasserstein,
    "kernelscalingnorm": KernelScalingNorm,
    "dyna-layer-rand-cosine-similarity": DynaLayerRandPickedCosineSimilarity,
    "hybrid-inversion": HybridInversion, 
    "non-zero-cosine-similarity": NonZeroGradCosineSimilarity,
}
