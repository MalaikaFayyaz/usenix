"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

import torch
import time
from typing import Dict, Callable, Optional

from .base_attack import _BaseAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

import logging

log = logging.getLogger(__name__)


class OptimizationBasedAttacker(_BaseAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        if objective_fn is None:
            raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
        else:
            self.objective = objective_fn(**self.cfg.objective)
        self.regularizers = []
        try:
            for key in self.cfg.regularization.keys():
                if self.cfg.regularization[key].scale > 0:
                    self.regularizers += [regularizer_lookup[key](self.setup, **self.cfg.regularization[key])]
        except AttributeError:
            pass  # No regularizers selected.

        try:
            self.augmentations = []
            for key in self.cfg.augmentations.keys():
                self.augmentations += [augmentation_lookup[key](**self.cfg.augmentations[key])]
            self.augmentations = torch.nn.Sequential(*self.augmentations).to(**setup)
        except AttributeError:
            self.augmentations = torch.nn.Sequential()  # No augmentations selected.

    def __repr__(self):
        n = "\n"
        return f"""Attacker (of type {self.__class__.__name__}) with settings:
    Hyperparameter Template: {self.cfg.type}

    Objective: {repr(self.objective)}
    Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
    Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}

    Optimization Setup:
        {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
        """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                candidate_solutions += [
                    self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
                ]
                scores[trial] = self.score_trial(candidate_solutions[trial], labels, shared_data, rec_models)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate.detach().clone()
        stats[f"Trial_{trial}_Best_iteration"] = None
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"], rec_model=rec_model, true_grad=shared_data[0]["gradients"])

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        
        # Gradient Clipping Method introduced by "High-Fidelity Gradient Inversion".
        if 'distance_constraint' in self.cfg.optim.keys():
            self.grad_decay = GradDecay(self.data_shape[1], self.cfg.optim.distance_constraint)
        else:
            self.grad_decay = None
            
        # Code to process reconstruction
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(objective_value.item())
                    else:
                        scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    # Adaption for `High-Fidelity Gradient Inversion`, code in `elif`` is for other methods to introduce
                    if "peroid_Add10" in self.cfg.objective.keys():
                        if objective_value < minimal_value_so_far or (iteration+1) % self.cfg.objective.peroid_Add10 == 0:
                            minimal_value_so_far = objective_value.detach()
                            best_candidate = candidate.detach().clone()
                            stats[f"Trial_{trial}_Best_iteration"] = iteration
                    elif objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()
                        stats[f"Trial_{trial}_Best_iteration"] = iteration


                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp

                    if self.trial_callback_did_invoked is not None:
                        self.trial_callback_did_invoked(
                            self, trial, iteration, candidate.detach().clone(), labels.detach().clone()
                        )

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())
                stats[f"Trial_{trial}_Task_Loss"].append(self.current_task_loss.item())
                stats[f"Trial_{trial}_Objective_Loss"].append(self.current_objective_loss.item())
                for name, reg_loss in self.current_regularizer_loss_by_name.items():
                    stats[f"Trial_{trial}_{name}_Reg_Loss"].append(reg_loss.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass
        
        return best_candidate.detach()

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                total_objective += objective
                total_task_loss += task_loss

            objective_loss = total_objective.clone().detach()
            regularizer_loss_by_name: Dict[str, torch.Tensor] = {}
            for regularizer in self.regularizers:
                current_loss = regularizer(candidate_augmented)
                regularizer_loss_by_name[regularizer.__class__.__name__] = current_loss.clone().detach()
                total_objective += current_loss

            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    elif self.cfg.optim.signed == "hard_quantify":
                        self._optim_signed_quantify(candidate, iteration, mul_scale=self.cfg.optim.mul_scale)
                    else:
                        pass
                if 'distance_constraint' in self.cfg.optim.keys() and iteration <= self.cfg.optim.distance_constraint.stop_iter:
                    self.grad_decay(candidate)

            self.current_objective_loss = objective_loss
            self.current_regularizer_loss_by_name = regularizer_loss_by_name
            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure


    def _optim_signed_quantify(self, candidate, iteration, level=4, mul_scale=2):
        """Specialized Signed SGD used in `High-Fidelity Gradient Inversion`.
        @Reference:
            Z. Ye, W. Luo, Q. Zhou, and Y. Tang, “High-Fidelity Gradient Inversion in Distributed Learning,” in AAAI Conference on Artificial Intelligence, 2024, pp. 19983–19991.
        """
        median = torch.median(torch.abs(candidate.grad))  # 1 for batch 8
        scale = level/2/(median+1e-10)
        alpha = 0.2
        tmp = torch.sign(candidate.grad) * alpha
        
        candidate.grad.mul_(scale).round_().div_(level/2.).clamp_(-1, 1).mul_(1-alpha)
        candidate.grad += tmp
        if iteration <= 100: 
            candidate.grad.mul_(mul_scale)


    def score_trial(self, candidate, labels, shared_data, rec_model=None):
        """Score candidate solutions based on some criterion."""
        if rec_model is None:
            rec_model = self._rec_models

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"], rec_model=rec_model, true_grad=shared_data[0]["gradients"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                score += objective(model, data["gradients"], candidate, labels)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        elif self.cfg.restarts.scoring == "default":
            score = self._score_by_objective_computation(candidate, labels, rec_model, shared_data)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else torch.as_tensor(float("inf"), **self.setup)

    def _score_by_objective_computation(self, candidate, labels, rec_model, shared_data):
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        objective: torch.nn.Module = objective_fn(**self.cfg.objective)

        regularizers: [torch.nn.Module] = []
        try:
            regularizers = [
                regularizer_lookup[key](self.setup, **self.cfg.regularization[key])
                for key in self.cfg.regularization.keys()
                if self.cfg.regularization[key].scale > 0
            ]
        except AttributeError:
            pass

        objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"], rec_model=rec_model, true_grad=shared_data[0]["gradients"])
        for reg in regularizers:
            reg.initialize(rec_model, shared_data, labels)

        score = 0
        for model, data in zip(rec_model, shared_data):
            objective_value, _ = objective(model, data["gradients"], candidate, labels)
            score += objective_value
        for reg in regularizers:
            score += reg(candidate)

        return score.detach()

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        stats["opt_index"] = optimal_index.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution)


class GradDecay:
    def __init__(self, width=224, dc=None):
        """Specialized Gradient Decay Method used in `High-Fidelity Gradient Inversion`.
        @Reference:
            Z. Ye, W. Luo, Q. Zhou, and Y. Tang, “High-Fidelity Gradient Inversion in Distributed Learning,” in AAAI Conference on Artificial Intelligence, 2024, pp. 19983–19991.
        """
        self.width = width
        self.decay_dis_rate = 1 if dc is None else dc.decay_dis_rate
        self.dc = dc
        self._get_decay_index()

    def _get_decay_index(self):
        x = []
        y = []
        half_width = self.width / 2
        square_max_dis = (half_width * self.decay_dis_rate) ** 2
        for i in range(self.width):
            for j in range(self.width):
                square_dis = (i-half_width)**2 + (j-half_width)**2
                if square_dis > square_max_dis:
                    x.append(i)
                    y.append(j)
        self.x = x
        self.y = y

    def __call__(self, candidate):
        candidate.grad[:, :, self.x, self.y] *= self.dc.decay_rate