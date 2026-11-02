"""
@Encoding:      UTF-8 
@File:          multi_stage_attack.py

@Introduction:  Implementation for multi-stage attack.
@Author:        Kaiming Zhu
@Date:          2024/5/27 19:43
"""

import logging
from typing import Optional, Dict, Callable

import torch
from omegaconf import OmegaConf, open_dict

from .base_attack import _BaseAttacker


log = logging.getLogger(__name__)


class MultiStageAttack(_BaseAttacker):
    """Multi-Stage Attack. Each attack will take the result of previous attack as initialize candidate.
    @Notes:
        We Implement this attacker in form of [Decorator Pattern](https://en.wikipedia.org/wiki/Decorator_pattern).
    """
    def __init__(self, model, loss_fn, attackers: [_BaseAttacker], num_trials=1, setup: Optional[Dict] = None):
        # default value drawback
        if setup is None:
            setup = dict(dtype=torch.float, device=torch.device("cpu"))

        # initializer
        super().__init__(model=model, loss_fn=loss_fn, cfg_attack=attackers[0].cfg, setup=setup)
        self.attackers = attackers
        self.trial_callback_did_invoked: Optional[Callable] = None
        self.trial_did_finish: Optional[Callable] = None
        self.num_trials = num_trials

        # properties to record attack progress
        self._current_trial = None
        self._current_stage = None

        # force all `num_trials` in `self.attackers` to 1
        for attacker in self.attackers:
            cfg = attacker.cfg
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.restarts.num_trials = 1

    @property
    def current_trial(self):
        return self._current_trial

    @property
    def current_stage(self):
        return self._current_stage

    def __repr__(self):
        result = "Multi-Stage Attack with components:\n"
        attacker_reprs = [repr(attacker) for i, attacker in enumerate(self.attackers)]
        for i, attacker_repr in enumerate(attacker_reprs):
            result += f"{i}. {attacker_repr}\n"

        return result

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # 1. setup result and stats to return
        candidate_datas: [torch.Tensor] = []
        candidate_labels: [torch.Tensor] = []
        stats: Dict = {}

        # 2. inject delegates
        datas: Optional[torch.Tensor] = None
        labels: Optional[torch.Tensor] = None

        def candidate_initialize_delegate():
            return datas

        def label_recover_delegate():
            return labels

        for i, attacker in enumerate(self.attackers):
            attacker.trial_callback_did_invoked = self.trial_callback_did_invoked
            if i != 0:
                attacker.candidate_initialize_delegate = candidate_initialize_delegate
                attacker.label_recover_delegate = label_recover_delegate

        # 3. attack for multi-trials
        for trial in range(self.num_trials):
            log.info(f"Multi-Stage: trial {trial} is going to start...")
            self._current_trial = trial

            # reset candidate and labels to None
            datas = None
            labels = None
            
            for i, attacker in enumerate(self.attackers):
                log.info(f"Multi-Stage: trial {trial}, attacker {i} is going to start...")
                self._current_stage = i

                # get result of attack
                current_result, current_stat = attacker.reconstruct(
                    server_payload=server_payload,
                    shared_data=shared_data,
                    server_secrets=server_secrets,
                    initial_data=initial_data if i == 0 else None,
                    dryrun=dryrun
                )

                # Judge that whether it make a better inversion than before.
                # If so, use the best result, or continue for the last one.
                if datas is None and labels is None:
                    datas = current_result["data"]
                    labels = current_result["labels"]
                else:
                    iter: Optional[int] = current_stat["Trial_0_Best_iteration"]
                    if iter is not None:
                        current_minimal_objective_value = current_stat["Trial_0_Val"][iter]
                        previous_obj_value = attacker.score_trial(datas, labels, shared_data).item()
                        if current_minimal_objective_value < previous_obj_value:
                            datas = current_result["data"]
                            labels = current_result["labels"]
                    else:
                        # iter is None only if Rec Loss is Nan, if so, use the best result before.
                        pass

                # merge current_stat to stats
                for key, value in current_stat.items():
                    new_key = key.removeprefix("Trial_0_")
                    new_key = f"Trial_{trial}_Stage_{i}_" + new_key
                    stats[new_key] = value

                # record for result
                if i == len(self.attackers) - 1:
                    candidate_datas.append(datas.detach())
                    candidate_labels.append(labels.detach())
            
            if self.trial_did_finish is not None:
                self.trial_did_finish(self)

        # 4. filter for redundant keys in stats
        redundant_keys = list(filter(lambda x: "opt" in x, stats.keys()))
        for key in redundant_keys:
            stats.pop(key, None)

        # 5. select best data
        scores = torch.Tensor([
            self.attackers[-1].score_trial(datas, labels, shared_data)
            for datas, labels in zip(candidate_datas, candidate_labels)
        ])
        optimal_val, optimal_index = torch.min(scores, dim=0)
        stats["opt_value"] = optimal_val.item()
        stats["opt_index"] = optimal_index.item()
        result = {
            "data": candidate_datas[optimal_index.item()],
            "labels": candidate_labels[optimal_index.item()]
        }

        # 6. remove delegates
        self.trial_did_finish = None
        for i, attacker in enumerate(self.attackers):
            attacker.trial_callback_did_invoked = None
            if i != 0:
                attacker.candidate_initialize_delegate = None
                attacker.label_recover_delegate = None

        # 7. recover counter
        self._current_trial = None
        self._current_stage = None

        # 8. return result
        return result, stats
