"""Load attacker code and instantiate appropriate objects."""
from copy import deepcopy
from typing import Dict, Optional
import logging

import torch

from .optimization_based_attack import OptimizationBasedAttacker
from .multiscale_optimization_attack import MultiScaleOptimizationAttacker
from .optimization_with_label_attack import OptimizationJointAttacker
from .optimization_permutation_attack import OptimizationPermutationAttacker
from .analytic_attack import AnalyticAttacker, ImprintAttacker, DecepticonAttacker, AprilAttacker
from .recursive_attack import RecursiveAttacker
from .multi_stage_attack import MultiStageAttack


log = logging.getLogger(__name__)


def prepare_attack(model, loss, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    if cfg_attack.attack_type == "optimization":
        attacker = OptimizationBasedAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "multiscale":
        attacker = MultiScaleOptimizationAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "analytic":
        attacker = AnalyticAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "april-analytic":
        attacker = AprilAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "imprint-readout":
        attacker = ImprintAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "decepticon-readout":
        attacker = DecepticonAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "recursive":
        attacker = RecursiveAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "joint-optimization":
        attacker = OptimizationJointAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "permutation-optimization":
        attacker = OptimizationPermutationAttacker(model, loss, cfg_attack, setup)
    else:
        raise ValueError(f"Invalid type of attack {cfg_attack.attack_type} given.")

    return attacker


def prepare_multi_stage_attack(model, loss, cfg_attacks: list, num_trials=1, setup: Optional[Dict] = None):
    """function to make MultiStageAttacker"""
    if setup is None:
        setup = dict(dtype=torch.float, device=torch.device("cpu"))

    if len(cfg_attacks) == 1:
        log.warning("prepare_multi_stage_attack: you have just given only one attack config, calling"
                    "`prepare_attack` may make your computation much faster.")

    attackers = [
        prepare_attack(model=model, loss=loss, cfg_attack=cfg, setup=setup)
        for cfg in cfg_attacks
    ]
    return MultiStageAttack(
        model=model,
        loss_fn=loss,
        attackers=attackers,
        num_trials=num_trials,
        setup=setup
    )


__all__ = ["prepare_attack"]
