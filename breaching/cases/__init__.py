"""This submodule controls the different cases of federated learning that could be attacked."""

from .servers import construct_server
from .users import construct_user
from .models import construct_model
from .data import construct_dataloader

__all__ = ["construct_server", "construct_user", "construct_model", "construct_case", "construct_dataloader"]


import torch


def construct_case(cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None, need_all_data=False):
    """This is a helper function that summarizes the startup, but I find the full protocol to often be clearer."""
    model, loss_fn = construct_model(
        cfg_model=cfg_case.model,
        cfg_data=cfg_case.data,
        pretrained=cfg_case.server.pretrained,
        state_dict_path=cfg_case.state_dict_path if "state_dict_path" in cfg_case.keys() else None,
        remote_state_dict_cached_filename=cfg_case.remote_state_dict_cached_filename if "remote_state_dict_cached_filename" in cfg_case.keys() else None,
    )
    
    # Server:
    server = construct_server(model, loss_fn, cfg_case, setup, external_dataloader=external_dataloader)
    model = server.vet_model(model)
    # User:
    user = construct_user(model, loss_fn, cfg_case, setup, need_all_data)
    return user, server, model, loss_fn
