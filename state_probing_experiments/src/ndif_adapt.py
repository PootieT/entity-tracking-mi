"""
A list of functions for remote NDIF models.
Basically just wrapping up some tracing/generation contexts with perdefined interventions.
NOTE: This Codebase is deprecated because NDIF does not support importing external functions that call remote execution.
"""

import nnsight
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnsight import CONFIG
import os


# def save_activations


def save_activations_ckpt(act, ckpt_dir, batch_idx, verbose=False):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"batch_{batch_idx}.pt")
    torch.save(act, ckpt_path)
    if verbose:
        print(f"Saved activations checkpoint to {ckpt_path}")
    

# def setup_ndif_api():
def setup_ndif_api():
    os.environ['HF_TOKEN'] = ...
    API = ...
    CONFIG.API.APIKEY = API
def save_model_activations(
    model: nnsight.LanguageModel,
    input_ids,
    attention_mask,
):
    num_layers = len(model.model.layers) + 1 # plus input embeddding 
    hidden_dim = model.model.layers[0].mlp.down_proj.out_features # assuming llama3 model
    with model.trace(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        remote=True
    ) as tracer:
        stacked_hs = torch.zeros((num_layers, input_ids.size(0), hidden_dim)).save() 
        stacked_hs[0, :, :] = model.model.layers[0].input[:, -1, :].detach().cpu()  
        for idx, layer in enumerate(model.model.layers):
            stacked_hs[idx + 1, :, :] = layer.output[:, -1, :].detach().cpu()
            
        return stacked_hs
            