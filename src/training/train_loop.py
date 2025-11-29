## Developer: inkbytefo
## Modified: 2025-11-28

import jax
import jax.numpy as jnp
from jax import random
import optax
from tqdm import tqdm

from src.models.agiformer import agiformer_init, agiformer_apply
import jax
from .loss import morph_loss, byte_loss, PAD_ID, kolmogorov_complexity_loss

def curriculum_effort(epoch):
    if epoch < 5:
        return 0.1
    elif epoch < 15:
        return 0.6
    else:
        return None

def apply_curriculum_to_batch(batch, epoch):
    if epoch < 5:
        b = batch.at[:, :, 1:].set(PAD_ID)
        return b
    return batch

def total_loss(params, batch, epoch, lambda_root=1.0, lambda_suffix=0.5, effort=1.0, mp_dtype=None, use_remat=False):
    apply_fn = jax.checkpoint(agiformer_apply) if use_remat else agiformer_apply
    outs, ortho_loss = apply_fn(params, batch, effort=effort)
    if isinstance(outs, dict):
        if mp_dtype is not None:
            outs = {
                "root": outs["root"].astype(mp_dtype),
                "suffix": outs["suffix"].astype(mp_dtype),
            }
        main_loss = morph_loss(outs, batch, lambda_root=lambda_root, lambda_suffix=lambda_suffix)
    else:
        targets = jnp.zeros_like(outs, dtype=jnp.int32)
        main_loss = byte_loss(outs, batch)
    
    # Add orthogonality loss if available (from HyperCognitive)
    total = main_loss + 0.1 * ortho_loss  # lambda_ortho = 0.1
    
    # Add Kolmogorov Complexity loss for model simplicity
    kloss = kolmogorov_complexity_loss(params, lambda_k=0.001)
    total = total + kloss
    
    return total
