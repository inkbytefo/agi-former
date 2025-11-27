## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from jax import random

from src.models.agiformer import agiformer_init, agiformer_apply
from src.training.loss import morph_loss, PAD_ID

def test_morph_loss_masking_pad():
    key = random.PRNGKey(0)
    params = agiformer_init(d_model=64, n_layers=1, num_heads=4, patch_size=4, window_size=8, thinking_steps=2, key=key)
    B, L, S = 2, 5, 5
    batch = jnp.full((B, L, 1+S), PAD_ID, dtype=jnp.int32)
    batch = batch.at[:, :, 0].set(0)
    outs = agiformer_apply(params, batch, effort=0.1)
    loss = morph_loss(outs, batch, lambda_root=1.0, lambda_suffix=1.0)
    assert jnp.isfinite(loss)

def test_gradients_exist_morph_head():
    key = random.PRNGKey(1)
    params = agiformer_init(d_model=64, n_layers=1, num_heads=4, patch_size=4, window_size=8, thinking_steps=2, key=key)
    B, L, S = 2, 5, params["suffix_slots"]
    batch = jnp.zeros((B, L, 1+S), dtype=jnp.int32)
    def loss_w(w):
        p = {**params, "morph_head": {**params["morph_head"], "W1_root": w}}
        o = agiformer_apply(p, batch, effort=0.6)
        return morph_loss(o, batch, lambda_root=1.0, lambda_suffix=0.5)
    grads_w = jax.grad(loss_w)(params["morph_head"]["W1_root"])
    assert jnp.any(jnp.abs(grads_w) > 0)
