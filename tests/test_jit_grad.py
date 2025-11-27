## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from jax import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.agiformer import agiformer_init, agiformer_apply


def test_agiformer_jit_and_grad():
    B, L = 2, 16
    D = 32
    patch_size = 4
    params = agiformer_init(d_model=D, n_layers=1, num_heads=4, patch_size=patch_size, window_size=8, thinking_steps=2, key=random.PRNGKey(0))

    key = random.PRNGKey(1)
    x = random.randint(key, (B, L), 0, 256)
    key2 = random.PRNGKey(2)
    x_morph = random.randint(key2, (B, L, 4), 0, 100)

    def loss_fn(p, inp):
        logits = agiformer_apply(p, inp)
        if isinstance(logits, dict):
            loss_root = jnp.mean(logits["root"] ** 2)
            loss_suffix = jnp.mean(logits["suffix"] ** 2)
            return loss_root + loss_suffix
        else:
            return jnp.mean(logits ** 2)

    # JIT compile (byte input)
    loss_jit = jax.jit(lambda inp: loss_fn(params, inp))
    loss_val = loss_jit(x)
    assert jnp.isfinite(loss_val), "Loss should be finite"

    # JIT compile (morphological input)
    loss_jit_m = jax.jit(lambda inp: loss_fn(params, inp))
    loss_val_m = loss_jit_m(x_morph)
    assert jnp.isfinite(loss_val_m), "Morphological loss should be finite"

    # Gradients w.r.t a representative float leaf (byte_head.W1)
    w1 = params["byte_head"]["W1"]
    def loss_w1(w):
        p = {
            **params,
            "byte_head": {**params["byte_head"], "W1": w}
        }
        return loss_fn(p, x)
    grads_w1 = jax.grad(loss_w1)(w1)
    assert jnp.any(jnp.abs(grads_w1) > 0), "Expected non-zero gradients for head.W1"
