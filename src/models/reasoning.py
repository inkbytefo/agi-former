## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from jax import random, lax

def reasoning_init(d_model, key: random.PRNGKey):
    k1, k2, k3, k4 = random.split(key, 4)
    W1 = random.normal(k1, (d_model, 4 * d_model)) * (1.0 / jnp.sqrt(d_model))
    b1 = jnp.zeros((4 * d_model,))
    W2 = random.normal(k2, (4 * d_model, d_model)) * (1.0 / jnp.sqrt(4 * d_model))
    b2 = jnp.zeros((d_model,))
    norm_gamma = jnp.ones((d_model,))
    norm_beta = jnp.zeros((d_model,))
    W_gate = random.normal(k3, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    b_gate = jnp.zeros((d_model,))
    W_halt = random.normal(k4, (d_model, 1)) * (1.0 / jnp.sqrt(d_model))
    b_halt = jnp.zeros((1,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "norm_gamma": norm_gamma, "norm_beta": norm_beta, "W_gate": W_gate, "b_gate": b_gate, "W_halt": W_halt, "b_halt": b_halt}

def layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    xhat = (x - mean) / jnp.sqrt(var + 1e-5)
    return gamma * xhat + beta

def reasoning_apply(params, x, thinking_steps: int, effort: float):
    effective_steps = jnp.maximum(1, jnp.asarray(jnp.floor(thinking_steps * effort), dtype=jnp.int32))

    def one_step(state, step_idx):
        normed = layer_norm(state, params["norm_gamma"], params["norm_beta"])
        h = jax.nn.gelu(jnp.einsum('bld,df->blf', normed, params["W1"]) + params["b1"])
        update = jnp.einsum('blf,fd->bld', h, params["W2"]) + params["b2"]
        g = jax.nn.sigmoid(jnp.einsum('bld,df->blf', normed, params["W_gate"]) + params["b_gate"])
        halt_bias = (1.0 - effort) * 5.0
        p_halt = jax.nn.sigmoid(jnp.einsum('bld,df->blf', normed, params["W_halt"]) + params["b_halt"] + halt_bias).squeeze(-1)
        active_scale = (step_idx < effective_steps).astype(jnp.float32)
        update_scale = (1.0 - p_halt)[..., None] * active_scale[..., None]
        state = state + update_scale * g * update
        return state, None
    state, _ = lax.scan(one_step, x, jnp.arange(thinking_steps, dtype=jnp.int32))
    return state
