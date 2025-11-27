## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from jax import random, lax

def rmsnorm_apply(x, gamma):
    norm = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-8)
    return x * (1.0 / norm) * gamma

def hebbian_memory_init(d_model, num_heads, key):
    k1, k2, k3 = random.split(key, 3)
    W_qkv = random.normal(k1, (d_model, 3 * d_model)) * (1.0 / jnp.sqrt(d_model))
    b_qkv = jnp.zeros((3 * d_model,))
    W_out = random.normal(k2, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    b_out = jnp.zeros((d_model,))
    W_decay = random.normal(k3, (d_model, num_heads)) * 0.02
    b_decay = jnp.ones((num_heads,)) * 4.0
    gamma = jnp.ones((d_model,))
    return {"W_qkv": W_qkv, "b_qkv": b_qkv, "W_out": W_out, "b_out": b_out, "W_decay": W_decay, "b_decay": b_decay, "gamma": gamma, "num_heads": num_heads}

def elu(x):
    return jnp.where(x > 0, x, jnp.exp(x) - 1)

def linear(x, W, b):
    return jnp.einsum('bld,df->blf', x, W) + b

def hebbian_memory_apply(params, x, effort: float):
    B, L, D = x.shape
    H = params["num_heads"]
    E = D // H
    qkv = linear(x, params["W_qkv"], params["b_qkv"])
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(B, L, H, E)
    k = k.reshape(B, L, H, E)
    v = v.reshape(B, L, H, E)
    q = elu(q) + 1.0
    k = elu(k) + 1.0
    q = q / jnp.sqrt(E)
    decay_logits = linear(x, params["W_decay"], params["b_decay"]).reshape(B, L, H, 1)
    base_lambdas = 0.9 + 0.1 * jax.nn.sigmoid(decay_logits)
    effort_mod = 1.0 / (effort + 0.1)
    lambdas = jnp.power(base_lambdas, effort_mod)

    def scan_one_batch(qb, kb, vb, lb):
        S0 = jnp.zeros((H, E, E))
        K0 = jnp.zeros((H, E))
        def step(carry, inp):
            S, Ksum = carry
            q_t, k_t, v_t, l_t = inp
            S = l_t.reshape(H, 1, 1) * S + jnp.einsum('he,hf->hef', k_t, v_t)
            Ksum = l_t.reshape(H, 1) * Ksum + k_t
            num = jnp.einsum('he,hef->hf', q_t, S)
            den = jnp.einsum('he,he->h', q_t, Ksum) + 1e-6
            out = num / den[:, None]
            return (S, Ksum), out
        (_, _), outs = lax.scan(step, (S0, K0), (qb, kb, vb, lb))
        return outs

    outs = jax.vmap(scan_one_batch, in_axes=(0,0,0,0))(q, k, v, lambdas)
    out = outs.reshape(B, L, D)
    out = linear(out, params["W_out"], params["b_out"])
    out = rmsnorm_apply(out, params["gamma"])
    return out
