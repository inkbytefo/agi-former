## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from jax import random, lax
from .memory import hebbian_memory_init, hebbian_memory_apply

def swa_init(d_model: int, num_heads: int, key: random.PRNGKey):
    k1, k2 = random.split(key)
    W_qkv = random.normal(k1, (d_model, 3 * d_model)) * (1.0 / jnp.sqrt(d_model))
    b_qkv = jnp.zeros((3 * d_model,))
    W_proj = random.normal(k2, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    b_proj = jnp.zeros((d_model,))
    return {"W_qkv": W_qkv, "b_qkv": b_qkv, "W_proj": W_proj, "b_proj": b_proj, "num_heads": num_heads}

def linear(x, W, b):
    return jnp.einsum('bld,df->blf', x, W) + b

def swa_apply(params, x, window_size: int, num_heads: int = None):
    B, L, D = x.shape
    H = int(params["num_heads"]) if num_heads is None else int(num_heads)
    E = D // H
    qkv = linear(x, params["W_qkv"], params["b_qkv"]).reshape(B, L, 3, H, E)
    q = qkv[:, :, 0]
    k = qkv[:, :, 1]
    v = qkv[:, :, 2]
    scale = 1.0 / jnp.sqrt(E)

    def one_batch(qb, kb, vb):
        ws = int(min(window_size, L))
        Kbuf = jnp.zeros((ws, H, E))
        Vbuf = jnp.zeros((ws, H, E))
        ptr0 = jnp.array(0, dtype=jnp.int32)
        count0 = jnp.array(0, dtype=jnp.int32)
        def step(carry, inp):
            Kbuf, Vbuf, ptr, count = carry
            q_t, k_t, v_t = inp
            Kbuf = lax.dynamic_update_slice(Kbuf, k_t[None, ...], (ptr, 0, 0))
            Vbuf = lax.dynamic_update_slice(Vbuf, v_t[None, ...], (ptr, 0, 0))
            ptr = (ptr + 1) % ws
            count = jnp.minimum(count + 1, ws)
            mask_vec = jnp.arange(ws) < count
            Ks_h = jnp.transpose(Kbuf, (1, 0, 2))
            Vs_h = jnp.transpose(Vbuf, (1, 0, 2))
            scores = jnp.einsum('he,hwe->hw', q_t, Ks_h) * scale
            scores = jnp.where(mask_vec[None, :], scores, -1e4)
            attn = jax.nn.softmax(scores, axis=-1)
            out = jnp.einsum('hw,hwe->he', attn, Vs_h)
            return (Kbuf, Vbuf, ptr, count), out
        (_, _, _, _), outs = lax.scan(step, (Kbuf, Vbuf, ptr0, count0), (qb, kb, vb))
        return outs

    outs = jax.vmap(one_batch)(q, k, v)
    out = outs.reshape(B, L, D)
    out = linear(out, params["W_proj"], params["b_proj"])
    return out

def swiglu_init(d_model, hidden_dim, key: random.PRNGKey):
    k1, k2, k3 = random.split(key, 3)
    W1 = random.normal(k1, (d_model, hidden_dim)) * (1.0 / jnp.sqrt(d_model))
    b1 = jnp.zeros((hidden_dim,))
    W2 = random.normal(k2, (d_model, hidden_dim)) * (1.0 / jnp.sqrt(d_model))
    b2 = jnp.zeros((hidden_dim,))
    W3 = random.normal(k3, (hidden_dim, d_model)) * (1.0 / jnp.sqrt(hidden_dim))
    b3 = jnp.zeros((d_model,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

def swiglu_apply(params, x):
    a = jax.nn.silu(jnp.einsum('bld,df->blf', x, params["W1"]) + params["b1"])
    b = jnp.einsum('bld,df->blf', x, params["W2"]) + params["b2"]
    h = a * b
    return jnp.einsum('blf,fd->bld', h, params["W3"]) + params["b3"]

def hybrid_block_init(d_model, num_heads, window_size, key: random.PRNGKey):
    k_attn, k_mem, k_gate, k_out, k_mlp = random.split(key, 5)
    attn = swa_init(d_model, num_heads, k_attn)
    memory = hebbian_memory_init(d_model, num_heads, k_mem)
    W_gate = random.normal(k_gate, (d_model, 1)) * (1.0 / jnp.sqrt(d_model))
    b_gate = jnp.zeros((1,))
    W_out = random.normal(k_out, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    b_out = jnp.zeros((d_model,))
    norm1_gamma = jnp.ones((d_model,))
    norm1_beta = jnp.zeros((d_model,))
    norm_mlp_gamma = jnp.ones((d_model,))
    norm_mlp_beta = jnp.zeros((d_model,))
    mlp = swiglu_init(d_model, 4 * d_model, k_mlp)
    return {"attn": attn, "memory": memory, "W_gate": W_gate, "b_gate": b_gate, "W_out": W_out, "b_out": b_out, "norm1_gamma": norm1_gamma, "norm1_beta": norm1_beta, "norm_mlp_gamma": norm_mlp_gamma, "norm_mlp_beta": norm_mlp_beta, "window_size": window_size, "mlp": mlp}

def layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    xhat = (x - mean) / jnp.sqrt(var + 1e-5)
    return gamma * xhat + beta

def hybrid_block_apply(params, x, effort: float):
    residual = x
    x_norm = layer_norm(x, params["norm1_gamma"], params["norm1_beta"])
    attn_out = swa_apply(params["attn"], x_norm, int(params["window_size"]), int(params["attn"]["num_heads"])) 
    memory_out = hebbian_memory_apply(params["memory"], x_norm, effort)
    g = jax.nn.sigmoid(jnp.einsum('bld,df->blf', x_norm, params["W_gate"]) + params["b_gate"]) 
    combined = g * attn_out + (1.0 - g) * memory_out
    x = residual + (jnp.einsum('bld,df->blf', combined, params["W_out"]) + params["b_out"]) 
    x = x + swiglu_apply(params["mlp"], layer_norm(x, params["norm_mlp_gamma"], params["norm_mlp_beta"]))
    return x
