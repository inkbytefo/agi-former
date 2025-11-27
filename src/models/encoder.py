## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from jax import random

def encoder_init(d_model: int, patch_size: int, key: random.PRNGKey, root_vocab_size: int = 50000, suffix_vocab_size: int = 1000):
    k_byte, k_root, k_suffix, k_z1, k_z2, k_r1, k_r2, k_h1, k_h2, k_pos = random.split(key, 10)
    byte_embedding = random.normal(k_byte, (256, d_model)) * (1.0 / jnp.sqrt(d_model))
    root_embedding = random.normal(k_root, (root_vocab_size, d_model)) * (1.0 / jnp.sqrt(d_model))
    suffix_embedding = random.normal(k_suffix, (suffix_vocab_size, d_model)) * (1.0 / jnp.sqrt(d_model))
    Wz_h = random.normal(k_z1, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    Uz_e = random.normal(k_z2, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    bz = jnp.zeros((d_model,))
    Wr_h = random.normal(k_r1, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    Ur_e = random.normal(k_r2, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    br = jnp.zeros((d_model,))
    Wh_h = random.normal(k_h1, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    Uh_e = random.normal(k_h2, (d_model, d_model)) * (1.0 / jnp.sqrt(d_model))
    bh = jnp.zeros((d_model,))
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, d_model, 2) / d_model))
    gamma = jnp.ones((d_model,))
    beta = jnp.zeros((d_model,))
    return {
        "byte_embedding": byte_embedding,
        "root_embedding": root_embedding,
        "suffix_embedding": suffix_embedding,
        "Wz_h": Wz_h, "Uz_e": Uz_e, "bz": bz,
        "Wr_h": Wr_h, "Ur_e": Ur_e, "br": br,
        "Wh_h": Wh_h, "Uh_e": Uh_e, "bh": bh,
        "inv_freq": inv_freq, "gamma": gamma, "beta": beta,
        "patch_size": patch_size,
        "root_vocab_size": root_vocab_size, "suffix_vocab_size": suffix_vocab_size,
    }

def encoder_apply(params, x):
    inv = params["inv_freq"]
    gamma = params["gamma"]
    beta = params["beta"]
    P = params["patch_size"]

    if x.ndim == 2:
        emb = params["byte_embedding"]
        xb = jnp.take(emb, x.astype(jnp.int32), axis=0)
        B, L, D = xb.shape
        N = L // P
        xb = xb.reshape(B, N, P, D)
        xb = jnp.mean(xb, axis=2)
        t = jnp.arange(N)
        freqs = jnp.einsum('i,j->ij', t, inv)
        embpos = jnp.concatenate([freqs, freqs], axis=-1)
        x1 = xb[..., :D // 2]
        x2 = xb[..., D // 2:]
        rotate = jnp.concatenate([-x2, x1], axis=-1)
        xb = xb * jnp.cos(embpos) + rotate * jnp.sin(embpos)
        mean = jnp.mean(xb, axis=-1, keepdims=True)
        var = jnp.mean((xb - mean) ** 2, axis=-1, keepdims=True)
        xhat = (xb - mean) / jnp.sqrt(var + 1e-5)
        xb = gamma * xhat + beta
        return xb
    else:
        B, L, S = x.shape
        D = gamma.shape[0]
        roots = x[..., 0]
        suffixes = x[..., 1:]
        h0 = jnp.take(params["root_embedding"], roots.astype(jnp.int32), axis=0)
        suff_embs = jnp.take(params["suffix_embedding"], suffixes.astype(jnp.int32), axis=0)
        mask = (suffixes >= 0).astype(jnp.float32)

        Wz_h, Uz_e, bz = params["Wz_h"], params["Uz_e"], params["bz"]
        Wr_h, Ur_e, br = params["Wr_h"], params["Ur_e"], params["br"]
        Wh_h, Uh_e, bh = params["Wh_h"], params["Uh_e"], params["bh"]

        def step(h, inp):
            e, m = inp
            z = jax.nn.sigmoid(jnp.einsum('d,df->f', h, Wz_h) + jnp.einsum('d,df->f', e, Uz_e) + bz)
            r = jax.nn.sigmoid(jnp.einsum('d,df->f', h, Wr_h) + jnp.einsum('d,df->f', e, Ur_e) + br)
            h_tilde = jnp.tanh(jnp.einsum('d,df->f', r * h, Wh_h) + jnp.einsum('d,df->f', e, Uh_e) + bh)
            h_next = h + m * (z * (h_tilde - h))
            return h_next, None

        HL = B * L
        h_init = h0.reshape(HL, D)
        e_seq = suff_embs.reshape(HL, S - 1, D)
        m_seq = mask.reshape(HL, S - 1)
        def compose(h, e, m):
            hN, _ = jax.lax.scan(step, h, (e, m))
            return hN
        h_final = jax.vmap(compose, in_axes=(0, 0, 0))(h_init, e_seq, m_seq)
        x_out = h_final.reshape(B, L, D)

        t = jnp.arange(L)
        freqs = jnp.einsum('i,j->ij', t, inv)
        embpos = jnp.concatenate([freqs, freqs], axis=-1)
        x1 = x_out[..., :D // 2]
        x2 = x_out[..., D // 2:]
        rotate = jnp.concatenate([-x2, x1], axis=-1)
        x_out = x_out * jnp.cos(embpos) + rotate * jnp.sin(embpos)
        mean = jnp.mean(x_out, axis=-1, keepdims=True)
        var = jnp.mean((x_out - mean) ** 2, axis=-1, keepdims=True)
        xhat = (x_out - mean) / jnp.sqrt(var + 1e-5)
        x_out = gamma * xhat + beta
        return x_out
