## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from jax import random
import optax

from src.models.agiformer import agiformer_init, agiformer_apply
import jax
from .loss import morph_loss, byte_loss, PAD_ID

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
    outs = apply_fn(params, batch, effort=effort)
    if isinstance(outs, dict):
        if mp_dtype is not None:
            outs = {
                "root": outs["root"].astype(mp_dtype),
                "suffix": outs["suffix"].astype(mp_dtype),
            }
        return morph_loss(outs, batch, lambda_root=lambda_root, lambda_suffix=lambda_suffix)
    else:
        targets = jnp.zeros_like(outs, dtype=jnp.int32)
        return byte_loss(outs, targets)

def _is_trainable_leaf(x):
    return isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.inexact)

def _split_params(params):
    if isinstance(params, dict):
        train = {}
        static = {}
        for k, v in params.items():
            t, s = _split_params(v)
            if t is not None:
                train[k] = t
            static[k] = s
        return train, static
    elif _is_trainable_leaf(params):
        return params, None
    else:
        return None, params

def _merge_params(train, static):
    if isinstance(static, dict):
        out = {}
        for k in static.keys() | (train.keys() if isinstance(train, dict) else set()):
            tv = train.get(k) if isinstance(train, dict) else None
            sv = static.get(k)
            out[k] = _merge_params(tv, sv)
        return out
    else:
        return train if train is not None else static

def train_epochs(data_iterator, root2id, suffix2id, suffix_slots, epochs=3, lr=1e-3, seed=0, key=None, weight_decay=1e-4, warmup_steps=100, decay_steps=1000, clip_norm=1.0, val_iterator=None, return_metrics=False, mp_dtype=None, use_remat=False):
    key = random.PRNGKey(seed) if key is None else key
    params = agiformer_init(d_model=256, n_layers=2, num_heads=4, patch_size=4, window_size=64, thinking_steps=3, key=key)
    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=lr, warmup_steps=warmup_steps, decay_steps=decay_steps, end_value=0.0)
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    train_params, static_params = _split_params(params)
    opt_state = tx.init(train_params)
    metrics = []
    for epoch in range(epochs):
        eff = curriculum_effort(epoch)
        train_losses = []
        for batch in data_iterator:
            batch = jnp.array(batch)
            batch = apply_curriculum_to_batch(batch, epoch)
            if eff is None:
                eff = random.uniform(random.fold_in(key, epoch), minval=0.2, maxval=1.0)
            def loss_fn_train(tp):
                full = _merge_params(tp, static_params)
                return total_loss(full, batch, epoch, lambda_root=1.0, lambda_suffix=0.5, effort=eff, mp_dtype=mp_dtype, use_remat=use_remat)
            val, grads = jax.value_and_grad(loss_fn_train)(train_params)
            train_losses.append(float(val))
            updates, opt_state = tx.update(grads, opt_state, train_params)
            train_params = optax.apply_updates(train_params, updates)
        val_loss = None
        if val_iterator is not None:
            vs = []
            for vbatch in val_iterator:
                vbatch = jnp.array(vbatch)
                fullp = _merge_params(train_params, static_params)
                v = total_loss(fullp, vbatch, epoch, lambda_root=1.0, lambda_suffix=0.5, effort=eff if eff is not None else 0.6, mp_dtype=mp_dtype, use_remat=use_remat)
                vs.append(float(v))
            val_loss = float(jnp.mean(jnp.array(vs))) if vs else None
        metrics.append({"epoch": epoch, "train_loss": float(jnp.mean(jnp.array(train_losses))) if train_losses else None, "val_loss": val_loss})
    params = _merge_params(train_params, static_params)
    return (params, metrics) if return_metrics else params
