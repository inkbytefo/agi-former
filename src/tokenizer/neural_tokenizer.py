## Developer: inkbytefo
## Modified: 2025-11-27

import re
from typing import Dict, List, Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import optax

PAD_ID = -1

def build_char_vocab() -> Dict[str, int]:
    letters = list("abcdefghijklmnopqrstuvwxyzçğıöşü")
    digits = list("0123456789")
    syms = ["-", "'", ".", ","]
    chars = ["<PAD>"] + letters + digits + syms
    return {c: i for i, c in enumerate(chars)}

def normalize_word(w: str) -> str:
    return w.lower()

def encode_word_chars(word: str, char2id: Dict[str, int], max_len: int = 32) -> jnp.ndarray:
    w = normalize_word(word)
    arr = jnp.full((max_len,), char2id["<PAD>"], dtype=jnp.int32)
    idxs: List[int] = []
    for ch in w[:max_len]:
        idxs.append(char2id.get(ch, char2id["<PAD>"]))
    if idxs:
        arr = arr.at[:len(idxs)].set(jnp.array(idxs, dtype=jnp.int32))
    return arr

def create_distill_dataset(
    texts: List[str],
    analyzer: Callable[[str], Tuple[str, List[str]]],
    char2id: Dict[str, int],
    suffix_slots: int,
    root2id: Dict[str, int],
    suffix2id: Dict[str, int],
    max_len: int = 32,
):
    xs = []
    root_targets = []
    suffix_targets = []
    for t in texts:
        words = [w for w in re.split(r"\s+", t.strip()) if w]
        for w in words:
            root, sfx = analyzer(w)
            x = encode_word_chars(w, char2id, max_len)
            xs.append(x)
            rid = root2id.get(root, 0)
            root_targets.append(rid)
            sids = []
            for i in range(suffix_slots):
                if i < len(sfx):
                    sids.append(suffix2id.get(sfx[i], 0))
                else:
                    sids.append(PAD_ID)
            suffix_targets.append(jnp.array(sids, dtype=jnp.int32))
    X = jnp.stack(xs, axis=0)
    y_root = jnp.array(root_targets, dtype=jnp.int32)
    y_sfx = jnp.stack(suffix_targets, axis=0)
    return X, y_root, y_sfx

def init_params(Vc: int, d: int, Vr: int, Vs: int, S: int, key: Optional[jax.Array] = None):
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    emb = jax.random.normal(k1, (Vc, d)) * 0.02
    W_root = jax.random.normal(k2, (d, Vr)) * 0.02
    W_sfx = jax.random.normal(k3, (S, d, Vs)) * 0.02
    return {"emb": emb, "W_root": W_root, "W_sfx": W_sfx}

def apply(params, X: jnp.ndarray):
    emb = params["emb"]
    W_root = params["W_root"]
    W_sfx = params["W_sfx"]
    xemb = emb[X]  # [B, L, d]
    h = jnp.mean(xemb, axis=1)  # [B, d]
    root_logits = h @ W_root  # [B, Vr]
    def slot_logits(i):
        return h @ W_sfx[i]
    sfx_logits = jnp.stack([slot_logits(i) for i in range(W_sfx.shape[0])], axis=1)  # [B, S, Vs]
    return {"root": root_logits, "suffix": sfx_logits}

def _masked_ce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    # logits: [B, V], targets: [B]
    B = logits.shape[0]
    onehot = jax.nn.one_hot(jnp.clip(targets, 0), logits.shape[-1])
    logp = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(onehot * logp, axis=-1)
    return jnp.mean(loss)

def _masked_ce_suffix(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    # logits: [B, S, Vs], targets: [B, S]
    B, S, Vs = logits.shape
    loss = 0.0
    count = 0
    for s in range(S):
        t = targets[:, s]
        mask = t != PAD_ID
        if jnp.any(mask):
            ll = _masked_ce(logits[:, s, :][mask], t[mask])
            loss = loss + ll
            count += 1
    return loss / jnp.maximum(1, count)

def loss_fn(params, X, y_root, y_sfx, lambda_root=1.0, lambda_suffix=0.5):
    outs = apply(params, X)
    lr = _masked_ce(outs["root"], y_root)
    ls = _masked_ce_suffix(outs["suffix"], y_sfx)
    return lambda_root * lr + lambda_suffix * ls

def train_distill(
    texts: List[str],
    analyzer: Callable[[str], Tuple[str, List[str]]],
    root2id: Dict[str, int],
    suffix2id: Dict[str, int],
    suffix_slots: int = 5,
    d: int = 64,
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    clip_norm: float = 1.0,
    max_len: int = 32,
    seed: int = 0,
):
    char2id = build_char_vocab()
    Vc = len(char2id)
    Vr = max(root2id.values()) + 1 if root2id else 1
    Vs = max(suffix2id.values()) + 1 if suffix2id else 1
    key = jax.random.PRNGKey(seed)
    params = init_params(Vc, d, Vr, Vs, suffix_slots, key)
    X, y_root, y_sfx = create_distill_dataset(texts, analyzer, char2id, suffix_slots, root2id, suffix2id, max_len)
    warm = 10
    decay_steps = warm + max(epochs * 10, 1)
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, warmup_steps=warm, decay_steps=decay_steps, end_value=0.0)
    tx = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(schedule, weight_decay))
    opt_state = tx.init(params)

    def step(params, opt_state):
        val, grads = jax.value_and_grad(loss_fn)(params, X, y_root, y_sfx)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, val

    for _ in range(epochs):
        params, opt_state, _ = step(params, opt_state)

    id2root = {v: k for k, v in root2id.items()}
    id2suffix = {v: k for k, v in suffix2id.items()}

    def analyze_fn(word: str):
        x = encode_word_chars(word, char2id, max_len)[None, :]
        outs = apply(params, x)
        rid = int(jnp.argmax(outs["root"], axis=-1)[0])
        suffix_ids: List[int] = []
        for s in range(suffix_slots):
            sid = int(jnp.argmax(outs["suffix"][0, s, :]))
            if sid == 0:
                break
            suffix_ids.append(sid)
        root = id2root.get(rid, normalize_word(word))
        suffixes = [id2suffix.get(sid, "") for sid in suffix_ids if sid > 0]
        return root, suffixes

    return analyze_fn
