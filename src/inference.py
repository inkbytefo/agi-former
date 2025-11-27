## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any

PAD_ID = -1
EOS_SUFFIX_ID = 0

def apply_top_k(logits: jnp.ndarray, k: Optional[int]) -> jnp.ndarray:
    if k is None or k <= 0:
        return logits
    k = int(k)
    values = jnp.sort(logits, axis=-1)[..., -k]
    mask = logits < values[..., None]
    return jnp.where(mask, -1e9, logits)

def sample_index(logits: jnp.ndarray, key: Any, temperature: float = 1.0, top_k: Optional[int] = None) -> int:
    logits = apply_top_k(logits, top_k)
    t = jnp.maximum(temperature, 1e-5)
    return int(jax.random.categorical(key, logits / t))

def invert_vocab(v2id: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in v2id.items()}

def detokenize_word(root_id: int, suffix_ids: List[int], id2root: Dict[int, str], id2suffix: Dict[int, str]) -> str:
    r = id2root.get(root_id, f"<ROOT_{root_id}>")
    parts = []
    for sid in suffix_ids:
        if sid == PAD_ID or sid == EOS_SUFFIX_ID:
            break
        parts.append(id2suffix.get(sid, f"<SUF_{sid}>"))
    return r + ("".join(parts) if parts else "")

def sample_next_word(outs: Dict[str, jnp.ndarray], key: Any, suffix_slots: int, temperature_root: float = 1.0, temperature_suffix: float = 1.0, top_k_root: Optional[int] = None, top_k_suffix: Optional[int] = None) -> Tuple[int, List[int]]:
    root_logits = outs["root"][0, -1, :]
    rid = sample_index(root_logits, key, temperature_root, top_k_root)
    sids: List[int] = []
    k = key
    for s in range(suffix_slots):
        k, sk = jax.random.split(k)
        logits_s = outs["suffix"][0, -1, s, :]
        sid = sample_index(logits_s, sk, temperature_suffix, top_k_suffix)
        sids.append(sid)
        if sid == EOS_SUFFIX_ID:
            break
    while len(sids) < suffix_slots:
        sids.append(PAD_ID)
    return rid, sids

def generate_words(params: Dict, prompt_text: str, root2id: Dict[str, int], suffix2id: Dict[str, int], id2root: Dict[int, str], id2suffix: Dict[int, str], suffix_slots: int, num_words: int = 5, effort: float = 0.6, temperature_root: float = 1.0, temperature_suffix: float = 1.0, top_k_root: Optional[int] = None, top_k_suffix: Optional[int] = None, seed: int = 0):
    from src.data.morphology import encode_text
    from src.models.agiformer import agiformer_apply
    seq = encode_text(prompt_text, root2id, suffix2id, suffix_slots)
    ctx = jnp.array([seq], dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    outputs = []
    for _ in range(num_words):
        outs = agiformer_apply(params, ctx, effort=effort)
        key, sk = jax.random.split(key)
        rid, sids = sample_next_word(outs, sk, suffix_slots, temperature_root, temperature_suffix, top_k_root, top_k_suffix)
        word = detokenize_word(rid, sids, id2root, id2suffix)
        outputs.append(word)
        new_word = jnp.array([[rid] + sids], dtype=jnp.int32)
        ctx = jnp.concatenate([ctx, new_word[:, None, :]], axis=1)
    return " ".join(outputs)

def load_model(path: str) -> Dict:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
