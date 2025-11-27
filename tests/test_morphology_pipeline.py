## Developer: inkbytefo
## Modified: 2025-11-27

import jax.numpy as jnp
from src.data.morphology import encode_text, build_vocab, PAD_ID

def test_encode_text_shapes_and_ids():
    text = "Merhaba dünya!"
    suffix_slots = 5
    root2id, suffix2id = build_vocab([text], root_limit=50000, suffix_limit=1000)
    seq = encode_text(text, root2id, suffix2id, suffix_slots)
    assert len(seq) == 2
    assert all(len(w) == 1 + suffix_slots for w in seq)
    arr = jnp.array(seq)
    assert arr.shape == (2, 1 + suffix_slots)
    # Check ID ranges (excluding PAD)
    roots = arr[:, 0]
    suffixes = arr[:, 1:]
    assert jnp.all(roots >= 0)
    valid_suffix = suffixes[suffixes != PAD_ID]
    if valid_suffix.size > 0:
        assert jnp.all(valid_suffix >= 0)
