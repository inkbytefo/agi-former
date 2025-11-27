## Developer: inkbytefo
## Modified: 2025-11-27

import jax
import jax.numpy as jnp

from src.models.agiformer import agiformer_init
from src.data.morphology import build_vocab
from src.inference import invert_vocab, generate_words

def test_generate_words_runs_and_returns_string():
    texts = ["Bugün geliyorum", "Yarın gidiyorum"]
    root2id, suffix2id = build_vocab(texts, root_limit=50000, suffix_limit=1000)
    id2root = invert_vocab(root2id)
    id2suffix = invert_vocab(suffix2id)
    params = agiformer_init(d_model=64, n_layers=1, num_heads=4, patch_size=4, window_size=8, thinking_steps=2)
    s = generate_words(params, "Merhaba", root2id, suffix2id, id2root, id2suffix, suffix_slots=5, num_words=2, effort=0.6, temperature_root=1.0, temperature_suffix=1.0, top_k_root=10, top_k_suffix=10, seed=0)
    assert isinstance(s, str)
    assert len(s) > 0

