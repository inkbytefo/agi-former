## Developer: inkbytefo
## Modified: 2025-11-27

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from src.training.train_loop import train_epochs
from src.data.morphology import build_vocab
from src.data.morph_loader import iter_morph_batches

def test_train_epochs_with_optax_updates_params():
    texts = ["Bugün geliyorum", "Yarın gidiyorum"]
    root2id, suffix2id = build_vocab(texts, root_limit=50000, suffix_limit=1000)
    batches = list(iter_morph_batches(texts, root2id, suffix2id, suffix_slots=5, analyzer=None, batch_size=2))
    params = train_epochs(batches, root2id, suffix2id, suffix_slots=5, epochs=1, lr=1e-4, seed=0, weight_decay=1e-5, warmup_steps=1, decay_steps=10, clip_norm=0.5)
    assert "morph_head" in params
    w = params["morph_head"]["W1_root"]
    assert jnp.isfinite(jnp.mean(w))

