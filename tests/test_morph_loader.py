## Developer: inkbytefo
## Modified: 2025-11-27

import numpy as np
from src.data.morphology import build_vocab, simple_turkish_analyzer
from src.data.morph_loader import iter_morph_batches

def test_iter_morph_batches_shapes():
    texts = ["Merhaba dünya", "Bugün geliyorum"]
    root2id, suffix2id = build_vocab(texts, root_limit=50000, suffix_limit=1000, analyzer=simple_turkish_analyzer)
    batches = list(iter_morph_batches(texts, root2id, suffix2id, suffix_slots=5, analyzer=simple_turkish_analyzer, batch_size=2))
    assert len(batches) == 1
    arr = batches[0]
    assert arr.dtype == np.int32
    assert arr.shape[0] == 2
    assert arr.shape[2] == 6

