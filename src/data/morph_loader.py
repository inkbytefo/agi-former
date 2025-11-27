## Developer: inkbytefo
## Modified: 2025-11-27

from typing import List, Dict, Callable, Iterable
import numpy as np

from .morphology import encode_text

def batch_encode_texts(texts: List[str], root2id: Dict[str,int], suffix2id: Dict[str,int], suffix_slots: int, analyzer: Callable) -> List[np.ndarray]:
    arrays = []
    for t in texts:
        seq = encode_text(t, root2id, suffix2id, suffix_slots, analyzer)
        arr = np.array(seq, dtype=np.int32)
        arrays.append(arr)
    return arrays

def iter_morph_batches(texts: Iterable[str], root2id: Dict[str,int], suffix2id: Dict[str,int], suffix_slots: int, analyzer: Callable, batch_size: int):
    buf: List[str] = []
    for t in texts:
        buf.append(t)
        if len(buf) >= batch_size:
            arrays = batch_encode_texts(buf, root2id, suffix2id, suffix_slots, analyzer)
            max_len = max(a.shape[0] for a in arrays)
            B = len(arrays)
            S = 1 + suffix_slots
            out = np.full((B, max_len, S), -1, dtype=np.int32)
            for i, a in enumerate(arrays):
                out[i, :a.shape[0], :] = a
            yield out
            buf = []
    if buf:
        arrays = batch_encode_texts(buf, root2id, suffix2id, suffix_slots, analyzer)
        max_len = max(a.shape[0] for a in arrays)
        B = len(arrays)
        S = 1 + suffix_slots
        out = np.full((B, max_len, S), -1, dtype=np.int32)
        for i, a in enumerate(arrays):
            out[i, :a.shape[0], :] = a
        yield out

