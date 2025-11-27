## Developer: inkbytefo
## Modified: 2025-11-27

from typing import Iterable, Tuple, Dict, Callable
from .morphology import build_vocab

def iter_texts_from_dir(dir_path: str) -> Iterable[str]:
    import os
    for root, _, files in os.walk(dir_path):
        for f in files:
            p = os.path.join(root, f)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        yield line.strip()
            except Exception:
                continue

def build_vocab_from_dir(dir_path: str, root_limit: int = 50000, suffix_limit: int = 1000, analyzer: Callable = None) -> Tuple[Dict[str,int], Dict[str,int]]:
    texts = iter_texts_from_dir(dir_path)
    return build_vocab(texts, root_limit, suffix_limit, analyzer)

