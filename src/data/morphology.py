## Developer: inkbytefo
## Modified: 2025-11-27

import re
from collections import Counter
from typing import List, Dict, Tuple, Iterable

PAD_ID = -1
UNK_ROOT_ID = 0
UNK_SUFFIX_ID = 0

def simple_turkish_analyzer(word: str) -> Tuple[str, List[str]]:
    w = word.lower()
    w = re.sub(r"[^a-zçğıöşü]", "", w)
    if not w:
        return "", []
    return w, []

def build_vocab(
    texts: Iterable[str],
    root_limit: int = 50000,
    suffix_limit: int = 1000,
    analyzer = simple_turkish_analyzer,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    if analyzer is None:
        analyzer = simple_turkish_analyzer
    root_counter = Counter()
    suffix_counter = Counter()
    for t in texts:
        words = [w for w in re.split(r"\s+", t.strip()) if w]
        for w in words:
            r, sfx = analyzer(w)
            if r:
                root_counter[r] += 1
            for s in sfx:
                if s:
                    suffix_counter[s] += 1
    # Deterministic ordering: freq desc, then lexicographic
    roots_sorted = sorted(root_counter.items(), key=lambda x: (-x[1], x[0]))[:root_limit-1]
    suffix_sorted = sorted(suffix_counter.items(), key=lambda x: (-x[1], x[0]))[:suffix_limit-1]
    root2id = {r: i+1 for i, (r, _) in enumerate(roots_sorted)}
    suffix2id = {s: i+1 for i, (s, _) in enumerate(suffix_sorted)}
    root2id["<UNK_ROOT>"] = UNK_ROOT_ID
    suffix2id["<UNK_SUFFIX>"] = UNK_SUFFIX_ID
    return root2id, suffix2id

def encode_word(
    word: str,
    root2id: Dict[str, int],
    suffix2id: Dict[str, int],
    suffix_slots: int,
    analyzer = simple_turkish_analyzer,
) -> List[int]:
    root, suffixes = analyzer(word)
    rid = root2id.get(root, UNK_ROOT_ID)
    out = [rid]
    for i in range(suffix_slots):
        if i < len(suffixes):
            sid = suffix2id.get(suffixes[i], UNK_SUFFIX_ID)
            out.append(sid)
        else:
            out.append(PAD_ID)
    return out

def encode_text(
    text: str,
    root2id: Dict[str, int],
    suffix2id: Dict[str, int],
    suffix_slots: int,
    analyzer = simple_turkish_analyzer,
) -> List[List[int]]:
    if analyzer is None:
        analyzer = simple_turkish_analyzer
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return [encode_word(w, root2id, suffix2id, suffix_slots, analyzer) for w in words]

