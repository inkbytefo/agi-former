## Developer: inkbytefo
## Modified: 2025-11-27

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import numpy as np

from src.data.stanza_wrapper import StanzaAnalyzer
from src.data.morphology import build_vocab, encode_text, PAD_ID

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--vocab_size", type=int, default=50000)
    p.add_argument("--suffix_vocab_size", type=int, default=1000)
    p.add_argument("--suffix_slots", type=int, default=5)
    p.add_argument("--val_split", type=float, default=0.1)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    texts = []
    with open(args.input_file, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s:
                texts.append(s)

    sa = StanzaAnalyzer()
    analyzer_fn = sa.analyze if sa.start() else None

    root2id, suffix2id = build_vocab(texts, root_limit=args.vocab_size, suffix_limit=args.suffix_vocab_size, analyzer=analyzer_fn)

    sequences = []
    max_len = 0
    for t in tqdm(texts, desc="Analyzing and encoding"):
        seq = encode_text(t, root2id, suffix2id, args.suffix_slots, analyzer=analyzer_fn)
        sequences.append(seq)
        if len(seq) > max_len:
            max_len = len(seq)

    N = len(sequences)
    S = 1 + args.suffix_slots
    data = np.full((N, max_len, S), PAD_ID, dtype=np.int32)
    for i, seq in enumerate(sequences):
        for j, tok in enumerate(seq):
            data[i, j, :] = np.array(tok, dtype=np.int32)

    idx = np.random.permutation(N)
    split = int(N * (1.0 - args.val_split))
    train_idx = idx[:split]
    val_idx = idx[split:]
    np.save(os.path.join(args.output_dir, "train.npy"), data[train_idx])
    np.save(os.path.join(args.output_dir, "val.npy"), data[val_idx])

    vocab = {"root2id": root2id, "suffix2id": suffix2id, "suffix_slots": args.suffix_slots}
    with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
