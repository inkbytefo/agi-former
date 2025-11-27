## Developer: inkbytefo
## Modified: 2025-11-27

import os
import json
import argparse
from typing import Dict

from src.data.npy_loader import load_dataset, iter_npy_batches
from src.training.train_loop import train_epochs
import pickle

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--decay_steps", type=int, default=1000)
    p.add_argument("--clip_norm", type=float, default=1.0)
    args = p.parse_args()

    with open(os.path.join(args.data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    root2id: Dict[str, int] = {k: int(v) for k, v in vocab["root2id"].items()}
    suffix2id: Dict[str, int] = {k: int(v) for k, v in vocab["suffix2id"].items()}
    suffix_slots: int = int(vocab["suffix_slots"])

    data = load_dataset(args.data_dir)
    batches = list(iter_npy_batches(data, args.batch_size))

    params = train_epochs(
        batches,
        root2id,
        suffix2id,
        suffix_slots,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        clip_norm=args.clip_norm,
    )

    out_path = os.path.join(args.data_dir, "model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Model kaydedildi: {out_path}")

if __name__ == "__main__":
    main()
