## Developer: inkbytefo
## Modified: 2025-11-27

import os
import json
import argparse
from typing import Dict

from src.data.npy_loader import load_dataset, iter_npy_batches
from src.training.train_loop import train_epochs
import pickle
import csv
import numpy as np

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

    train_data = np.load(os.path.join(args.data_dir, "train.npy"), mmap_mode="r")
    val_data = np.load(os.path.join(args.data_dir, "val.npy"), mmap_mode="r")
    train_batches = list(iter_npy_batches(train_data, args.batch_size))
    val_batches = list(iter_npy_batches(val_data, args.batch_size))

    res = train_epochs(
        train_batches,
        root2id,
        suffix2id,
        suffix_slots,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        clip_norm=args.clip_norm,
        val_iterator=val_batches,
        return_metrics=True,
    )
    params, metrics = res

    log_path = os.path.join(args.data_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()
        for m in metrics:
            w.writerow(m)

    best = None
    best_loss = float("inf")
    for m in metrics:
        if m["val_loss"] is not None and m["val_loss"] < best_loss:
            best_loss = m["val_loss"]
            best = params

    out_path = os.path.join(args.data_dir, "model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Model kaydedildi: {out_path}")
    if best is not None:
        best_path = os.path.join(args.data_dir, "best_model.pkl")
        with open(best_path, "wb") as f:
            pickle.dump(best, f)
        print(f"En iyi model kaydedildi: {best_path} (Val Loss={best_loss})")

if __name__ == "__main__":
    main()
