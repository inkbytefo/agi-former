## Developer: inkbytefo
## Modified: 2025-11-28

import os
import sys
import argparse
from typing import Iterable, Optional, List

import numpy as np
import jax
import jax.numpy as jnp

from datasets import load_dataset

from src.data.morphology import build_vocab, encode_text, PAD_ID
from src.training.train_loop import train_epochs


def stream_texts(dataset: str = "wikimedia/wikipedia", config: Optional[str] = "20220301.tr", split: str = "train") -> Iterable[str]:
    try:
        ds = load_dataset(dataset, config, split=split, streaming=True)
        for ex in ds:
            t = ex.get("text") or ex.get("content") or ""
            if t:
                yield t
    except Exception:
        try:
            ds = load_dataset("oscar-corpus/OSCAR-2201", "tr", split=split, streaming=True)
            for ex in ds:
                t = ex.get("text") or ""
                if t:
                    yield t
        except Exception:
            ds = load_dataset("musabg/wikipedia-tr", split=split, streaming=True)
            for ex in ds:
                t = ex.get("text") or ""
                if t:
                    yield t


def build_vocab_from_stream(text_iter: Iterable[str], root_limit: int, suffix_limit: int, sample_limit: int) -> tuple:
    buf = []
    for i, t in enumerate(text_iter):
        buf.append(t)
        if i + 1 >= sample_limit:
            break
    return build_vocab(buf, root_limit=root_limit, suffix_limit=suffix_limit, analyzer=None)


def make_epoch_iterator(text_iter_fn, root2id, suffix2id, suffix_slots: int, batch_size: int, steps_per_epoch: int):
    def _iter():
        steps = 0
        buf: List[str] = []
        for t in text_iter_fn():
            buf.append(t)
            if len(buf) >= batch_size:
                seqs = [encode_text(x, root2id, suffix2id, suffix_slots, analyzer=None) for x in buf]
                max_len = max(len(s) for s in seqs)
                B = len(seqs)
                S = 1 + suffix_slots
                arr = np.full((B, max_len, S), PAD_ID, dtype=np.int32)
                for i, s in enumerate(seqs):
                    for j, tok in enumerate(s):
                        arr[i, j, :] = np.array(tok, dtype=np.int32)
                yield arr
                buf = []
                steps += 1
                if steps_per_epoch is not None and steps >= steps_per_epoch:
                    break
    return _iter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=str, default="agiformer-phase9")
    p.add_argument("--dataset", type=str, default="wikimedia/wikipedia")
    p.add_argument("--config", type=str, default="20220301.tr")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--vocab_samples", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--suffix_slots", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--decay_steps", type=int, default=10000)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--wandb_mode", type=str, default="online")
    args = p.parse_args()

    try:
        import wandb
        wandb.init(project=args.project, mode=args.wandb_mode)
    except Exception:
        wandb = None

    def text_iter_once():
        return stream_texts(args.dataset, args.config, args.split)

    root2id, suffix2id = build_vocab_from_stream(text_iter_once(), root_limit=50000, suffix_limit=1000, sample_limit=args.vocab_samples)

    epoch_iter_factory = make_epoch_iterator(text_iter_once, root2id, suffix2id, args.suffix_slots, args.batch_size, args.steps_per_epoch)

    def log_cb(m):
        if wandb is not None:
            wandb.log(m)

    params, metrics = train_epochs(
        epoch_iter_factory,
        root2id,
        suffix2id,
        args.suffix_slots,
        epochs=args.epochs,
        lr=args.lr,
        seed=0,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        clip_norm=args.clip_norm,
        val_iterator=None,
        return_metrics=True,
        mp_dtype=None,
        use_remat=False,
        log_callback=log_cb,
    )

    if wandb is not None:
        for m in metrics:
            wandb.log({"final_epoch": m["epoch"], "final_train_loss": m["train_loss"], "final_val_loss": m["val_loss"]})

    out_dir = os.path.join("data", "phase9")
    os.makedirs(out_dir, exist_ok=True)
    import pickle
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({"root2id": root2id, "suffix2id": suffix2id, "suffix_slots": args.suffix_slots}, f, ensure_ascii=False)

    print("Eğitim tamamlandı. Model ve vocab data/phase9 dizinine kaydedildi.")


if __name__ == "__main__":
    main()
