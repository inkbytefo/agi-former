## Developer: inkbytefo
## Modified: 2025-11-27

import os
import argparse
import jax
import jax.numpy as jnp

from src.data.morphology import build_vocab, encode_text
from src.data.morph_loader import iter_morph_batches
from src.data.stanza_wrapper import StanzaAnalyzer
from src.models.agiformer import agiformer_init
from src.training.train_loop import train_epochs
from src.inference import invert_vocab, generate_words
from src.tokenizer.neural_tokenizer import train_distill

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_stanza", action="store_true")
    parser.add_argument("--use_neural_tokenizer", action="store_true")
    parser.add_argument("--distill_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--suffix_slots", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--decay_steps", type=int, default=1000)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    args = parser.parse_args()

    sample_path = os.path.join("data", "sample.txt")
    if not os.path.exists(sample_path):
        raise FileNotFoundError("data/sample.txt eksik. Lütfen 100 cümlelik bir korpus ekleyin.")
    with open(sample_path, "r", encoding="utf-8", errors="ignore") as fh:
        texts = [line.strip() for line in fh if line.strip()]

    analyzer_fn = None
    if args.use_stanza:
        sa = StanzaAnalyzer()
        if sa.start():
            analyzer_fn = sa.analyze

    root2id, suffix2id = build_vocab(texts, root_limit=50000, suffix_limit=1000, analyzer=analyzer_fn)
    if args.use_neural_tokenizer:
        teacher = analyzer_fn if analyzer_fn is not None else (lambda w: (w.lower(), []))
        analyzer_fn = train_distill(texts, teacher, root2id, suffix2id, suffix_slots=args.suffix_slots, epochs=args.distill_epochs)
    batches = list(iter_morph_batches(texts, root2id, suffix2id, suffix_slots=args.suffix_slots, analyzer=analyzer_fn, batch_size=args.batch_size))

    params = train_epochs(batches, root2id, suffix2id, suffix_slots=args.suffix_slots, epochs=args.epochs, lr=args.lr, warmup_steps=args.warmup_steps, decay_steps=args.decay_steps, clip_norm=args.clip_norm)

    id2root = invert_vocab(root2id)
    id2suffix = invert_vocab(suffix2id)
    out = generate_words(params, texts[0], root2id, suffix2id, id2root, id2suffix, suffix_slots=args.suffix_slots, num_words=10, effort=0.6, temperature_root=0.9, temperature_suffix=1.0, top_k_root=20, top_k_suffix=20)
    print(out)

if __name__ == "__main__":
    main()
