## Developer: inkbytefo
## Modified: 2025-11-28

import os
import argparse
import json
import numpy as np
import jax
import jax.numpy as jnp
import sys

# Add project root to PYTHONPATH when running as a script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.inference import load_model
from src.training.train_loop import total_loss
from src.models.agiformer import agiformer_apply

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    with open(os.path.join(args.data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    suffix_slots = int(vocab["suffix_slots"])

    val = np.load(os.path.join(args.data_dir, "val.npy"), mmap_mode="r")
    if val.shape[0] > 100:
        val = val[:100]

    params = load_model(args.model_path)

    def measure(eff):
        losses = []
        for i in range(0, val.shape[0], args.batch_size):
            batch = jnp.array(val[i:i+args.batch_size])
            v = total_loss(params, batch, epoch=0, lambda_root=1.0, lambda_suffix=0.5, effort=eff)
            losses.append(float(v))
        return float(jnp.mean(jnp.array(losses)))

    low = measure(0.2)
    high = measure(1.0)
    imp = ((low - high) / max(low, 1e-8)) * 100.0
    print(f"Low Effort Loss: {low:.4f}, High Effort Loss: {high:.4f}, Improvement: {imp:.2f}%")

if __name__ == "__main__":
    main()
