
import jax
import jax.numpy as jnp
from jax import random
import optax
import numpy as np
from tqdm import tqdm
import argparse

from src.models.agiformer import agiformer_init, agiformer_apply

def generate_recall_task(batch_size, seq_len, vocab_size=100, key=None):
    if key is None:
        key = random.PRNGKey(0)
    
    # Task: "A=3 ... A?" -> "3"
    # Tokens: 0..9 digits, 10='=', 11='?', 12='.', 13..vocab_size garbage
    
    k1, k2, k3 = random.split(key, 3)
    
    # Random keys (A, B, C...) mapped to 20..30
    keys = random.randint(k1, (batch_size,), 20, 30)
    # Random values (0..9)
    vals = random.randint(k2, (batch_size,), 0, 10)
    
    inputs = np.full((batch_size, seq_len), 12, dtype=np.int32) # Fill with '.' (12)
    targets = np.full((batch_size, seq_len), -1, dtype=np.int32) # Ignore index
    
    for i in range(batch_size):
        # Set operation at start: "Key = Val"
        inputs[i, 0] = keys[i]
        inputs[i, 1] = 10 # '='
        inputs[i, 2] = vals[i]
        
        # Query at end: "Key ?"
        inputs[i, -2] = keys[i]
        inputs[i, -1] = 11 # '?'
        
        # Target for last token is Val
        targets[i, -1] = vals[i]
        
    return jnp.array(inputs), jnp.array(targets)

def loss_fn(params, inputs, targets):
    print("Inside loss_fn")
    print(f"Inputs shape: {inputs.shape}")
    logits = agiformer_apply(params, inputs)
    print(f"Logits type: {type(logits)}")
    if hasattr(logits, 'shape'):
        print(f"Logits shape: {logits.shape}")
    
    B, N, P, V = logits.shape
    logits = logits.reshape(B, N*P, V)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    mask = (targets != -1)
    loss = loss * mask
    return jnp.sum(loss) / jnp.sum(mask)

def accuracy(params, inputs, targets):
    logits = agiformer_apply(params, inputs)
    B, N, P, V = logits.shape
    logits = logits.reshape(B, N*P, V)
    preds = jnp.argmax(logits, axis=-1)
    
    mask = (targets != -1)
    correct = (preds == targets) & mask
    return jnp.sum(correct) / jnp.sum(mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    
    print("Recall Testi Başlatılıyor...")
    
    key = random.PRNGKey(42)
    
    # Config for small model
    model_config = {
        "d_model": 64,
        "n_layers": 2,
        "num_heads": 2,
        "patch_size": 1, # Important for 1-to-1 token mapping
        "window_size": 32,
        "thinking_steps": 1,
        "root_vocab_size": 256, # Byte mode
        "suffix_vocab_size": 1,
        "suffix_slots": 0,
    }
    
    # Init model
    # We need to pass 'enc' with vocab sizes if we use the updated init?
    # No, we updated agiformer_init to take root_vocab_size etc.
    
    params = agiformer_init(key=key, **model_config)
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    # @jax.jit
    def step(params, opt_state, inputs, targets):
        grads = jax.grad(loss_fn)(params, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss = loss_fn(params, inputs, targets)
        return params, opt_state, loss

    # Train loop
    pbar = tqdm(range(args.steps))
    for i in pbar:
        k_batch = random.fold_in(key, i)
        inputs, targets = generate_recall_task(32, 32, key=k_batch)
        
        params, opt_state, loss = step(params, opt_state, inputs, targets)
        
        if i % 100 == 0:
            acc = accuracy(params, inputs, targets)
            pbar.set_description(f"Loss: {loss:.4f}, Acc: {acc:.4f}")
            
    # Final test
    print("\nFinal Test:")
    k_test = random.fold_in(key, 99999)
    inputs, targets = generate_recall_task(100, 32, key=k_test)
    acc = accuracy(params, inputs, targets)
    print(f"Test Accuracy: {acc:.4f}")
    
    if acc > 0.9:
        print("BAŞARILI: Model hafıza testini geçti!")
    else:
        print("BAŞARISIZ: Model hafıza testini geçemedi.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
