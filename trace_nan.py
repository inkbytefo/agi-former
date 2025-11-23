import torch
import torch.nn as nn
from src.models.agiformer import AGIFORMER
from src.data.curriculum import CurriculumDataLoader

# Configuration
D_MODEL = 512
N_LAYERS = 6
NUM_HEADS = 8
PATCH_SIZE = 4
WINDOW_SIZE = 128
THINKING_STEPS = 3
BATCH_SIZE = 2  # Smaller for debugging
SEQ_LEN = 256    # Smaller for debugging
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("TRACING NaN PROPAGATION WITH AMP")
print("=" * 60)

# Create model
model = AGIFORMER(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    num_heads=NUM_HEADS,
    patch_size=PATCH_SIZE,
    window_size=WINDOW_SIZE,
    thinking_steps=THINKING_STEPS
).to(DEVICE)

# Set plasticity
for module in model.modules():
    if hasattr(module, 'set_plasticity'):
        module.set_plasticity(0.1)

model.train()

# Create simple input
input_seq = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(DEVICE)
target_seq = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(DEVICE)

def check_tensor(t, name):
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    tmin = t.float().min().item() if not has_nan else float('nan')
    tmax = t.float().max().item() if not has_nan else float('nan')
    
    status = "✓" if not (has_nan or has_inf) else "❌"
    print(f"{status} {name:30s} | dtype={str(t.dtype):12s} | min={tmin:10.4f} max={tmax:10.4f} | NaN={has_nan} Inf={has_inf}")
    return has_nan or has_inf

print("\n[TEST 1] WITHOUT AMP")
print("-" * 80)
with torch.no_grad():
    x = model.encoder(input_seq)
    check_tensor(x, "Encoder output")
    
    for i, layer in enumerate(model.layers):
        x_norm = layer.norm1(x)
        check_tensor(x_norm, f"  Layer {i} - norm1")
        
        attn_out = layer.attn(x_norm)
        check_tensor(attn_out, f"  Layer {i} - attn")
        
        mem_out = layer.memory(x_norm)
        check_tensor(mem_out, f"  Layer {i} - memory")
        
        if check_tensor(mem_out, f"  Layer {i} - memory"):
            print(f"\n❌ NaN FOUND IN LAYER {i} MEMORY! Stopping here.")
            break
        
        x = layer(x)
        if check_tensor(x, f"Layer {i} output"):
            break
    else:
        x = model.norm_f(x)
        check_tensor(x, "Final norm")
        x = model.reasoning(x)
        check_tensor(x, "Reasoning")
        logits = model.head(x, target_bytes=target_seq)
        check_tensor(logits, "Logits")

print("\n" + "=" * 80)
print("\n[TEST 2] WITH AMP")
print("-" * 80)

# Reset model (fresh forward pass)
with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=(DEVICE=='cuda')):
        x = model.encoder(input_seq)
        check_tensor(x, "Encoder output")
        
        for i, layer in enumerate(model.layers):
            x_norm = layer.norm1(x)
            check_tensor(x_norm, f"  Layer {i} - norm1")
            
            attn_out = layer.attn(x_norm)
            check_tensor(attn_out, f"  Layer {i} - attn")
            
            mem_out = layer.memory(x_norm)
            check_tensor(mem_out, f"  Layer {i} - memory")
            
            if check_tensor(mem_out, f"  Layer {i} - memory OUTPUT"):
                print(f"\n❌ NaN FOUND IN LAYER {i} MEMORY! Stopping here.")
                # Debug the memory module
                print("\nDebugging memory internals...")
                from src.models.memory import HebbianMemory
                # We need to manually trace through
                break
            
            x = layer(x)
            if check_tensor(x, f"Layer {i} output"):
                break
        else:
            x = model.norm_f(x)
            check_tensor(x, "Final norm")
            x = model.reasoning(x)
            check_tensor(x, "Reasoning")
            logits = model.head(x, target_bytes=target_seq)
            check_tensor(logits, "Logits")

print("\n" + "=" * 80)
print("Done!")
