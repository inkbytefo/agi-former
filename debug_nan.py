import torch
import torch.nn as nn
from src.models.agiformer import AGIFORMER

# Configuration matching train_curriculum.py
D_MODEL = 512
N_LAYERS = 6
NUM_HEADS = 8
PATCH_SIZE = 4
WINDOW_SIZE = 128
THINKING_STEPS = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("DEBUGGING NaN ISSUE")
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

print(f"Model created on {DEVICE}")

# Set plasticity to Stage 1 value (the problematic case)
print("\nSetting plasticity to 0.1 (Stage 1)...")
for module in model.modules():
    if hasattr(module, 'set_plasticity'):
        module.set_plasticity(0.1)
        print(f"  Set plasticity on {module.__class__.__name__}")

# Create dummy input
BATCH_SIZE = 2
SEQ_LEN = 1024  # Match training configuration
input_seq = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(DEVICE)
target_seq = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(DEVICE)

print(f"\nInput shape: {input_seq.shape}")
print(f"Target shape: {target_seq.shape}")

# Test forward pass step by step with NaN checking
print("\n" + "=" * 60)
print("TESTING FORWARD PASS WITH NaN DETECTION")
print("=" * 60)

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name}!")
        print(f"   Shape: {tensor.shape}")
        print(f"   Stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
        return True
    elif torch.isinf(tensor).any():
        print(f"⚠️ Inf detected in {name}!")
        print(f"   Shape: {tensor.shape}")
        return True
    else:
        print(f"✓ {name} is clean (min={tensor.min().item():.4f}, max={tensor.max().item():.4f})")
        return False

try:
    # Step 1: Encoder
    print("\n[1] ENCODER")
    x = model.encoder(input_seq)
    if check_nan(x, "Encoder output"):
        print("ERROR: NaN in encoder! Stopping.")
        exit(1)
    
    # Step 2: Hybrid Layers
    for i, layer in enumerate(model.layers):
        print(f"\n[{i+2}] HYBRID LAYER {i}")
        x = layer(x)
        if check_nan(x, f"Layer {i} output"):
            print(f"ERROR: NaN in layer {i}! Stopping.")
            # Let's check the components
            print("\n  Debugging layer components:")
            x_test = model.encoder(input_seq)
            for j in range(i):
                x_test = model.layers[j](x_test)
            
            # Test attention
            print("  Testing attention...")
            x_norm = layer.norm1(x_test)
            attn_out = layer.attn(x_norm)
            check_nan(attn_out, "  Attention output")
            
            # Test memory
            print("  Testing memory...")
            memory_out = layer.memory(x_norm)
            check_nan(memory_out, "  Memory output")
            
            exit(1)
    
    # Step 3: Final Norm
    print("\n[8] FINAL NORM")
    x = model.norm_f(x)
    if check_nan(x, "Final norm output"):
        print("ERROR: NaN in final norm! Stopping.")
        exit(1)
    
    # Step 4: Reasoning
    print("\n[9] REASONING")
    x = model.reasoning(x)
    if check_nan(x, "Reasoning output"):
        print("ERROR: NaN in reasoning! Stopping.")
        exit(1)
    
    # Step 5: Head
    print("\n[10] AUTOREGRESSIVE HEAD")
    logits = model.head(x, target_bytes=target_seq)
    if check_nan(logits, "Logits"):
        print("ERROR: NaN in logits! Stopping.")
        exit(1)
    
    # Step 6: Loss
    print("\n[11] LOSS COMPUTATION")
    criterion = nn.CrossEntropyLoss()
    B, N, P, V = logits.shape
    loss = criterion(
        logits.contiguous().view(-1, V),
        target_seq.contiguous().view(-1)
    )
    
    print(f"\n✅ SUCCESS! Loss = {loss.item():.4f}")
    print(f"   BPC = {loss.item() / torch.log(torch.tensor(2.0)).item():.4f}")
    
except Exception as e:
    print(f"\n❌ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()
