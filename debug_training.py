import torch
import torch.nn as nn
from src.models.agiformer import AGIFORMER
from src.data.curriculum import CurriculumDataLoader

# Configuration matching train_curriculum.py
D_MODEL = 512
N_LAYERS = 6
NUM_HEADS = 8
PATCH_SIZE = 4
WINDOW_SIZE = 128
THINKING_STEPS = 3
BATCH_SIZE = 4
SEQ_LEN = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("DEBUGGING TRAINING MODE")
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

# Set plasticity to Stage 1 value
print("\nSetting plasticity to 0.1 (Stage 1)...")
for module in model.modules():
    if hasattr(module, 'set_plasticity'):
        module.set_plasticity(0.1)

# Load actual curriculum data
print("\nLoading curriculum data...")
curriculum = CurriculumDataLoader(
    data_dir="./data",
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    max_steps=5000
)

loader = curriculum.get_loader(step=0)
criterion = nn.CrossEntropyLoss()

print(f"Got loader for stage 1")

# TRAINING MODE (not eval)
model.train()
print("\nModel in TRAINING mode")

# Get first batch
print("\nGetting first batch from loader...")
for batch_idx, (input_seq, target_seq) in enumerate(loader):
    print(f"\n[BATCH {batch_idx}]")
    
    input_seq = input_seq.to(DEVICE)
    target_seq = target_seq.to(DEVICE)
    
    # Try forward pass WITH AMP (like in training script)
    print(f"  Running forward pass with AMP...")
    try:
        with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
            logits = model(input_seq, target_bytes=target_seq)
            
            # Check logits
            print(f"  Logits dtype: {logits.dtype}")
            print(f"  Logits stats: min={logits.float().min().item():.4f}, max={logits.float().max().item():.4f}")
            
            if torch.isnan(logits).any():
                print(f"  ❌ NaN in logits after forward!")
            else:
                print(f"  ✅ Logits are clean after forward")
                
                # Compute loss
                B, N, P, V = logits.shape
                loss = criterion(
                    logits.contiguous().view(-1, V),
                    target_seq.contiguous().view(-1)
                )
                
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Loss dtype: {loss.dtype}")
                
                if torch.isnan(loss):
                    print(f"  ❌ NaN in loss!")
                else:
                    print(f"  ✅ Loss is clean!")
                    
                    # Try backward
                    print(f"\n  Running backward pass...")
                    loss.backward()
                    
                    # Check gradients
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"    ❌ NaN gradient in {name}")
                                has_nan_grad = True
                    
                    if not has_nan_grad:
                        print(f"  ✅ All gradients are clean!")
                    
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    # Test multiple batches
    if batch_idx >= 2:
        break

print("\n" + "=" * 60)
print("Done!")
