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
print("DEBUGGING WITH ACTUAL DATA")
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

# Get loader for step 0 (Stage 1)
loader = curriculum.get_loader(step=0)

print(f"Got loader for stage 1")

# Get first batch
print("\nGetting first batch from loader...")
for batch_idx, (input_seq, target_seq) in enumerate(loader):
    print(f"\n[BATCH {batch_idx}]")
    print(f"  Input shape: {input_seq.shape}")
    print(f"  Target shape: {target_seq.shape}")
    print(f"  Input dtype: {input_seq.dtype}")
    print(f"  Target dtype: {target_seq.dtype}")
    
    # Check for invalid values in input data
    print(f"\n  Input stats:")
    print(f"    min={input_seq.min().item()}, max={input_seq.max().item()}")
    print(f"    Has NaN: {torch.isnan(input_seq.float()).any().item()}")
    print(f"    Has Inf: {torch.isinf(input_seq.float()).any().item()}")
    
    print(f"\n  Target stats:")
    print(f"    min={target_seq.min().item()}, max={target_seq.max().item()}")
    print(f"    Has NaN: {torch.isnan(target_seq.float()).any().item()}")
    print(f"    Has Inf: {torch.isinf(target_seq.float()).any().item()}")
    
    # Check for out-of-range values
    if input_seq.min() < 0 or input_seq.max() > 255:
        print(f"\n  ❌ INPUT OUT OF RANGE [0, 255]!")
    if target_seq.min() < 0 or target_seq.max() > 255:
        print(f"\n  ❌ TARGET OUT OF RANGE [0, 255]!")
    
    # Move to device
    input_seq = input_seq.to(DEVICE)
    target_seq = target_seq.to(DEVICE)
    
    # Try forward pass
    print(f"\n  Running forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            logits = model(input_seq, target_bytes=target_seq)
            
            # Check logits
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
            
            if torch.isnan(logits).any():
                print(f"  ❌ NaN in logits!")
            else:
                print(f"  ✅ Logits are clean!")
                
                # Compute loss
                criterion = nn.CrossEntropyLoss()
                B, N, P, V = logits.shape
                loss = criterion(
                    logits.contiguous().view(-1, V),
                    target_seq.contiguous().view(-1)
                )
                
                print(f"  Loss: {loss.item():.4f}")
                if torch.isnan(loss):
                    print(f"  ❌ NaN in loss!")
                else:
                    print(f"  ✅ SUCCESS!")
    
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    # Only test first batch
    if batch_idx == 0:
        break

print("\n" + "=" * 60)
print("Done!")
