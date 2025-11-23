## Developer: inkbytefo
## Phase 8: Scaling & Emergence
## Modified: 2025-11-23

"""
AGIFORMER Phase 8 - Scaled Model Training (100M Parameters)
Implements 100M parameter model with:
- 768 dimensional embeddings (50% wider)
- 12 transformer layers (2x deeper)  
- Gradient accumulation for T4 GPU compatibility
- Optimized memory decay (sticky retention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.agiformer import AGIFORMER
from src.data.clean_turkish_data import get_clean_loader
import time
import json
import os

# ============================================================
# SCALED CONFIGURATION (100M Class)
# ============================================================
D_MODEL = 768           # 512 → 768 (+50% width)
N_LAYERS = 12           # 6 → 12 (2x depth for reasoning capacity)
NUM_HEADS = 12          # 8 → 12 (maintain aspect ratio)
PATCH_SIZE = 4          # Keep same (proven stable)
WINDOW_SIZE = 256       # 128 → 256 (wider local context)
THINKING_STEPS = 3      # Keep (System 2 preserved)

# Training Configuration (T4 Optimized)
BATCH_SIZE = 2          # Reduced for VRAM (16GB T4)
ACCUM_STEPS = 4         # Effective batch = 8 (2 * 4)
SEQ_LEN = 1024          # Keep same
MAX_STEPS = 50000       # 20K → 50K (2.5x longer exposure)

# Learning Rate (Lower for larger model)
BASE_LR = 2e-4          # Conservative for stability
WARMUP_STEPS = 500      # Longer warmup for 100M model
GRAD_CLIP = 0.5         # Gradient clipping

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def train_scaled():
    print("=" * 60)
    print("AGIFORMER PHASE 8: Scaling & Emergence (100M)")
    print("=" * 60)
    
    # Create Model
    print(f"\n🚀 Initializing Scaled Model...")
    print(f"   Architecture: {N_LAYERS} layers × {D_MODEL} dim")
    
    model = AGIFORMER(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        window_size=WINDOW_SIZE,
        thinking_steps=THINKING_STEPS
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params/1e6:.1f}M")
    print(f"   Device: {DEVICE}")
    
    if total_params < 80e6:
        print(f"   ⚠️ Warning: Expected ~100M, got {total_params/1e6:.1f}M")
    
    # Data Loaders
    print(f"\n📚 Loading Turkish Wikipedia Data...")
    train_loader = get_clean_loader("./data", BATCH_SIZE, SEQ_LEN, "train")
    val_loader = get_clean_loader("./data", BATCH_SIZE, SEQ_LEN, "val")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    
    # Training State
    model.train()
    step = 0
    best_val_loss = float('inf')
    optimizer.zero_grad()
    
    metrics = {
        "train_bpc": [],
        "val_bpc": [],
        "steps": [],
        "lr": []
    }
    
    start_time = time.time()
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUM_STEPS})")
    print(f"   Max Steps: {MAX_STEPS:,}")
    print(f"   Target Duration: ~5-7 days on T4")
    print(f"\n▶️  Starting Training...\n")
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(100):  # Enough epochs to reach MAX_STEPS
        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            if step >= MAX_STEPS:
                break
            
            input_seq = input_seq.to(DEVICE)
            target_seq = target_seq.to(DEVICE)
            
            # Learning Rate Warmup
            if step < WARMUP_STEPS:
                lr = BASE_LR * (step + 1) / WARMUP_STEPS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Forward Pass (with AMP)
            with torch.amp.autocast('cuda', enabled=(DEVICE=='cuda')):
                logits = model(input_seq, target_bytes=target_seq)
                
                B, N, P, V = logits.shape
                loss = criterion(
                    logits.contiguous().view(-1, V),
                    target_seq.contiguous().view(-1)
                )
                
                # Normalize loss for accumulation
                loss = loss / ACCUM_STEPS
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"⚠️ NaN at step {step}! Skipping...")
                continue
            
            # Backward Pass
            scaler.scale(loss).backward()
            
            # Gradient Accumulation Step
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                # Unscale and clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Metrics
                real_loss = loss.item() * ACCUM_STEPS
                bpc = real_loss / torch.log(torch.tensor(2.0)).item()
                current_lr = optimizer.param_groups[0]['lr']
                
                # Logging
                if step % 10 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                    eta_hours = (MAX_STEPS - step) / (steps_per_sec * 3600) if steps_per_sec > 0 else 0
                    
                    print(f"Step {step:5d}/{MAX_STEPS}: Loss={real_loss:.4f} | BPC={bpc:.4f} | "
                          f"LR={current_lr:.2e} | ETA={eta_hours:.1f}h")
                    
                    metrics["train_bpc"].append(bpc)
                    metrics["steps"].append(step)
                    metrics["lr"].append(current_lr)
                
                # Validation
                if step % 2000 == 0 and step > 0:
                    val_loss, val_bpc = validate(model, val_loader, criterion)
                    print(f"\n-- VALIDATION (Step {step}) --")
                    print(f"   Val Loss: {val_loss:.4f} | Val BPC: {val_bpc:.4f}")
                    
                    metrics["val_bpc"].append(val_bpc)
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), "best_model_scaled.pth")
                        print(f"   ✅ Saved best_model_scaled.pth (BPC: {val_bpc:.4f})")
                    
                    # Save checkpoint
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': real_loss,
                    }, f"checkpoint_step_{step}.pth")
                    print(f"   💾 Saved checkpoint_step_{step}.pth\n")
                    
                    model.train()
                
                step += 1
        
        if step >= MAX_STEPS:
            break
    
    # ============================================================
    # TRAINING COMPLETE
    # ============================================================
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    # Save final model
    torch.save(model.state_dict(), "last_model_scaled.pth")
    print(f"Saved: last_model_scaled.pth")
    
    # Save metrics
    with open("metrics_scaled.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: metrics_scaled.json")
    
    # Training summary
    elapsed = time.time() - start_time
    print(f"\nTraining Duration: {elapsed/3600:.1f} hours")
    print(f"Final BPC: {metrics['train_bpc'][-1]:.4f}")
    print(f"Best Val BPC: {min(metrics['val_bpc']):.4f}")

# ============================================================
# VALIDATION FUNCTION
# ============================================================
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq = input_seq.to(DEVICE)
            target_seq = target_seq.to(DEVICE)
            
            logits = model(input_seq, target_bytes=target_seq)
            
            B, N, P, V = logits.shape
            loss = criterion(
                logits.contiguous().view(-1, V),
                target_seq.contiguous().view(-1)
            )
            
            total_loss += loss.item()
            count += 1
            
            if count >= 50:  # Limit validation batches
                break
    
    avg_loss = total_loss / count
    bpc = avg_loss / torch.log(torch.tensor(2.0)).item()
    return avg_loss, bpc

if __name__ == "__main__":
    train_scaled()
