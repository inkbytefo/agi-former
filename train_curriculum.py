## Developer: inkbytefo
## Modified: 2025-11-23

"""
AGIFORMER Phase 7 - Curriculum Learning & Neuroplasticity
Implements the 3-stage curriculum and dynamic Hebbian decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.agiformer import AGIFORMER
from src.data.curriculum import CurriculumDataLoader
import time
import json
import os

# Configuration
D_MODEL = 512
N_LAYERS = 6
NUM_HEADS = 8
PATCH_SIZE = 4
WINDOW_SIZE = 128
THINKING_STEPS = 3

BATCH_SIZE = 4
SEQ_LEN = 1024
MAX_STEPS = 5000
BASE_LR = 3e-4
WARMUP_STEPS = 100
GRAD_CLIP = 0.5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_curriculum():
    print("=" * 60)
    print("AGIFORMER PHASE 7: Curriculum Learning & Neuroplasticity")
    print("=" * 60)
    
    # Model
    model = AGIFORMER(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        window_size=WINDOW_SIZE,
        thinking_steps=THINKING_STEPS
    ).to(DEVICE)
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {DEVICE}")
    
    # Curriculum Data Loader
    curriculum = CurriculumDataLoader(
        data_dir="./data",
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_steps=MAX_STEPS
    )
    
    # Validation Loader (Constant - typically Wikipedia or a mix)
    # We use the standard clean loader for validation to have a consistent benchmark
    from src.data.clean_turkish_data import get_clean_loader
    val_loader = get_clean_loader(
        data_dir="./data",
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        split="val"
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    step = 0
    best_val_loss = float('inf')
    
    metrics = {"train_bpc": [], "val_bpc": [], "steps": [], "plasticity": []}
    
    start_time = time.time()
    
    print("Starting Curriculum Training...")
    
    while step < MAX_STEPS:
        # Get loader for current stage
        current_loader = curriculum.get_loader(step)
        
        # Iterate through the loader
        # Note: If the loader runs out, the inner loop finishes, and we get it again (new epoch equivalent)
        for batch_idx, (input_seq, target_seq) in enumerate(current_loader):
            if step >= MAX_STEPS:
                break
            
            # Check if we need to switch stage (e.g. if we crossed a threshold mid-epoch)
            if curriculum.check_stage_change(step):
                break # Break inner loop to refresh loader
            
            # Update Plasticity (Neuroplasticity)
            alpha = curriculum.get_plasticity_alpha(step)
            for module in model.modules():
                if hasattr(module, 'set_plasticity'):
                    module.set_plasticity(alpha)
            
            input_seq = input_seq.to(DEVICE)
            target_seq = target_seq.to(DEVICE)
            
            # Learning rate warmup
            if step < WARMUP_STEPS:
                lr = BASE_LR * (step + 1) / WARMUP_STEPS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            optimizer.zero_grad()
            
            # Forward with AMP
            with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
                logits = model(input_seq, target_bytes=target_seq)
                
                B, N, P, V = logits.shape
                loss = criterion(
                    logits.contiguous().view(-1, V),
                    target_seq.contiguous().view(-1)
                )
            
            if torch.isnan(loss):
                print(f"⚠️ NaN detected at step {step}! Skipping batch...")
                continue
            
            bpc = loss.item() / torch.log(torch.tensor(2.0)).item()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            # Log
            current_lr = optimizer.param_groups[0]['lr']
            if step % 10 == 0:
                print(f"Step {step}: Loss={loss.item():.4f} | BPC={bpc:.4f} | Alpha={alpha:.2f} | LR={current_lr:.2e}")
                metrics["train_bpc"].append(bpc)
                metrics["steps"].append(step)
                metrics["plasticity"].append(alpha)
            
            # Validation
            if step % 200 == 0 and step > 0:
                val_loss, val_bpc = validate(model, val_loader, criterion)
                print(f"-- VALIDATION: Loss = {val_loss:.4f} | BPC = {val_bpc:.4f} --")
                
                metrics["val_bpc"].append(val_bpc)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model_curriculum.pth")
                    print("Saved best_model_curriculum.pth")
                
                model.train()
                # Restore plasticity after validation (just in case)
                for module in model.modules():
                    if hasattr(module, 'set_plasticity'):
                        module.set_plasticity(alpha)
            
            step += 1
            
    # Save final
    print("Saving last model state...")
    torch.save(model.state_dict(), "last_model_curriculum.pth")
    
    # Save metrics
    with open("metrics_curriculum.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f}s")

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
            
            if count >= 50:
                break
    
    avg_loss = total_loss / count
    bpc = avg_loss / torch.log(torch.tensor(2.0)).item()
    return avg_loss, bpc

if __name__ == "__main__":
    train_curriculum()
