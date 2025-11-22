## Developer: inkbytefo
## Modified: 2025-11-22

"""
Kaşgarlı Testi - Turkish Wikipedia Benchmark
Hypothesis: Byte-level models learn agglutinative languages more efficiently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.agiformer import AGIFORMER
from src.data.turkish_wiki import get_turkish_wiki_dataloader
import time
import json
import os

# Configuration (IDENTICAL to English training)
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

def train_turkish():
    """
    Train AGIFORMER on Turkish Wikipedia.
    Logs metrics for comparison with English baseline.
    """
    print("=" * 60)
    print("KAŞGARLI TESTİ - Turkish Wikipedia Benchmark")
    print("=" * 60)
    
    # Model (same architecture)
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
    
    # Data
    train_loader = get_turkish_wiki_dataloader(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        split="train"
    )
    
    val_loader = get_turkish_wiki_dataloader(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        split="val"
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    step = 0
    best_val_loss = float('inf')
    
    # Metrics log
    metrics = {"train_bpc": [], "val_bpc": [], "steps": []}
    
    start_time = time.time()
    
    for epoch in range(100):  # Enough epochs to reach MAX_STEPS
        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            if step >= MAX_STEPS:
                break
            
            input_seq = input_seq.to(DEVICE)
            target_seq = target_seq.to(DEVICE)
            
            # Learning rate warmup
            if step < WARMUP_STEPS:
                lr = BASE_LR * (step + 1) / WARMUP_STEPS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Forward
            logits = model(input_seq, target_bytes=target_seq)
            
            # Loss
            B, N, P, V = logits.shape
            loss = criterion(
                logits.contiguous().view(-1, V),
                target_seq.contiguous().view(-1)
            )
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"⚠️ NaN detected at step {step}! Skipping batch...")
                continue
            
            # BPC (Bits Per Character)
            bpc = loss.item() / torch.log(torch.tensor(2.0)).item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            # Log
            current_lr = optimizer.param_groups[0]['lr']
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f} | BPC = {bpc:.4f} | LR = {current_lr:.2e}")
                metrics["train_bpc"].append(bpc)
                metrics["steps"].append(step)
            
            # Validation
            if step % 200 == 0 and step > 0:
                val_loss, val_bpc = validate(model, val_loader, criterion)
                print(f"-- VALIDATION: Loss = {val_loss:.4f} | BPC = {val_bpc:.4f} --")
                
                metrics["val_bpc"].append(val_bpc)
                
                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best_model_turkish.pth")
                    print("Saved best model (Turkish).")
                
                model.train()
            
            step += 1
        
        if step >= MAX_STEPS:
            break
    
    # Save final
    print("Saving last model state...")
    torch.save(model.state_dict(), "last_model_turkish.pth")
    print("Saved last_model_turkish.pth")
    
    # Save metrics
    with open("metrics_turkish.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f}s")
    print(f"Final validation BPC: {best_val_loss / torch.log(torch.tensor(2.0)).item():.4f}")

def validate(model, val_loader, criterion):
    """Validation loop"""
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
    train_turkish()
