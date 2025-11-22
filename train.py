import torch
import torch.nn as nn
import torch.optim as optim
from src.models.agiformer import AGIFORMER
from src.data.real_data import get_enwik8_dataloader
import time
import os
import math

def train():
    # Hyperparams
    BATCH_SIZE = 4 # Reduced for CPU/Memory safety, increase on GPU
    SEQ_LEN = 1024
    D_MODEL = 512
    N_LAYERS = 6
    PATCH_SIZE = 4
    LR = 1e-4 # Lowered from 5e-4 to prevent NaN
    STEPS = 5000 # Run for longer on real data
    VAL_INTERVAL = 100
    SAVE_INTERVAL = 500
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = './data'
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Training on {DEVICE}...")
    
    # Model
    model = AGIFORMER(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        patch_size=PATCH_SIZE,
        dropout=0.1
    ).to(DEVICE)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Data
    train_loader = get_enwik8_dataloader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, split='train')
    val_loader = get_enwik8_dataloader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, split='val')
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Loss: CrossEntropy
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for step, (seq, _) in enumerate(train_loader):
        if step >= STEPS:
            break
            
        seq = seq.to(DEVICE)
        
        # Next-Patch Prediction Logic
        split_idx = seq.size(1) - PATCH_SIZE
        x = seq[:, :split_idx]  # Input
        y = seq[:, PATCH_SIZE:] # Target (shifted by 1 patch)
        
        # Forward
        # Pass 'y' for teacher forcing
        logits = model(x, target_bytes=y)
        
        # Prepare targets
        B, L_y = y.shape
        y_reshaped = y.view(B, L_y // PATCH_SIZE, PATCH_SIZE)
        
        # Flatten
        loss = criterion(logits.contiguous().view(-1, 256), y_reshaped.contiguous().view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        
        if step % 10 == 0:
            bpc = loss.item() / math.log(2)
            print(f"Step {step}: Loss = {loss.item():.4f} | BPC = {bpc:.4f}")
            
        # Validation
        if step % VAL_INTERVAL == 0 and step > 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for v_step, (v_seq, _) in enumerate(val_loader):
                    if v_step >= 20: # Limit val steps for speed
                        break
                    v_seq = v_seq.to(DEVICE)
                    v_split = v_seq.size(1) - PATCH_SIZE
                    vx = v_seq[:, :v_split]
                    vy = v_seq[:, PATCH_SIZE:]
                    
                    # Inference mode (no teacher forcing in forward, but we need it for loss?)
                    # Actually, for validation loss we SHOULD use teacher forcing to measure perplexity.
                    # If we want to measure generation quality, that's different.
                    # Standard perplexity/BPC metric uses teacher forcing.
                    v_logits = model(vx, target_bytes=vy)
                    
                    B_v, L_vy = vy.shape
                    vy_reshaped = vy.view(B_v, L_vy // PATCH_SIZE, PATCH_SIZE)
                    v_loss = criterion(v_logits.contiguous().view(-1, 256), vy_reshaped.contiguous().view(-1))
                    
                    val_loss += v_loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            avg_bpc = avg_val_loss / math.log(2)
            print(f"VALIDATION: Loss = {avg_val_loss:.4f} | BPC = {avg_bpc:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print("Saved best model.")
            
            model.train()

    print(f"Training finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    train()
