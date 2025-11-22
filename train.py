import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.agiformer import AGIFORMER
from src.data.real_data import get_enwik8_dataloader
import time
import os
import math

def get_lr(step, warmup_steps, d_model):
    # Transformer-style learning rate schedule
    # lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
    # Simplified: Linear Warmup then constant/decay
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0

def train():
    # Hyperparams
    BATCH_SIZE = 4
    SEQ_LEN = 1024
    D_MODEL = 512
    N_LAYERS = 6
    PATCH_SIZE = 4
    
    # Optimization
    BASE_LR = 3e-4 
    WARMUP_STEPS = 100
    STEPS = 5000
    VAL_INTERVAL = 200
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = './data'
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Training on {DEVICE}...")
    
    model = AGIFORMER(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        patch_size=PATCH_SIZE,
        dropout=0.1
    ).to(DEVICE)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    train_loader = get_enwik8_dataloader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, split='train')
    val_loader = get_enwik8_dataloader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, split='val')
    
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    start_time = time.time()
    
    # Training Loop
    step = 0
    train_iter = iter(train_loader)
    
    try:
        while step < STEPS:
            try:
                seq, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                seq, _ = next(train_iter)
                
            seq = seq.to(DEVICE)
            
            # LR Schedule
            lr_mult = get_lr(step, WARMUP_STEPS, D_MODEL)
            for param_group in optimizer.param_groups:
                param_group['lr'] = BASE_LR * lr_mult
                
            # Data Prep
            split_idx = seq.size(1) - PATCH_SIZE
            x = seq[:, :split_idx]
            y = seq[:, PATCH_SIZE:]
            
            # Forward
            optimizer.zero_grad()
            
            logits = model(x, target_bytes=y)
            
            # Loss
            B, L_y = y.shape
            y_reshaped = y.view(B, L_y // PATCH_SIZE, PATCH_SIZE)
            loss = criterion(logits.contiguous().view(-1, 256), y_reshaped.contiguous().view(-1))
            
            # Check NaN
            if torch.isnan(loss):
                print(f"CRITICAL: NaN Loss at step {step}. Skipping batch.")
                step += 1
                continue
                
            # Backward
            loss.backward()
            
            # Aggressive Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            
            if step % 10 == 0:
                bpc = loss.item() / math.log(2)
                print(f"Step {step}: Loss = {loss.item():.4f} | BPC = {bpc:.4f} | LR = {optimizer.param_groups[0]['lr']:.2e}")
                
            # Validation
            if step % VAL_INTERVAL == 0 and step > 0:
                model.eval()
                val_loss = 0
                val_steps = 0
                with torch.no_grad():
                    for v_step, (v_seq, _) in enumerate(val_loader):
                        if v_step >= 20: break
                        v_seq = v_seq.to(DEVICE)
                        v_split = v_seq.size(1) - PATCH_SIZE
                        vx = v_seq[:, :v_split]
                        vy = v_seq[:, PATCH_SIZE:]
                        
                        v_logits = model(vx, target_bytes=vy)
                        
                        B_v, L_vy = vy.shape
                        vy_reshaped = vy.view(B_v, L_vy // PATCH_SIZE, PATCH_SIZE)
                        v_loss = criterion(v_logits.contiguous().view(-1, 256), vy_reshaped.contiguous().view(-1))
                        val_loss += v_loss.item()
                        val_steps += 1
                
                avg_val_loss = val_loss / val_steps
                avg_bpc = avg_val_loss / math.log(2)
                print(f"-- VALIDATION: Loss = {avg_val_loss:.4f} | BPC = {avg_bpc:.4f} --")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), "best_model.pth")
                    print("Saved best model.")
                
                model.train()
                
            step += 1
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Saving last model state...")
        torch.save(model.state_dict(), "last_model.pth")
        print("Saved last_model.pth")

    print(f"Training finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    train()
