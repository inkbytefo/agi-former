## Developer: inkbytefo
## Modified: 2025-11-22

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.agiformer import AGIFORMER
import time

def overfit_test():
    # Hyperparams for Overfitting
    BATCH_SIZE = 1
    SEQ_LEN = 256 # Short sequence
    D_MODEL = 256
    N_LAYERS = 2 # Shallow for quick overfitting
    PATCH_SIZE = 4
    LR = 1e-3 # Higher LR for overfitting
    STEPS = 200
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running Overfit Test on {DEVICE}...")
    
    # Model
    model = AGIFORMER(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        patch_size=PATCH_SIZE,
        dropout=0.0 # No dropout for overfitting
    ).to(DEVICE)
    
    # Single Batch Data (Repeated Pattern)
    # Pattern: 0, 1, 2, 3, ...
    pattern = torch.arange(0, 16, dtype=torch.long).repeat(SEQ_LEN // 16 + 1)[:SEQ_LEN]
    x_single = pattern.unsqueeze(0).to(DEVICE) # (1, L)
    
    # Prepare Inputs and Targets
    # Input: [0...L-P]
    # Target: [P...L] (Next Patch Prediction)
    split_idx = SEQ_LEN - PATCH_SIZE
    x_in = x_single[:, :split_idx]
    y_target = x_single[:, PATCH_SIZE:]
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    start_time = time.time()
    
    for step in range(STEPS):
        # Forward
        logits = model(x_in, target_bytes=y_target)
        
        # Reshape targets
        B, L_y = y_target.shape
        y_reshaped = y_target.view(B, L_y // PATCH_SIZE, PATCH_SIZE)
        
        loss = criterion(logits.contiguous().view(-1, 256), y_reshaped.contiguous().view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            
        if loss.item() < 0.1:
            print(f"SUCCESS: Overfit achieved at step {step} with Loss = {loss.item():.4f}")
            break

    print(f"Test finished in {time.time() - start_time:.2f}s")
    print(f"Final Loss: {loss.item():.4f}")

if __name__ == "__main__":
    overfit_test()
