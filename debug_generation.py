import torch
import torch.nn.functional as F
from src.models.agiformer import AGIFORMER
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "best_model_scaled.pth"

def load_data():
    with open("data/trwiki_clean_train.bin", "rb") as f:
        data = f.read(1024 + 4) # Read a bit more for targets
    return data

def debug_model():
    # Load Model
    print(f"Loading {MODEL_PATH}...")
    model = AGIFORMER(
        d_model=768,
        n_layers=12,
        num_heads=12,
        patch_size=4,
        window_size=256,
        thinking_steps=3
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load Data
    raw_data = load_data()
    # Input: first 1024 bytes
    input_bytes = list(raw_data[:1024])
    # Target: bytes 4..1027
    target_bytes = list(raw_data[4:1028])
    
    x = torch.tensor(input_bytes, dtype=torch.long).unsqueeze(0).to(DEVICE)
    y = torch.tensor(target_bytes, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Forward Pass
    with torch.no_grad():
        logits = model(x) # (1, 256, 4, 256)
        
    print(f"Logits shape: {logits.shape}")
    
    # Calculate Loss manually
    # Flatten
    logits_flat = logits.view(-1, 256)
    target_flat = y.view(-1)
    
    loss = F.cross_entropy(logits_flat, target_flat)
    bpc = loss.item() / 0.693147
    
    print(f"Manual Loss: {loss.item():.4f}")
    print(f"Manual BPC: {bpc:.4f}")
    
    # Check predictions
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1) # (1, 256, 4)
    preds_flat = preds.view(-1).cpu().tolist()
    target_flat = target_flat.cpu().tolist()
    
    correct = sum([1 for i in range(len(preds_flat)) if preds_flat[i] == target_flat[i]])
    accuracy = correct / len(preds_flat)
    
    print(f"Accuracy: {accuracy:.2%}")
    
    # Print first 20 predictions vs targets
    print("\n--- First 20 Predictions ---")
    print(f"{'Target':<10} | {'Pred':<10} | {'Match'}")
    for i in range(20):
        t_char = chr(target_flat[i]) if 32 <= target_flat[i] <= 126 else f"\\x{target_flat[i]:02x}"
        p_char = chr(preds_flat[i]) if 32 <= preds_flat[i] <= 126 else f"\\x{preds_flat[i]:02x}"
        match = "✅" if target_flat[i] == preds_flat[i] else "❌"
        print(f"{t_char:<10} | {p_char:<10} | {match}")

    # Try to generate from this context
    print("\n--- Generating Continuation ---")
    generated = input_bytes[:]
    
    for _ in range(5): # 5 patches
        ctx = generated[-1024:]
        x_gen = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits_gen = model(x_gen)
            
        last_patch = logits_gen[0, -1, :, :]
        next_bytes = torch.argmax(last_patch, dim=-1).tolist()
        generated.extend(next_bytes)
        
        print(f"Generated patch: {next_bytes} -> {[chr(b) if 32<=b<=126 else '?' for b in next_bytes]}")

if __name__ == "__main__":
    debug_model()
