import torch
from src.models.agiformer import AGIFORMER
import os
import sys

def generate_text(model_path, prompt_text, max_new_tokens=100):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    D_MODEL = 512
    N_LAYERS = 6
    PATCH_SIZE = 4
    
    print(f"Loading {model_path}...")
    model = AGIFORMER(d_model=D_MODEL, n_layers=N_LAYERS, patch_size=PATCH_SIZE).to(DEVICE)
    
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=DEVICE)
        
    model.load_state_dict(state_dict)
    
    # Check for NaN/Inf in weights
    print("Checking model weights for NaNs...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"WARNING: {name} contains NaN or Inf!")
            
    model.eval()
    
    # Prompt Processing
    input_bytes = [ord(c) for c in prompt_text]
    # Pad to Patch Size
    pad_len = (PATCH_SIZE - (len(input_bytes) % PATCH_SIZE)) % PATCH_SIZE
    if pad_len > 0:
        input_bytes.extend([32] * pad_len) # Space padding
        
    print(f"Prompt: '{prompt_text}' (Bytes: {len(input_bytes)})")
    print("-" * 50)
    print(prompt_text, end='', flush=True)
    
    generated = input_bytes[:]
    
    with torch.no_grad():
        for _ in range(max_new_tokens // PATCH_SIZE):
            # Context Window (Keep it reasonable, e.g., 1024)
            context = generated[-1024:]
            curr_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            # Forward (Returns Indices in Inference Mode)
            pred_patches = model(curr_tensor) 
            
            # Get last patch bytes
            last_patch = pred_patches[0, -1, :].cpu().tolist()
            generated.extend(last_patch)
            
            # Decode and Print
            decoded_str = ""
            for b in last_patch:
                if 32 <= b <= 126 or b == 10 or b == 9:
                    decoded_str += chr(b)
                else:
                    # Non-printable check
                    decoded_str += "?"
            
            print(f"{decoded_str} {last_patch}", end=' ', flush=True)
            
    print("\n" + "-" * 50)

if __name__ == "__main__":
    # Test with a simple wikipedia-style prompt
    generate_text("best_model.pth", "The history of ")
