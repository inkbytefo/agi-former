import torch
from src.models.agiformer import AGIFORMER
import os
import sys

def generate_text(model_path, prompt_text, max_new_tokens=200, temperature=0.8):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Config
    D_MODEL = 512
    N_LAYERS = 6
    PATCH_SIZE = 4
    
    print(f"Loading {model_path} (Temp={temperature})...")
    model = AGIFORMER(d_model=D_MODEL, n_layers=N_LAYERS, patch_size=PATCH_SIZE).to(DEVICE)
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    input_bytes = [ord(c) for c in prompt_text]
    pad_len = (PATCH_SIZE - (len(input_bytes) % PATCH_SIZE)) % PATCH_SIZE
    if pad_len > 0:
        input_bytes.extend([32] * pad_len)
        
    print(f"Prompt: '{prompt_text}'")
    print("-" * 50)
    print(prompt_text, end='', flush=True)
    
    generated = input_bytes[:]
    
    with torch.no_grad():
        for _ in range(max_new_tokens // PATCH_SIZE):
            context = generated[-1024:] # Keep context manageable
            curr_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            # Pass Temperature
            pred_patches = model(curr_tensor, temperature=temperature) 
            
            last_patch = pred_patches[0, -1, :].cpu().tolist()
            generated.extend(last_patch)
            
            decoded_str = ""
            for b in last_patch:
                if 32 <= b <= 126 or b == 10 or b == 9:
                    decoded_str += chr(b)
                else:
                    # Simple representation for non-printables
                    pass 
            
            print(decoded_str, end='', flush=True)
            
    print("\n" + "-" * 50)

if __name__ == "__main__":
    # Test with a generic English prompt to see if it generalizes beyond XML
    generate_text("best_model.pth", "The history of ", temperature=0.7)
