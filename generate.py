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
    
    # Encode prompt to UTF-8 bytes
    input_bytes = list(prompt_text.encode('utf-8'))
    
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
            
            # Real-time decoding for display is tricky with multi-byte chars
            # We'll just collect and decode at the end or try best effort
            pass
            
    print("\n" + "-" * 50)
    try:
        full_text = bytes(generated).decode('utf-8', errors='replace')
        # Print only the new part
        print(full_text[len(prompt_text):])
    except:
        print("\n[Decoding Error]")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate text with AGIFORMER')
    parser.add_argument('--prompt', type=str, default="The history of ", help='Text prompt to start generation')
    parser.add_argument('--temp', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--model', type=str, default="best_model.pth", help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Check if user meant to use the Turkish model but it's named differently
    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists("best_model_turkish.pth"):
        print(f"Note: '{model_path}' not found, using 'best_model_turkish.pth' instead.")
        model_path = "best_model_turkish.pth"
        
    generate_text(model_path, args.prompt, temperature=args.temp)
