import torch
from src.models.agiformer import AGIFORMER
import os

def generate_text(model_path, prompt_text, max_new_tokens=200):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model Config (Train ile aynı olmalı)
    D_MODEL = 512
    N_LAYERS = 6
    PATCH_SIZE = 4
    
    print(f"Loading model from {model_path} on {DEVICE}...")
    
    # Load Model
    model = AGIFORMER(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        patch_size=PATCH_SIZE,
        dropout=0.1 # Dropout doesn't matter for eval but init might expect it
    ).to(DEVICE)
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found.")
        if model_path == "best_model.pth" and os.path.exists("last_model.pth"):
            print("Falling back to 'last_model.pth'...")
            model_path = "last_model.pth"
        else:
            print("Error: No model file found.")
            return

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
    model.eval()
    
    # Prepare Prompt
    # Convert string to bytes list
    input_bytes = [ord(c) for c in prompt_text]
    input_tensor = torch.tensor(input_bytes, dtype=torch.long).unsqueeze(0).to(DEVICE) # (1, L)
    
    print(f"Prompt: {prompt_text}")
    print("-" * 40)
    print(prompt_text, end='', flush=True)
    
    # Generation Loop
    # AGIFORMER currently predicts patches.
    # We need to feed the sequence, get the last patch prediction, append, and repeat.
    # Since our LocalHead is autoregressive, we need to be careful.
    
    with torch.no_grad():
        generated = input_bytes[:]
        
        for _ in range(max_new_tokens // PATCH_SIZE):
            # Prepare current context
            # Ensure length is divisible by PATCH_SIZE for encoder convenience
            # (Encoder handles padding/cutting via logic, but let's keep it simple)
            
            curr_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            # Adjust length to match patch boundaries for input
            L = curr_tensor.size(1)
            pad_len = (PATCH_SIZE - (L % PATCH_SIZE)) % PATCH_SIZE
            if pad_len > 0:
                # Pad with 0 just for encoding alignment if needed, 
                # but our model logic usually truncates. 
                # Let's just rely on what we have.
                pass

            # Forward
            # We don't pass target_bytes, triggering Inference Mode in LocalHead
            # Logits: (1, N_Patches, Patch_Size, 256)
            # Note: In inference mode, LocalHead returns byte INDICES, not Logits!
            # Wait, check agiformer.py logic:
            # If target_bytes is None -> returns bytes directly.
            
            pred_bytes = model(curr_tensor) # (1, N_Patches, Patch_Size)
            
            # Get the LAST patch
            last_patch = pred_bytes[0, -1, :].cpu().tolist()
            
            # Append to generation
            generated.extend(last_patch)
            
            # Print continuously
            text_chunk = "".join([chr(b) if 32 <= b <= 126 else "?" for b in last_patch])
            print(text_chunk, end='', flush=True)
            
    print("\n" + "-" * 40)

if __name__ == "__main__":
    generate_text("best_model.pth", "The history of artificial intelligence ")
