import torch
import torch.nn.functional as F
from src.models.agiformer import AGIFORMER
import os

def sample(logits, temperature):
    """Sample from logits with temperature."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

def test_adaptive_generation():
    # --- CONFIG ---
    MODEL_PATH = "best_model_scaled.pth"
    D_MODEL = 768
    N_LAYERS = 12
    NUM_HEADS = 12
    PATCH_SIZE = 4
    WINDOW_SIZE = 256
    THINKING_STEPS = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading {MODEL_PATH} on {DEVICE}...")
    
    model = AGIFORMER(
        d_model=D_MODEL, 
        n_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        window_size=WINDOW_SIZE,
        thinking_steps=THINKING_STEPS
    ).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print("Model bulunamadı!")
        return

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    prompts = [
        "Türkiye Cumhuriyeti ",
        "Yapay zeka ",
        "İstanbul ",
        "Osmanlı İmparatorluğu "
    ]
    
    print("\n" + "="*60)
    print("ADAPTIVE GENERATION TEST (Minimal Padding)")
    print("="*60)
    
    for prompt in prompts:
        # 1. Encode
        input_bytes = list(prompt.encode('utf-8'))
        
        # 2. Minimal Padding (Only align to patch_size=4)
        pad_needed = (PATCH_SIZE - (len(input_bytes) % PATCH_SIZE)) % PATCH_SIZE
        if pad_needed > 0:
            input_bytes.extend([32] * pad_needed)
            
        print(f"\nPrompt: '{prompt}' ({len(input_bytes)} bytes, {pad_needed} padding)")
        print("-" * 60)
        print(prompt, end="", flush=True)
        
        generated = input_bytes[:]
        
        # 3. Generation Loop
        with torch.no_grad():
            for step in range(50):  # Generate ~200 bytes
                # Use full context (or sliding window if too long)
                ctx = generated[-1024:] if len(generated) > 1024 else generated
                curr_tensor = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(DEVICE)
                
                # v2.0: Returns logits (B, N, 4, 256)
                logits = model(curr_tensor)
                
                # Get last patch logits: (4, 256)
                last_patch_logits = logits[0, -1, :, :]
                
                # Sample with temperature 0.6 (balanced)
                next_bytes = sample(last_patch_logits, temperature=0.6).tolist()
                generated.extend(next_bytes)
                
                # Decode and print incrementally
                try:
                    chunk_bytes = bytes(next_bytes)
                    decoded = chunk_bytes.decode('utf-8', errors='ignore')
                    print(decoded, end="", flush=True)
                except:
                    pass
        
        print("\n")

if __name__ == "__main__":
    test_adaptive_generation()
