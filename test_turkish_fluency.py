import torch
import torch.nn.functional as F
from src.models.agiformer import AGIFORMER
import os

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "best_model_scaled.pth"

def load_model():
    print(f"Loading {MODEL_PATH}...")
    model = AGIFORMER(
        d_model=768,
        n_layers=12,
        num_heads=12,
        patch_size=4,
        window_size=256,
        thinking_steps=3
    ).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"{MODEL_PATH} not found")

def sample(logits, temperature):
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

def generate(model, prompt, max_new_tokens=100, temperature=0.7):
    input_bytes = list(prompt.encode('utf-8'))
    pad = (4 - len(input_bytes) % 4) % 4
    input_bytes.extend([32]*pad)
    
    generated = input_bytes[:]
    
    with torch.no_grad():
        for _ in range(max_new_tokens // 4):
            ctx = generated[-1024:]
            x = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            logits = model(x)
            last_patch_logits = logits[0, -1, :, :]
            
            new_bytes = sample(last_patch_logits, temperature).tolist()
            generated.extend(new_bytes)
            
    return bytes(generated).decode('utf-8', errors='replace')

def run_tests():
    model = load_model()
    
    prompts = [
        "Türkiye Cumhuriyeti ",
        "İstanbul ",
        "Yapay zeka ",
        "Bir varmış bir yokmuş, ",
        "Merhaba, bugün hava "
    ]
    
    temperatures = [0.1, 0.7]
    
    print("\n" + "="*50)
    print("TURKISH FLUENCY TEST REPORT (Step ~8000)")
    print("="*50 + "\n")
    
    for prompt in prompts:
        print(f"PROMPT: '{prompt}'")
        
        # PAD TO 1024 BYTES (Left Padding with Space)
        # This simulates the training condition where the model always sees 1024 bytes.
        input_bytes = list(prompt.encode('utf-8'))
        pad_len = 1024 - len(input_bytes)
        if pad_len > 0:
            padded_prompt = " " * pad_len + prompt
        else:
            padded_prompt = prompt[-1024:]
            
        print(f"(Padded to 1024 bytes with spaces)")
        print("-" * 20)
        
        for temp in temperatures:
            # Generate using the PADDED prompt
            # We only want to see the NEW output
            output = generate(model, padded_prompt, max_new_tokens=150, temperature=temp)
            
            # Extract the new part (after the 1024 bytes)
            generated_text = output[len(padded_prompt):].replace('\n', ' ')
            print(f"Temp {temp}: ...{generated_text}")
        print("\n")

if __name__ == "__main__":
    run_tests()
