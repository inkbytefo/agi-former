import torch
from src.models.agiformer import AGIFORMER
import os

# --- AYARLAR ---
MODEL_PATH = "best_model_curriculum.pth"
D_MODEL = 512
N_LAYERS = 6
PATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_text(model, prompt, max_len=100, temp=0.6):
    input_bytes = list(prompt.encode('utf-8'))
    pad = (4 - len(input_bytes) % 4) % 4
    input_bytes.extend([32]*pad)
    
    generated = input_bytes[:]
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_len // 4):
            ctx = torch.tensor(generated[-1024:], dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model(ctx, temperature=temp)
            last_patch = logits[0, -1, :].cpu().tolist()
            generated.extend(last_patch)
            
            # Erken durdurma (Eğer EOS veya çok fazla boşluk üretirse)
            if last_patch == [0, 0, 0, 0]: break

    try:
        return bytes(generated).decode('utf-8', errors='replace')
    except:
        return str(generated)

def run_test():
    print(f"Loading {MODEL_PATH} on {DEVICE}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model dosyası bulunamadı.")
        return

    model = AGIFORMER(d_model=D_MODEL, n_layers=N_LAYERS, patch_size=PATCH_SIZE).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    prompts = [
        ("STAGE 1 (Sözlük)", "Elma kelimesinin anlamı: "),
        ("STAGE 2 (Hikaye)", "Küçük kedi bahçede koşarken "),
        ("STAGE 3 (Bilgi)", "Türkiye'nin başkenti Ankara "),
        ("STAGE 3 (Tarih)", "1923 yılında Cumhuriyet ")
    ]
    
    print("\n" + "="*50)
    print("AGIFORMER v7 - CURRICULUM INTELLIGENCE TEST")
    print("="*50)
    
    for stage, p in prompts:
        print(f"\n[{stage}] Prompt: {p}")
        print("-" * 30)
        output = generate_text(model, p, temp=0.5) # Düşük temp = Daha tutarlı
        print(f">> {output}")
        print("-" * 30)

if __name__ == "__main__":
    run_test()
