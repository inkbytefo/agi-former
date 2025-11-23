import torch
from src.models.agiformer import AGIFORMER
import os

def run_needle_test(model_path, noise_len=1000):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading {model_path} for Recall Test...")
    # Config (Eğitimdeki ile aynı olmalı)
    model = AGIFORMER(d_model=512, n_layers=6, patch_size=4, thinking_steps=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 1. Senaryo Oluşturma
    secret_key = "1453"
    needle = f"Gizli şifre {secret_key}."
    
    # Samanlık (Gürültü) - Wikipedia benzeri rastgele metin
    haystack = " Tarih boyunca birçok medeniyet kurulmuş ve yıkılmıştır. " * (noise_len // 10)
    
    query = " Soru: Gizli şifre nedir? Cevap:"
    
    full_prompt = needle + haystack + query
    
    print(f"\n--- TEST SETUP ---")
    print(f"Context Length: {len(full_prompt)} bytes")
    print(f"Needle: '{secret_key}' at the very beginning.")
    print(f"Query: At the very end.")
    print("-" * 30)
    
    # 2. Generation
    input_bytes = list(full_prompt.encode('utf-8'))
    # Pad
    pad = (4 - len(input_bytes) % 4) % 4
    input_bytes.extend([32]*pad)
    
    generated = input_bytes[:]
    
    print("Generating answer...", end=" ", flush=True)
    
    with torch.no_grad():
        # Sadece cevabı üretmek için 10 byte (2-3 patch) yeterli
        for _ in range(3): 
            # context = generated[-2048:] # ESKİ: Slicing hafızayı siliyordu
            context = generated # YENİ: Tüm geçmişi ver, Hebbian Memory (Linear Attention) halleder.
            curr_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            # Greedy decoding (Temperature 0) - Hafızayı test ediyoruz, yaratıcılığı değil
            pred_patches = model(curr_tensor, temperature=0.0)
            last_patch = pred_patches[0, -1, :].cpu().tolist()
            generated.extend(last_patch)
            
    # 3. Sonuç Analizi
    full_text = bytes(generated).decode('utf-8', errors='replace')
    answer = full_text[len(full_prompt):].strip()
    
    print(f"\n\nModel Output: '{answer}'")
    
    if secret_key in answer:
        print("\n✅ SUCCESS: Memory retained the information!")
    else:
        print("\n❌ FAILURE: Information lost in noise.")

if __name__ == "__main__":
    # Model eğitimi bitince çalıştırılacak
    if os.path.exists("best_model_turkish.pth"):
        run_needle_test("best_model_turkish.pth", noise_len=500)
    else:
        print("Model file not found yet. Wait for training to finish.")
