import torch
from src.models.agiformer import AGIFORMER
import os

MODEL_PATH = "best_model_scaled.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_recall_test():
    print(f"--- RECALL TEST (Needle in Haystack) ---")
    
    # Model Yükle
    # Model Yükle
    # SCALED CONFIGURATION (100M Class) matching train_scaled.py
    model = AGIFORMER(
        d_model=768,
        n_layers=12,
        num_heads=12,
        patch_size=4,
        window_size=256,
        thinking_steps=3
    ).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Model bulunamadı.")
        return
    model.eval()

    # Senaryo: Uzun bir gürültü metni içine gizlenmiş bir şifre
    # Context: [Gürültü] + "ŞİFRE: 1453" + [Gürültü] + "ŞİFRE NEDİR? CEVAP:"
    
    needle = "1453"
    haystack_prefix = "Bu bir hafıza testidir. " * 50  # Yaklaşık 1200 byte
    haystack_suffix = " Şimdi testi tamamlayalım. " * 20
    
    prompt_text = f"{haystack_prefix} GİZLİ KOD: {needle}. {haystack_suffix} SORU: GİZLİ KOD NEDİR? CEVAP: "
    
    print(f"Context Uzunluğu: {len(prompt_text)} byte")
    
    # Generate
    input_bytes = list(prompt_text.encode('utf-8'))
    pad = (4 - len(input_bytes) % 4) % 4
    input_bytes.extend([32]*pad)
    
    generated = input_bytes[:]
    
    print("Düşünülüyor...", end="", flush=True)
    
    with torch.no_grad():
        # Sadece 4-8 byte (1-2 patch) üretmemiz yeterli, cevap kısa
        for _ in range(4): 
            ctx = torch.tensor(generated[-1024:], dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model(ctx, temperature=0.1) # Çok düşük temp (Greedy)
            last_patch = logits[0, -1, :].cpu().tolist()
            generated.extend(last_patch)
            print(".", end="", flush=True)
            
    full_text = bytes(generated).decode('utf-8', errors='replace')
    answer = full_text[len(prompt_text):].strip()
    
    print(f"\n\nBeklenen: {needle}")
    print(f"Model Cevabı: {answer}")
    
    if needle in answer:
        print("✅ BAŞARILI: Model gizli kodu hatırladı.")
    else:
        print("❌ BAŞARISIZ: Model bağlamı kaybetti.")

if __name__ == "__main__":
    run_recall_test()
