import torch
import torch.nn as nn
from src.models.agiformer import AGIFORMER
import os

def test_recall():
    print("--- Samanlıkta İğne (Needle in a Haystack) Testi ---")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")
    
    # Model Konfigürasyonu (Eğitimdeki ile aynı olmalı)
    D_MODEL = 512
    N_LAYERS = 6
    NUM_HEADS = 8
    PATCH_SIZE = 4
    WINDOW_SIZE = 128
    THINKING_STEPS = 3
    
    model = AGIFORMER(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        window_size=WINDOW_SIZE,
        thinking_steps=THINKING_STEPS
    ).to(DEVICE)
    
    # Eğer eğitilmiş model varsa yükle, yoksa rastgele ağırlıklarla test et (mekanizma kontrolü)
    model_path = "best_model_turkish.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("⚠️ Uyarı: Eğitilmiş model bulunamadı. Rastgele ağırlıklarla mekanizma testi yapılıyor.")
    
    model.eval()
    
    # Test Verisi Hazırlama
    needle = "Şifre: 1453."
    question = " Soru: Şifre neydi?"
    
    # Samanlık (Gürültü) - Yaklaşık 2000 karakterlik rastgele metin veya tekrar
    haystack = " Bu bir gürültü metnidir. Modelin bunu hatırlaması gerekmez. " * 100 
    
    full_text = needle + haystack + question
    print(f"Metin Uzunluğu: {len(full_text)} karakter")
    
    # Byte çevirimi
    input_bytes = torch.tensor([list(full_text.encode('utf-8'))], dtype=torch.long).to(DEVICE)
    
    # Forward Pass (Generation)
    # Basit bir generation döngüsü
    print("Yanıt üretiliyor...")
    
    generated = []
    with torch.no_grad():
        # Context'i verip devamını ürettireceğiz
        # AGIFORMER forward fonksiyonu logits döndürür.
        # Generation için basit bir döngü kurmamız lazım veya modelin içindeki inference'ı kullanabiliriz.
        # Ancak modelin forward'ı tüm sequence için logits veriyor.
        
        # Son 10 tokenı ürettirelim
        curr_input = input_bytes
        
        for _ in range(10):
            logits = model(curr_input)
            # Son patch'in son token'ı (veya patch bazlı tahmin)
            # Logits shape: (B, N_Patches, Patch_Size, Vocab)
            
            last_patch_logits = logits[:, -1, :, :] # (B, Patch_Size, Vocab)
            probs = torch.softmax(last_patch_logits, dim=-1)
            pred_patch = torch.argmax(probs, dim=-1) # (B, Patch_Size)
            
            # Byte'ları listeye ekle
            pred_bytes = pred_patch[0].cpu().tolist()
            generated.extend(pred_bytes)
            
            # Yeni input oluştur (Autoregressive)
            # Not: Bu çok basit bir append, patch hizalaması gerekebilir ama test için yeterli
            new_input = torch.cat([curr_input, pred_patch], dim=1)
            curr_input = new_input

    # Sonucu decode et
    try:
        decoded_output = bytes(generated).decode('utf-8', errors='ignore')
        print(f"Model Yanıtı: {decoded_output}")
        
        if "1453" in decoded_output:
            print("✅ BAŞARILI: Model şifreyi hatırladı!")
        else:
            print("❌ BAŞARISIZ: Model şifreyi hatırlayamadı.")
            
    except Exception as e:
        print(f"Decode hatası: {e}")

if __name__ == "__main__":
    test_recall()
