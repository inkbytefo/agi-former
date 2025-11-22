## Developer: inkbytefo
## Modified: 2025-11-22

import torch
from src.models.agiformer import AGIFORMER
import os
import sys

# Configuration
MODEL_PATH = "best_model_turkish.pth"
D_MODEL = 512
N_LAYERS = 6
PATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    print(f"Loading {MODEL_PATH} on {DEVICE}...")
    model = AGIFORMER(d_model=D_MODEL, n_layers=N_LAYERS, patch_size=PATCH_SIZE).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found.")
        return None
        
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate(model, prompt_text, max_new_tokens=150, temperature=0.7):
    # Encode prompt to UTF-8 bytes
    input_bytes = list(prompt_text.encode('utf-8'))
    
    pad_len = (PATCH_SIZE - (len(input_bytes) % PATCH_SIZE)) % PATCH_SIZE
    if pad_len > 0:
        input_bytes.extend([32] * pad_len)
    
    generated = input_bytes[:]
    
    with torch.no_grad():
        for _ in range(max_new_tokens // PATCH_SIZE):
            context = generated[-1024:] # Keep context manageable
            curr_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            # Pass Temperature
            pred_patches = model(curr_tensor, temperature=temperature) 
            
            last_patch = pred_patches[0, -1, :].cpu().tolist()
            generated.extend(last_patch)
            
            # Stop if we generate too many null bytes or weird control chars (optional heuristic)
            if all(b == 0 for b in last_patch):
                break
    
    # Decode
    decoded_str = ""
    # Helper to decode bytes safely
    try:
        decoded_str = bytes(generated).decode('utf-8', errors='replace')
    except:
        # Fallback for very broken sequences
        decoded_str = str(generated)
    
    return decoded_str

def run_tests():
    model = load_model()
    if not model: return

    test_cases = [
        {
            "category": "Morfoloji (Çekim Ekleri)",
            "prompt": "Kitaplarımı masanın üzerine ",
            "temp": 0.5,
            "desc": "Beklenti: 'koydum', 'bıraktım' gibi mantıklı fiil çekimleri."
        },
        {
            "category": "Ünlü Uyumu (Kalın - A/I)",
            "prompt": "Kapının kolu kırıl",
            "temp": 0.5,
            "desc": "Beklenti: 'dı', 'mış' (Kalın ünlü uyumu)."
        },
        {
            "category": "Ünlü Uyumu (İnce - E/İ)",
            "prompt": "Pencerenin önünde gel",
            "temp": 0.5,
            "desc": "Beklenti: 'en', 'ecek', 'di' (İnce ünlü uyumu)."
        },
        {
            "category": "Genel Kültür (Tarih)",
            "prompt": "Türkiye Cumhuriyeti 1923 yılında ",
            "temp": 0.6,
            "desc": "Beklenti: 'kuruldu', 'ilan edildi' gibi tarihsel gerçeklik."
        },
        {
            "category": "Bilim/Teknoloji",
            "prompt": "Yapay zeka teknolojileri günümüzde ",
            "temp": 0.7,
            "desc": "Beklenti: Mantıklı bir cümle tamamlama."
        },
        {
            "category": "Uzun Üretim (Hikaye)",
            "prompt": "Bir zamanlar uzak bir ülkede, ",
            "temp": 0.8,
            "desc": "Beklenti: Hikaye örgüsü ve tutarlılık."
        },
        {
            "category": "Zorlayıcı (Agltinasyon)",
            "prompt": "Çekoslovakyalılaştıramadıklarımızdan ",
            "temp": 0.6,
            "desc": "Beklenti: Kelimenin devamını veya cümleyi bozmadan sürdürme."
        }
    ]

    print("\n" + "="*70)
    print("AGIFORMER TÜRKÇE MODEL TEST RAPORU")
    print("="*70 + "\n")

    for i, case in enumerate(test_cases):
        print(f"TEST {i+1}: {case['category']}")
        print(f"Amaç: {case['desc']}")
        print(f"Prompt: '{case['prompt']}' (Temp: {case['temp']})")
        print("-" * 30)
        
        try:
            output = generate(model, case['prompt'], max_new_tokens=100, temperature=case['temp'])
            # Highlight the new text
            prompt_len = len(case['prompt'])
            # Simple heuristic to find where prompt ends in decoded string might be tricky due to encoding
            # So we just print the whole thing
            print(f">> ÇIKTI:\n{output}")
        except Exception as e:
            print(f">> HATA: {e}")
            
        print("\n" + "*"*70 + "\n")

if __name__ == "__main__":
    run_tests()
