import torch
import torch.nn as nn
from src.models.memory import HebbianMemory

def test_hebbian_memory():
    print("--- Hebbian Memory Integrity Test ---")
    
    # Konfigürasyon
    B, L, D = 2, 128, 64  # Batch, Length, Dim
    H = 4                 # Heads
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    
    # Modül Başlatma
    try:
        memory = HebbianMemory(d_model=D, num_heads=H, dropout=0.0).to(DEVICE)
        print("✅ Modül başlatıldı.")
    except Exception as e:
        print(f"❌ Modül başlatma hatası: {e}")
        return

    # Forward Pass Testi
    x = torch.randn(B, L, D).to(DEVICE)
    try:
        out = memory(x)
        print(f"✅ Forward pass başarılı. Çıktı boyutu: {out.shape}")
        
        if out.shape != (B, L, D):
            print(f"❌ Boyut hatası! Beklenen: {(B, L, D)}, Alınan: {out.shape}")
            return
            
        if torch.isnan(out).any():
            print("❌ Çıktıda NaN tespit edildi! (Sayısal kararsızlık)")
            return
        else:
            print("✅ Çıktı temiz (NaN yok).")
            
    except Exception as e:
        print(f"❌ Forward pass hatası: {e}")
        return

    # Backward Pass Testi (Gradient Flow)
    try:
        loss = out.sum()
        loss.backward()
        
        # Gradient kontrolü
        grad_norm = memory.qkv.weight.grad.norm().item()
        decay_grad = memory.decay_logits.grad
        
        print(f"✅ Backward pass başarılı.")
        print(f"   - QKV Grad Norm: {grad_norm:.4f}")
        
        if decay_grad is not None:
            print(f"   - Decay Param Grad: {decay_grad.norm().item():.4f} (Lambda öğreniliyor)")
        else:
            print("❌ Decay parametresi gradient almıyor!")
            
    except Exception as e:
        print(f"❌ Backward pass hatası: {e}")

if __name__ == "__main__":
    test_hebbian_memory()
