# 📑 MİMARİ ÖNERİ: AGIFORMER Faz 7 - "Curriculum Learning & Neuroplasticity"

**Tarih:** 23 Kasım 2025
**Konu:** İnsan Benzeri Öğrenme Sürecinin (Pedagojik Eğitim) Mimariye Entegrasyonu
**Hedef:** Wernicke Afazisi (Anlamsız akıcılık) sorununu çözmek ve semantik tutarlılığı artırmak.

---

## 1. Yönetici Özeti (Executive Summary)
Mevcut AGIFORMER mimarisi (Byte-Level + Hebbian Memory), Türkçenin morfolojik yapısını (mekanik zeka) çözmüştür. Ancak model, doğrudan karmaşık veriyle (Wikipedia) eğitildiği için kelime anlamlarını (semantik zeka) oturtmakta zorlanmaktadır.

Bu öneri, eğitimi **3 Aşamalı Müfredat (Curriculum)** sistemine dönüştürmeyi ve modelin hafıza plastisitesini (değişebilirliğini) eğitim süresince dinamik olarak yönetmeyi hedefler.

---

## 2. Veri Mimarisi: Aşamalı Müfredat (Curriculum Data Pipeline)

Modelin eğitim verisi, rastgele bir akış yerine, basitten karmaşığa doğru giden bir sıralamaya tabi tutulacaktır.

### Yeni Modül: `src/data/curriculum.py`

Bu modül, eğitim adımına (`global_step`) göre veri kaynağını dinamik olarak değiştiren bir `CurriculumDataLoader` sınıfı içerecektir.

*   **Aşama 1: Lexical Grounding (Sözlük Aşaması)**
    *   **Kaynak:** TDK Sözlük Tanımları, Wiktionary (Tr).
    *   **İçerik:** `Kelime: Tanım.` formatında basit yapıtaşları.
    *   **Amaç:** Byte kombinasyonlarının (kelimelerin) atomik anlamlarını sabitlemek.
    *   **Süre:** İlk %10 - %15 adım.

*   **Aşama 2: Syntactic Scaffolding (Sentaks İskelesi)**
    *   **Kaynak:** Çocuk Hikayeleri, Basit Haber Metinleri.
    *   **İçerik:** Düşük entropili, Özne-Nesne-Yüklem kurallarına sıkı sıkıya uyan kısa cümleler.
    *   **Amaç:** Gramer kurallarını ve basit mantık ilişkilerini oturtmak.
    *   **Süre:** %15 - %40 adım.

*   **Aşama 3: Semantic Expansion (Ansiklopedik Genişleme)**
    *   **Kaynak:** Wikipedia (Temizlenmiş), Bilimsel Makaleler.
    *   **İçerik:** Yüksek entropili, karmaşık ve uzun metinler.
    *   **Amaç:** Dünya bilgisini ve soyut kavramları öğrenmek.
    *   **Süre:** %40 - %100 adım.

---

## 3. Model Mimarisi: Nöroplastisite (Dynamic Hebbian Decay)

İnsan beynindeki **"Çocukken hızlı öğrenme/unutma, yetişkinken seçici öğrenme/hatırlama"** mekanizmasını simüle etmek için `HebbianMemory` modülü güncellenmelidir.

### Güncellenecek Modül: `src/models/memory.py`

Mevcut `HebbianMemory` sınıfına bir `plasticity_schedule` eklenecektir.

**Mekanik Değişiklik:**
Şu anki sabit veya serbest öğrenilen `lambda` (decay) parametresi yerine, eğitim adımına bağlı bir çarpan (scalar) eklenecektir.

$$
M_t = (\lambda \cdot \alpha_t) M_{t-1} + (1 - \lambda) (K_t V_t^T)
$$

Burada $\alpha_t$ (Alpha), zamanla azalan bir **Plastisite Katsayısıdır.**

*   **Çocukluk (Stage 1):** $\alpha \approx 0.1$ (Hafıza çok geçirgen, her şeyi yazıyor, çabuk unutuyor).
*   **Gençlik (Stage 2):** $\alpha \approx 0.5$ (Denge).
*   **Yetişkinlik (Stage 3):** $\alpha \rightarrow 0.99$ (Hafıza dirençli, sadece çok güçlü sinyaller (gradients) hafızayı değiştirebilir).

---

## 4. Uygulama Planı (Implementation Tasks)

Geliştirici ekip için iş paketleri:

### Görev 1: Veri Hazırlığı (`data`)
*   [ ] `src/data/curriculum.py` oluşturulması.
*   [ ] Hugging Face üzerinden `turkish-dictionary` ve `turkish-children-stories` veri setlerinin entegrasyonu.
*   [ ] `Wikipedia` veri setinin (mevcut clean script ile) son aşama olarak bağlanması.

### Görev 2: Hafıza Modülü Güncellemesi (`model`)
*   [ ] `src/models/memory.py` içine `set_plasticity(step)` metodunun eklenmesi.
*   [ ] `forward` fonksiyonunda `lambda` parametresinin dışarıdan gelen katsayı ile manipüle edilmesi.

### Görev 3: Eğitim Döngüsü (`train`)
*   [ ] Yeni `train_curriculum.py` scriptinin yazılması.
*   [ ] Eğitim döngüsünde her N adımda bir veri yükleyicinin (DataLoader) ve Plastisite katsayısının güncellenmesi mantığının kurulması.

---

## 5. Beklenen Etki (Impact Analysis)

Bu mimari değişiklik uygulandığında:
1.  **Halüsinasyon Azalması:** Model, kelime köklerini ilk aşamada "ezberlediği" için, olmayan kelimeler (örn: *ekrekiyetin*) türetme oranı düşecektir.
2.  **Mantıksal Tutarlılık:** Basit cümlelerden karmaşığa geçiş, modelin "cümlenin sonunu getirme" yeteneğini güçlendirecektir.
3.  **Konverjans Hızı:** Başlangıçta basit veri kullanıldığı için Loss değeri çok daha hızlı düşecek, eğitim maliyeti azalacaktır.
