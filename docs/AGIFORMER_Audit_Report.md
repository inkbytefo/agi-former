# AGIFORMER v1.0: Birleştirilmiş Mimari Denetim & Gelecek Yol Haritası

**Tarih:** 23.11.2025
**Denetçi:** Antigravity (Yapay Zeka Asistanı)
**Durum:** Kod Tabanı ile Doğrulandı

## 1. Yönetici Özeti

Kapsamlı bir "Red Team" analizi ve `src/models/` dizininin doğrudan kod incelemesinin ardından, ön raporlarda belirtilen **3 Kritik Yapısal Zayıflık** ve **2 Kaçırılmış Fırsatın** mevcut AGIFORMER v1.0 kod tabanında **fiilen mevcut olduğunu** onaylıyorum.

Mevcut mimari (Faz 8), sağlam bir "Kavram Kanıtı" (Ford T) niteliğindedir, ancak GPT-4 seviyesinde performans (Ferrari) için gereken dinamik uyarlanabilirlikten yoksundur. "Kör Hafıza" ve "Sert Yamalama" mekanizmaları, modelin karmaşık akıl yürütme görevlerine ölçeklenmesini engelleyen en önemli darboğazlardır.

---

## 2. Doğrulanmış Zafiyetler (Kod Tabanı Kanıtları)

### 🔴 1. Veriden Bağımsız Unutma ("Kör" Hafıza)
*   **Ciddiyet:** Kritik
*   **Konum:** `src/models/memory.py` (Satır 39, 88-89)
*   **Kanıt:**
    ```python
    self.decay_logits = nn.Parameter(torch.tensor([8.0] * num_heads))
    # ...
    lambdas = 0.995 + (0.005 * raw_sigmoid) # Kafa başına statik değer
    ```
*   **Etki:** Model *küresel* bir unutma hızı öğrenir. Dolgu metnini görmezden gelirken belirli bir şifreyi "aklında tutmaya" karar veremez. Her şeyi aynı sabit hızda unutur.

### 🔴 2. Yama Sınırlarında Süreksizlik ("Kekeleme")
*   **Ciddiyet:** Yüksek
*   **Konum:** `src/models/agiformer.py` (`LocalAutoregressiveHead`)
*   **Kanıt:** Dekoder, *her* yama için `None` (sıfır durum) ile başlayan bir GRU kullanır.
    ```python
    out, _ = self.rnn(rnn_input) # Gizli durum bir sonraki yamaya AKTARILMAZ
    ```
*   **Etki:** Model her 4 baytlık sınırda hafıza kaybı yaşar. Bağlamı yalnızca küresel gizli vektörden yeniden oluşturmak zorundadır, bu da metin üretiminde potansiyel "aksaklıklara" veya ritim bozukluklarına yol açar.

### 🔴 3. Kör Akıl Yürütme (Sistem 2 İzolasyonu)
*   **Ciddiyet:** Yüksek
*   **Konum:** `src/models/reasoning.py`
*   **Kanıt:** `RecurrentReasoningBlock`, sabit `thinking_steps=3` boyunca döngüye girer ve gizli vektör `z`'yi izole bir şekilde dönüştürür.
    *   Düşünme süreci sırasında `HebbianMemory`'ye **erişimi yoktur**.
    *   Basit belirteçler (örn. "ve") için erken çıkış yapamaz.
*   **Etki:** Verimsiz işlem kullanımı ve akıl yürütme aşamasında gerçekleri "arayıp bulma" yeteneğinin olmaması.

### 🟠 4. İlkel Birleştirme ("Gürültülü" Karışım)
*   **Ciddiyet:** Orta
*   **Konum:** `src/models/layers.py` (`HybridBlock`)
*   **Kanıt:**
    ```python
    x = residual + self.out_proj(attn_out + memory_out)
    ```
*   **Etki:** Model gürültüyü kapılayamaz (filtreleyemez). Hafıza ilgisiz çağrışımlar döndürürse, bunlar zorla Dikkat çıktısına eklenir. Bir kapılama mekanizması (örn. `sigmoid(alpha) * Mem + (1-alpha) * Attn`) eksiktir.

### 🟠 5. Sert Yamalama & Seyrek Gömülüler
*   **Ciddiyet:** Orta
*   **Konum:** `src/models/encoder.py`
*   **Kanıt:**
    *   `Conv1d(stride=4, kernel_size=4)`: Örtüşme yok.
    *   `nn.Embedding(256, d_model)`: Çok seyrek girdi temsili.
*   **Etki:** Yama sınırlarında bilgi kaybı ve ilk katmanlarda bayt anlambiliminin "sığ" bir şekilde anlaşılması.

---

## 3. Stratejik Yol Haritası: AGIFORMER v2.0 ("Ferrari" Yükseltmesi)

Bu bulgulara dayanarak, bir sonraki ana sürüm için aşağıdaki mimari değişiklikler zorunludur.

| Bileşen | Önerilen Yükseltme | Beklenen Fayda |
| :--- | :--- | :--- |
| **Hafıza** | **Girdiye Bağlı Unutma** (`Mamba` tarzı) | $\lambda_t = \sigma(W x_t)$. Modelin önemli bilgileri dinamik olarak hafızaya "kilitlemesini" sağlar. |
| **Dekoder** | **Durumlu (Stateful) RNN / MLP** | GRU gizli durumunu yamalar arasında taşıyın VEYA darboğazı kaldırmak için paralel bir MLP dekoderine geçin. |
| **Akıl Yürütme** | **Hafıza Destekli & Uyarlanabilir** | Düşünme döngüsüne Hafıza ile Çapraz Dikkat (Cross-Attention) ekleyin. Erken çıkış için ACT (Uyarlanabilir İşlem Süresi) kullanın. |
| **Kodlayıcı** | **Örtüşen Yamalar (Yumuşak)** | `kernel_size=6`, `stride=4` olarak değiştirin. Daha pürüzsüz bayt entegrasyonu için bir "kayan pencere" etkisi yaratır. |
| **Birleştirme** | **Kapılı (Gated) Artıklar** | Yerel Dikkat ile Küresel Hafızayı dengelemek için öğrenilmiş bir kapı kullanın. |
| **Çekirdek** | **SwiGLU & RMSNorm** | Daha iyi gradyan akışı ve kapasite için MLP ve Normalizasyon katmanlarını modernize edin. |

---

## 4. Sonuç & Sonraki Adımlar

Mevcut **Faz 8** eğitimi boşa değildir. Saf bir lineer dikkat modelinin "temel zekasını" ölçmek için çok önemli bir kıyaslama noktası görevi görür.

**Acil Eylem Planı:**
1.  **Faz 8'i Tamamla:** Bir kıyaslama noktası oluşturmak için mevcut eğitimin bitmesine izin verin.
2.  **`test_recall.py` Çalıştır:** "Kör Hafıza"nın *tam olarak* ne kadar kötü olduğunu deneysel olarak ölçmemiz gerekiyor. Model "Samanlıkta İğne" testinde başarısız olursa, Yükseltme #1'in aciliyetini doğrular.
3.  **v2.0 Dalını Hazırla:** Model eğitilirken ayrı bir dalda `InputDependentMemory` ve `StatefulDecoder` sınıflarını kodlamaya başlayın.

**Nihai Karar:** "Red Team" haklı. Güçlü bir temel inşa ediyoruz, ancak "çatının" (akıl yürütme ve uzun vadeli hatırlama) v2.0'da onarılması gereken yapısal çatlakları var.
