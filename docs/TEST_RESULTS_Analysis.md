# Model Intelligence Test Results

## Test Setup

**Model**: `best_model_curriculum.pth` (20K steps, Val BPC 1.78)  
**Device**: CUDA

Two tests executed:
1. **Curriculum Intelligence Test** - Evaluates learning across 3 stages
2. **Recall Test (Fixed)** - Needle-in-haystack memory test

---

## Test 1: Curriculum Intelligence

### Results

| Stage | Prompt | Model Output | Assessment |
|-------|--------|--------------|------------|
| **Stage 1 (Sözlük)** | "Elma kelimesinin anlamı: " | "16 Nisan 1914 tarihinde Ceneviz kullandı..." | ❌ No semantic understanding |
| **Stage 2 (Hikaye)** | "Küçük kedi bahçede koşarken " | "km yağında bazı kesimde kullanılır..." | ❌ Grammatically broken |
| **Stage 3 (Bilgi)** | "Türkiye'nin başkenti Ankara " | "Hande Kuru Yarımadası İstanbul..." | ⚠️ Partial recognition |
| **Stage 3 (Tarih)** | "1923 yılında Cumhuriyet " | "İstanbul Ali Danat Seçim görevinde..." | ⚠️ Context-aware but incoherent |

### Analysis

**What the Model Learned:**
- ✅ **Turkish morphology**: Proper suffixes, vowel harmony
- ✅ **Wikipedia patterns**: Names, dates, locations
- ✅ **Grammatical structure**: Subject-object patterns

**What the Model Struggles With:**
- ❌ **Semantic coherence**: Outputs are "grammatically Turkish" but nonsensical
- ❌ **Lexical grounding**: Dictionary stage didn't establish word meanings
- ❌ **Narrative flow**: Can't maintain topic across generations

**Why This Happens:**
- **Model size**: 31M parameters is very small for byte-level LM
- **Training data**: Only 20K steps (~150MB exposure)
- **Curriculum effectiveness**: Stages worked but insufficient duration

---

## Test 2: Recall Test (Needle in Haystack)

### Test Configuration

```
Context: 1,789 bytes
Needle: "1453" 
Position: Embedded mid-context
Query: "SORU: GİZLİ KOD NEDİR? CEVAP: "
```

### Result

**Expected**: `1453`  
**Model Output**: `Sen Akdeniz kul`

**Status**: ❌ **FAILED** - Model lost information in noise

### Analysis

**Problem**: Model failed to recall specific information from earlier in context

**Possible Causes:**

1. **Hebbian Memory Decay Too Aggressive**:
   - Stage 3 plasticity (α=0.99) should retain long-term memory
   - But decay parameter λ might still be too low
   - Effective decay: `λ * α = ~0.99 * 0.99 ≈ 0.98`

2. **Attention Window Limitation**:
   - Sliding window attention (128 tokens) may not capture full context
   - Hebbian memory supposed to handle global context, but may need tuning

3. **Training Duration**:
   - 20K steps insufficient for complex memory tasks
   - Model needs more exposure to question-answer patterns

**Why Hebbian Memory Didn't Help**:
- Model trained on continuous text (Wikipedia), not Q&A format
- No explicit training on information retrieval tasks
- Memory module learned compression, not targeted recall

---

## Overall Assessment

### Strengths
✅ **Numerical Stability**: 0 NaN in 20K steps (AMP fix working)  
✅ **Curriculum Mechanism**: 3-stage transitions executed correctly  
✅ **Morphological Learning**: Turkish structure internalized  
✅ **BPC Achievement**: 1.78 is respectable for 31M params

### Weaknesses
❌ **Semantic Understanding**: "Dreaming" rather than coherent outputs  
❌ **Long-term Recall**: Failed needle-in-haystack test  
❌ **Lexical Grounding**: Dictionary stage ineffective at current scale  
❌ **Context Utilization**: Not leveraging full Hebbian memory potential

---

## Recommendations

### Immediate Improvements

**1. Extended Training (Critical)**
```python
MAX_STEPS = 50000  # 20K → 50K
```
- More data exposure needed for semantic patterns
- Current 20K = ~150MB, target: 400-500MB

**2. Hebbian Memory Tuning**
```python
# In src/models/memory.py
# Current: lambdas = 0.99 + (0.01 * sigmoid(param))
# Proposed: lambdas = 0.995 + (0.005 * sigmoid(param))
# Higher base decay for better long-term retention
```

**3. Question-Answer Fine-tuning**
- Create QA dataset from Turkish Wikipedia
- Format: "Soru: ... Cevap: ..."
- 5K steps fine-tuning after main curriculum

### Architectural Changes

**1. Increase Model Size**
```python
D_MODEL = 768   # 512 → 768
N_LAYERS = 8    # 6 → 8
# ~100M parameters
```

**2. Adaptive Window Attention**
- Dynamically expand attention window for queries
- Detect question patterns → use full context

**3. Explicit Memory Module**
- Add retrieval-specific attention head
- Train with contrastive Q&A loss

### Alternative Approaches

**Option A: Sparse Curriculum**
- Stage 1: TDK Dictionary (focused, 10K steps)
- Stage 2: Skip (or minimal)
- Stage 3: Wikipedia (40K steps)

**Option B: Iterative Curriculum**
- Cycle through stages multiple times
- Each cycle with increasing difficulty
- 3 full cycles = 60K steps total

**Option C: Hybrid Training**
- Pre-train on Wikipedia (standard)
- Fine-tune with curriculum (dictionary → stories)
- Reverse order approach

---

## Conclusion

The curriculum mechanism **works mechanically** (stages transition, plasticity updates) but **doesn't achieve semantic intelligence** at current scale and duration.

**Key Insight**: 31M parameters + 20K steps = mechanical Turkish learned, semantic understanding NOT learned.

**Path Forward**: Either **scale up significantly** (100M+ params, 50K+ steps) OR **narrow scope** to specific tasks (QA, classification) rather than open-ended generation.

**Status**: Phase 7 infrastructure ✅ SOLID, but model capabilities ⚠️ LIMITED to current constraints.
