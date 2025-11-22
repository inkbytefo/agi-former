# Phase 6: AGI-Zeka (Cognitive Scaling) - Hebbian Memory Implementation

## Goal Description
To transition AGIFORMER from "Mechanical Intelligence" to "Semantic Intelligence", we will implement **Hebbian Memory (Fast Weights)**. This mechanism allows the model to maintain a dynamic, short-term memory state that updates with every token, mimicking synaptic plasticity. This addresses the "Wernicke's Aphasia" issue by enforcing stronger context retention and logical coherence.

## User Review Required
> [!IMPORTANT]
> This change introduces a new module `HebbianMemory` which effectively replaces or enhances the existing `LinearAttention`.
> We will introduce a **decay factor (lambda)** to the memory updates, making it a "Leaky" Hebbian Memory. This is crucial for handling long sequences without saturating the memory state.

## Proposed Changes

### 1. New Component: Hebbian Memory
Create a new file `src/models/memory.py` containing the `HebbianMemory` class.

#### [NEW] [memory.py](file:///c:/Users/tpoyr/OneDrive/Desktop/agiformer1/src/models/memory.py)
- **Class:** `HebbianMemory`
- **Functionality:**
    - Computes Queries, Keys, Values.
    - Implements the Fast Weight update rule: $W_t = \lambda W_{t-1} + \phi(K_t) V_t^T$.
    - Computes Output: $O_t = \phi(Q_t) W_t$.
    - Uses a learnable or fixed decay rate $\lambda$ (sigmoid parameterized) to control forgetting.
    - Numerically stable implementation using `cumsum` (parallel scan) for training efficiency.

### 2. Integration: Hybrid Block Update
Modify `src/models/layers.py` to use `HebbianMemory` instead of the standard `LinearAttention`.

#### [MODIFY] [layers.py](file:///c:/Users/tpoyr/OneDrive/Desktop/agiformer1/src/models/layers.py)
- Import `HebbianMemory` from `.memory`.
- Update `HybridBlock` to initialize `self.memory = HebbianMemory(...)` instead of `self.ssm = LinearAttention(...)`.
- Update `forward` pass to use `self.memory`.

### 3. Architecture Update
Ensure `AGIFORMER` in `src/models/agiformer.py` propagates the necessary config (though `d_model` and `num_heads` are likely sufficient).

#### [MODIFY] [agiformer.py](file:///c:/Users/tpoyr/OneDrive/Desktop/agiformer1/src/models/agiformer.py)
- (Optional) If we add specific memory hyperparameters (like initial decay), pass them down. For now, defaults should suffice.

## Verification Plan

### Automated Tests
1.  **Unit Test:** Create `tests/test_memory.py` to verify:
    - Shape consistency.
    - Gradient flow (backward pass).
    - Causal masking (ensure future tokens don't leak into memory).
2.  **Integration Test:** Run `test_turkish_model.py` again with the new memory module to see if "hallucinations" persist or if coherence improves (qualitative).

### Manual Verification
- Compare the "Story Generation" (Test 6) output before and after the change. We expect better consistency in the narrative.
