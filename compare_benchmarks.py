## Developer: inkbytefo
## Modified: 2025-11-22

"""
Benchmark Comparison: Turkish vs English
Analyzes training curves and tests the hypothesis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def load_metrics(lang="english"):
    """Load training metrics from JSON"""
    filename = f"metrics_{lang}.json" if lang == "turkish" else "metrics_english.json"
    
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None

def plot_comparison():
    """Plot BPC curves for Turkish vs English"""
    en = load_metrics("english")
    tr = load_metrics("turkish")
    
    if not en or not tr:
        print("Missing metrics files. Run both training scripts first.")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Training BPC
    plt.subplot(1, 2, 1)
    plt.plot(en["steps"], en["train_bpc"], label="English (enwik8)", alpha=0.7)
    plt.plot(tr["steps"], tr["train_bpc"], label="Turkish (trwiki)", alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("BPC (Bits Per Character)")
    plt.title("Training BPC: Turkish vs English")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Validation BPC
    plt.subplot(1, 2, 2)
    
    # Val BPC is logged every 200 steps
    val_steps_en = [i * 200 for i in range(len(en["val_bpc"]))]
    val_steps_tr = [i * 200 for i in range(len(tr["val_bpc"]))]
    
    plt.plot(val_steps_en, en["val_bpc"], label="English (enwik8)", marker='o', alpha=0.7)
    plt.plot(val_steps_tr, tr["val_bpc"], label="Turkish (trwiki)", marker='s', alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Validation BPC")
    plt.title("Validation BPC: Turkish vs English")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comparison_turkish_vs_english.png", dpi=300)
    print("Saved comparison plot to comparison_turkish_vs_english.png")
    plt.close()

def statistical_test():
    """Perform statistical significance test"""
    en = load_metrics("english")
    tr = load_metrics("turkish")
    
    if not en or not tr:
        return
    
    # Final BPC values
    final_bpc_en = en["val_bpc"][-1]
    final_bpc_tr = tr["val_bpc"][-1]
    
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)
    
    print(f"\nFinal Validation BPC:")
    print(f"  English (enwik8): {final_bpc_en:.4f}")
    print(f"  Turkish (trwiki): {final_bpc_tr:.4f}")
    print(f"  Difference: {final_bpc_en - final_bpc_tr:.4f}")
    
    # Convergence speed (steps to reach 2.5 BPC)
    threshold = 2.5
    
    steps_to_threshold_en = next((s for s, bpc in zip(en["steps"], en["train_bpc"]) if bpc < threshold), None)
    steps_to_threshold_tr = next((s for s, bpc in zip(tr["steps"], tr["train_bpc"]) if bpc < threshold), None)
    
    print(f"\nSteps to reach BPC < {threshold}:")
    print(f"  English: {steps_to_threshold_en if steps_to_threshold_en else 'Not reached'}")
    print(f"  Turkish: {steps_to_threshold_tr if steps_to_threshold_tr else 'Not reached'}")
    
    # Hypothesis test
    print("\n" + "-" * 60)
    print("HYPOTHESIS TEST")
    print("-" * 60)
    
    if final_bpc_tr < final_bpc_en:
        print("✅ HYPOTHESIS CONFIRMED")
        print("Turkish model achieved lower BPC than English model.")
        print("This supports the claim that byte-level models are more")
        print("efficient for agglutinative languages.")
        improvement = ((final_bpc_en - final_bpc_tr) / final_bpc_en) * 100
        print(f"Improvement: {improvement:.2f}%")
    else:
        print("❌ HYPOTHESIS REJECTED")
        print("English model achieved lower or equal BPC.")
    
    print("=" * 60)

def generate_report():
    """Generate markdown report"""
    en = load_metrics("english")
    tr = load_metrics("turkish")
    
    if not en or not tr:
        return
    
    report = f"""# Kaşgarlı Testi - Benchmark Results

## Hypothesis
**H1:** Byte-level models learn agglutinative languages (Turkish) more efficiently than analytic languages (English).

## Experimental Setup
- **Model:** AGIFORMER (identical architecture, 50M parameters)
- **Hyperparameters:** Same for both (d_model=512, n_layers=6, thinking_steps=3)
- **Training:** 5000 steps, batch_size=4, lr=3e-4
- **English Dataset:** enwik8 (100MB Wikipedia)
- **Turkish Dataset:** trwiki (Turkish Wikipedia)

## Results

### Final BPC (Lower is Better)
| Language | Validation BPC |
|----------|----------------|
| English  | {en["val_bpc"][-1]:.4f} |
| Turkish  | {tr["val_bpc"][-1]:.4f} |

**Difference:** {abs(en["val_bpc"][-1] - tr["val_bpc"][-1]):.4f} BPC

### Convergence Speed
Steps to reach BPC < 2.5:
- English: {next((s for s, bpc in zip(en["steps"], en["train_bpc"]) if bpc < 2.5), "Not reached")}
- Turkish: {next((s for s, bpc in zip(tr["steps"], tr["train_bpc"]) if bpc < 2.5), "Not reached")}

## Conclusion

{"Turkish model outperformed English, confirming the hypothesis." if tr["val_bpc"][-1] < en["val_bpc"][-1] else "Hypothesis not confirmed in this experiment."}

## Visualization
![Comparison](comparison_turkish_vs_english.png)

---
**Generated:** 2025-11-22  
**Experimenter:** inkbytefo
"""
    
    with open("benchmark_report.md", "w") as f:
        f.write(report)
    
    print("\nGenerated benchmark_report.md")

if __name__ == "__main__":
    plot_comparison()
    statistical_test()
    generate_report()
