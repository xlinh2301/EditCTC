# Pattern: Non-Autoregressive Gated Inference Policy

- **Type**: Strategy / Inference Algorithm
- **Status**: Active

## 💡 Dual-Threshold Gating Policy
To prevent spurious replacements at inference time, EditCTC enforces a strict dual-condition gating rule:

$$\hat{y}_i = \begin{cases} 
\text{argmax}_{v \in \mathcal{V}} P_{vocab, i}(v) & \text{if } P_{change, i} \ge \tau_{change} \;\land\; (P_{vocab, i}(\hat{v}_{best}) - P_{vocab, i}(s_i)) \ge \Delta_{margin} \\
s_i & \text{otherwise (preserve seed)}
\end{cases}$$

### Calibrated Thresholds:
- **$\tau_{change}$ (Change Threshold)**: $0.50$ (Strict binary confidence)
- **$\Delta_{margin}$ (Margin Threshold)**: $0.15$ (Ensures the proposed replacement has substantially stronger visual support than the seed)

---

# Pattern: Multi-Seed Benchmark Protocol (52 Seed Runs)

- **Type**: Convention / Verification Gate
- **Status**: Standardized

## 💡 Protocol Details
To prevent cherry-picking and evaluate true statistical robustness, every architecture variant is trained across **5 to 6 independent random seeds** with fixed test splits:
- **In-Domain Test Set**: 585 clean, verified utility meter crops.
- **Cross-Data Test Set**: 1,145 challenging outdoor, tilted, and wet meter crops.
- **Metrics Tracked**:
  - Sequence-level Accuracy ($\%$)
  - Character Error Rate (CER $\%$)
  - Over-correction Rate ($\%$)
  - Seed variance ($\sigma_{\text{acc}}$)
