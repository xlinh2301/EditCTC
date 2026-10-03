# Pattern: Decoupled Change vs. Token Heads (ARCH-1)

- **Type**: Architecture & Optimization Strategy
- **Status**: Active

## 💡 Problem Description: The Over-Correction Pathology
When a single 97-way classification head is trained to perform both token verification and sequence substitution, the cross-entropy gradient tends to encourage replacing correct seed tokens with frequent dataset priors. This causes severe **over-correction**, where accurate CTC predictions are corrupted during post-editing.

## 🛠 Architectural Solution: Decoupled Multi-Task Objective
EditCTC decouples the decision into two separate prediction heads:
1. **Change Head**: A binary linear classifier predicting whether the token at position $i$ requires modification ($z_i \in \{0, 1\}$):
   $$\hat{P}_{change, i} = \sigma(W_{ch}^T h_i + b_{ch})$$
   $$\mathcal{L}_{change} = \text{BCEWithLogits}(W_{ch}^T h_i, z_i)$$
2. **Edit Token Head**: A 97-way linear classifier predicting the replacement character vocabulary distribution:
   $$\hat{P}_{vocab, i} = \text{softmax}(W_{tok}^T h_i + b_{tok})$$
   $$\mathcal{L}_{token} = \text{CrossEntropy}(\hat{P}_{vocab, i}, y_i)$$

### Total Loss:
$$\mathcal{L}_{total} = \mathcal{L}_{ctc} + \gamma_{change} \mathcal{L}_{change} + \gamma_{token} \mathcal{L}_{token}$$
*(Optimal weights: $\gamma_{change} = 2.0, \gamma_{token} = 1.0$).*

## 📈 Empirical Validation
- **False Edit Rate**: Drops from $4.18\%$ in coupled baseline to **$0.08\%$** with decoupled heads.
- **In-Domain Accuracy Retention**: Preserves $93.09\%$ accuracy on clean inputs while boosting out-of-domain cross-data accuracy.
