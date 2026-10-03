# Pattern: Fine-Stroke Character Confusion & Glare Mitigation

- **Type**: Failure Mode & Mitigation Strategy
- **Status**: Active

## 💡 Identified Character Confusion Clusters
Analysis of 1,145 cross-data test error trajectories reveals distinct digit-confusion pairs caused by outdoor physical degradation:

| Confusion Pair | Root Cause in Water Meter Images | Frequency (% of Errors) |
|:---|:---|:---|
| `8` $\leftrightarrow$ `9` | Lower loop closure obscured by vertical dial border reflection | 28.4% |
| `3` $\leftrightarrow$ `8` | Left vertical spine shadow creates false closure | 22.1% |
| `0` $\leftrightarrow$ `6` | Upper hook eroded by condensation droplets | 18.7% |
| `1` $\leftrightarrow$ `7` | Top horizontal stroke faded by sunlight overexposure | 14.2% |
| `5` $\leftrightarrow$ `6` | Top bar discontinuity due to low resolution | 9.8% |

---

## 🛠 Multi-Layer Mitigation Strategy

1. **High-Resolution Feature Retention (`SharedHighResVisualMemory`)**:
   - Preserves $1/4$ spatial resolution ($4 \times 96 = 384$ tokens) rather than standard $1/8$ pooling, retaining micro-stroke topology.
2. **Local Visual Refinement Block (ARCH-5)**:
   - Residual depthwise-separable conv layers ($3 \times 3$) prior to cross-attention enhance edge and curve continuity.
3. **Focal Change Loss Weighting**:
   - Applies class-balanced focal weighting $\alpha_t (1 - p_t)^\gamma$ to difficult digit pairs during Change Head training.
