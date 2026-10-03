# Pattern: Hybrid Dial Pointer Trigonometry & Digit Roller OCR Fusion

## 1. Architectural Taxonomy: Dial Pointers vs. Roller Counters
Water meters in multi-utility deployments belong to two primary mechanical display paradigms:
1. **Digit Roller Counters (Primary Reading)**: Linear mechanical word-wheels displaying cumulative consumption in cubic meters ($m^3$).
2. **Circular Dial Pointers (Sub-Unit Gauges)**: 4 to 6 circular dials with mechanical rotating needles measuring fractional units ($\times 0.1, \times 0.01, \times 0.001, \times 0.0001\text{ m}^3$).

```
                      ┌──────────────────────────────────────┐
                      │    Water Meter Faceplate Image       │
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    Digit Roller Branch    │                   │    Dial Pointer Branch    │
   │  (PPLCNetV4 + EditCTC)    │                   │ (Keypoint / Angle Det)    │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │ Sequence Output: "00248"                      │ Needle Angle θ_k
                 ▼                                               ▼
                 │                                 v_k = (θ_k / 2π) * 10 mod 10
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │    Continuous Carry-Over & Fusion Engine      │
                 │   V_total = V_roller + Σ_k (v_k * 10^(-k))    │
                 │   Rule: Enforce Unit Drum Consistency         │
                 └───────────────────────────────────────────────┘
```

---

## 2. Dial Pointer Mathematical Formulation

### A. Needle Orientation Trigonometry
Given detected dial center $\mathbf{c} = (x_0, y_0)$ and needle tip coordinate $\mathbf{p} = (x_1, y_1)$ detected via Lite-FCOS or Keypoint R-CNN:
\[
\Delta x = x_1 - x_0, \quad \Delta y = -(y_1 - y_0) \quad (\text{screen inversion})
\]
The raw pointer angle $\theta \in [0, 2\pi)$ is calculated via the 2-argument arctangent:
\[
\theta = \operatorname{arctan2}(\Delta y, \Delta x) \pmod{2\pi}
\]

### B. Scale Value Mapping
For a 10-division clockwise circular gauge with scale zero aligned at vertical top ($\pi/2$):
\[
\phi = \left(\frac{\pi}{2} - \theta\right) \pmod{2\pi}
\]
\[
v_{\text{pointer}} = \frac{\phi}{2\pi} \times 10.0 \in [0.0, 10.0)
\]

---

## 3. Roller-Pointer Consistency Verification & Carry-Over Rules

In physical water meters, the lowest-order roller drum (e.g. $1\text{ m}^3$ wheel) is directly geared to the highest-order dial pointer ($\times 0.1\text{ m}^3$). 
- When the $\times 0.1$ pointer is in the range $[0.0, 1.0)$, the roller wheel should be entering a new integer state.
- When the $\times 0.1$ pointer is in the range $[9.0, 10.0)$, the roller wheel is in a transition state $(d \rightarrow d+1)$.

### Fusion Consistency Constraint
\[
\text{Expected Roller State } \hat{d}_{\text{roller}} = \begin{cases} 
d, & \text{if } v_{\times 0.1} \in [0.0, 9.0) \\
d \text{ transitioning to } d+1, & \text{if } v_{\times 0.1} \in [9.0, 10.0)
\end{cases}
\]

If EditCTC's CTC head outputs high entropy $\mathcal{H}(p) > 1.2$ at the unit digit, the fused system uses $v_{\times 0.1}$ to deterministically disambiguate whether the carry-over has completed, eliminating rolling-drum errors.
