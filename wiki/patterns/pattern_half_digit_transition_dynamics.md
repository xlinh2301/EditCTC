# Pattern: Half-Digit Transition Dynamics on Mechanical Meter Wheels

- **Type**: Domain Phenomenon & Mitigation Strategy
- **Status**: Active (Addressed via ARCH-4C Continuous Uncertainty + Decoupled Refinement)

---

## 1. Physical Problem: Continuous Mechanical Wheel Rotation
In mechanical utility meters (water, gas, electric), digit drums rotate continuously as fluid flows. At any given moment, a digit wheel may be midway through a physical transition (e.g., transitioning between `3` and `4`, or `8` and `9`):

```text
+-------------------+
|  [Top Half of 3]  |  <--- Outgoing digit moving up
| - - - - - - - - - |  <--- Viewing Window Centerline
| [Bottom Half of 4]|  <--- Incoming digit entering from bottom
+-------------------+
```

### Traditional Approaches & Limitations:
1. **10-Class Standard Convention (Lowest-Digit Rule)**:
   - Constrains output to integers $\{0, \dots, 9\}$. Rolling state between $d$ and $d+1$ is truncated to $d$ ($9 \to 0$ exception labeled $9$).
   - *Failure*: Introduces off-by-one errors when higher-order drums are transitioning before the lowest drum completes a cycle.
2. **20-Class Extended Vocabulary Formulation**:
   - Explicitly defines 20 classes: $\{0, \dots, 9\}$ for static digits, and $\{10, \dots, 19\}$ for rolling half-characters where class $s \in [10, 19]$ denotes transition from integer $(s - 10)$ to $(s - 9)$ (e.g., class $12$ denotes $2 \to 3$, outputting decimal $2.5$).
   - *Failure*: Rigid discretization cannot handle varying intermediate stages ($7a, 7b, 7c$) and requires complex multi-class re-labeling of vast datasets.
3. **Bottom-Portion Visual Focus**:
   - Incoming digit characteristics appear at the bottom edge of the drum aperture. Attention mechanisms must prioritize the lower vertical receptive field.

---

## 2. EditCTC Continuous Uncertainty Formulation (ARCH-4C)

Instead of forcing discrete 20-class labeling, EditCTC leverages continuous posterior distributions from the CTC head:

```
CTC Softmax Distribution p(y_t | X)
            │
            ▼
Extract Top-2 Probabilities & Shannon Entropy
c_i = [ p_1,  p_2,  p_1 - p_2,  H(p) ]
            │
            ▼
Continuous Uncertainty MLP Embedding: E_conf(c_i)
            │
            ▼
q_i = E_tok(s_i) + tanh(α) * E_conf(c_i)
            │
            ▼
Transformer Decoder Cross-Attends to Visual Memory with Gaussian Spatial Prior
            │
            ▼
Resolves Decimal Ambiguity via Neighboring Drum Carry-Over Dynamics
```

### Key Equations:
1. **Shannon Entropy Indicator**:
   \[
   \mathcal{H}(p) = -\sum_{c \in \mathcal{V}} p(c) \log p(c)
   \]
   At transition boundaries, $\mathcal{H}(p) > 1.2$ and $(p_1 - p_2) < 0.10$.
2. **Gated Query Injection**:
   \[
   \mathbf{q}_i = \mathbf{E}_{\text{tok}}(s_i) + \tanh(\alpha) \cdot \mathbf{W}_{\text{conf}} \mathbf{c}_i
   \]
3. **Right-to-Left Carry-Over Validation**:
   The 4-layer Transformer decoder attends to the visual feature tokens of wheel $i+1$. If wheel $i+1$ has not crossed the 0.0 rotation threshold (i.e. dial angle $< 0.8$ cycle), wheel $i$ is validated as the lower digit floor; otherwise, the incremented digit is emitted.
