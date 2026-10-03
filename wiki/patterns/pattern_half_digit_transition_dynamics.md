# Pattern: Half-Digit Transition Dynamics on Mechanical Meter Wheels

- **Type**: Domain Phenomenon & Mitigation Strategy
- **Status**: Active (Addressed via ARCH-4C Continuous Uncertainty)

## 💡 Physical Problem: Continuous Mechanical Wheel Rotation
In mechanical utility meters (water, gas, electric), digit drums rotate continuously as fluid flows. At any given moment, a digit wheel may be midway through a transition (e.g., transitioning between `3` and `4`, or `8` and `9`):

```text
[Top Half of 3]
--------------  <-- Viewing Window Boundary
[Bottom Half of 4]
```

### Failure Modes in Standard Models:
1. **Discrete 10-Class Classification Failure**: Standard models forced to emit a single discrete label fluctuate randomly between 3 and 4.
2. **CTC High-Entropy Peak**: The CTC softmax distribution at transition timesteps exhibits high entropy ($\mathcal{H} > 1.2$) and near-equal top-1/top-2 probabilities ($p_1 \approx 0.48, p_2 \approx 0.45$).
3. **Carry-Over Inconsistency**: If the lower digit has not completed a full revolution ($9 \to 0$), reading the upper wheel as the next digit produces large measurement errors (e.g. reading $149.9 \text{ m}^3$ instead of $139.9 \text{ m}^3$).

---

## 🛠 Mitigation via ARCH-4C & Visual Context Verification

1. **Entropy Peak Signaling**:
   - The 4D confidence vector $\mathbf{c}_i = [p_1, p_2, p_1 - p_2, \mathcal{H}(p)]$ flags the transition state because the margin $p_1 - p_2 < 0.10$ and entropy $\mathcal{H} \ge 1.0$.
2. **Context-Aware Visual Refinement**:
   - Gated query $\mathbf{q}_i = E_{tok}(s_i) + \tanh(\alpha) \cdot \text{MLP}(\mathbf{c}_i)$ signals the 4-layer Transformer decoder to cross-attend to neighboring digits in visual memory.
3. **Mechanical Carry-Over Rule Validation**:
   - By observing the visual state of the right-adjacent wheel (e.g., whether wheel $i+1$ is near $0.0$ or $9.0$), the decoder resolves the carry-over state correctly, choosing the floor digit when the sub-dial is $< 0.8$ of rotation.
