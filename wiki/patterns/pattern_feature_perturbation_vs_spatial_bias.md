# Pattern: Feature Perturbation vs. Gaussian Spatial Bias

- **Type**: Theoretical Comparison & Architecture
- **Status**: Active (ARCH-4 Superiority)

## 💡 Comparative Mechanism

### 1. PerturbCTC (Feature Perturbation Approach)
PerturbCTC applies stochastic convolutional perturbations $\delta \sim \text{Conv}(\mathbf{F})$ during training to prevent the network from relying on brittle feature shortcuts. A KL-divergence loss minimizes the gap between the perturbed prediction $p(\mathbf{y} \mid \mathbf{F} + \delta)$ and an approximated target posterior.
- **Advantage**: Simple to train with zero inference latency overhead.
- **Disadvantage**: Does not fix alignment drift when test images have severe geometric perspective tilt or specular reflection.

### 2. EditCTC ARCH-4 (Analytical Gaussian Spatial Bias)
Rather than perturbing features, ARCH-4 computes an explicit coordinate bridge between the 1D CTC output space and 2D visual feature grid:
$$\mathbf{B}_{i, k} = \lambda \cdot \exp\left(-\frac{(u_k - c_i)^2}{2\sigma^2}\right)$$
- **Advantage**: Guaranteed spatial focus during inference. Even if visual features have background noise or identical repeating characters (e.g. `00000`), the attention mechanism is mathematically bounded within $3\sigma$ of the character center.
