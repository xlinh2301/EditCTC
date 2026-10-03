# Pattern: Continuous CTC Uncertainty Injection (ARCH-2B / ARCH-4C)

- **Type**: Strategy / Architecture
- **Status**: Active (ARCH-4C Component)

## 💡 Problem Description: Binary vs. Continuous Confidence
Earlier error-correction pipelines passed only discrete token IDs to the decoder, completely discarding the rich probabilistic uncertainty emitted by the CTC head. Consequently, the decoder could not differentiate between a high-confidence prediction ($p = 0.99$) and a borderline guess ($p = 0.35$).

## 🛠 Mathematical Formulation: 4D Confidence Vector
For each decoded timestep $t_i$, the top-1 and top-2 softmax probabilities are extracted:
$$p_1 = \max_{v \in \mathcal{V}} P(y_{t_i} = v), \quad p_2 = \max_{v \in \mathcal{V} \setminus \{\hat{v}_1\}} P(y_{t_i} = v)$$

The 4D continuous uncertainty vector $\mathbf{c}_i \in \mathbb{R}^4$ is defined as:
$$\mathbf{c}_i = \begin{bmatrix} p_1 \\ p_2 \\ p_1 - p_2 \\ -\sum_{v \in \mathcal{V}} P(y_{t_i} = v) \log P(y_{t_i} = v) \end{bmatrix} = \begin{bmatrix} \text{Top-1 Prob} \\ \text{Top-2 Prob} \\ \text{Confidence Margin} \\ \text{Shannon Entropy} \end{bmatrix}$$

### Gated Injection into Query:
The 4D vector is projected through a 2-layer MLP to dimension $d = 384$ and fused with the discrete token embedding $E_{tok}(s_i)$ via a learnable scalar $\alpha$:
$$\mathbf{e}_{conf, i} = \text{MLP}_{64 \to 384}(\text{LayerNorm}(\mathbf{c}_i))$$
$$\mathbf{q}_i = E_{tok}(s_i) + \tanh(\alpha) \cdot \mathbf{e}_{conf, i}$$

Where $\alpha$ is initialized to $0.0$ to ensure stable early training without destabilizing the pretrained Transformer representation.

## 📈 Empirical Validation
- **Variance across Seeds**: ARCH-4C achieves the lowest standard deviation among all models ($\sigma = \pm 0.37\%$).
- **Selective Correction**: Low-confidence tokens ($p_1 < 0.6$) trigger refinement with $84.2\%$ precision, while high-confidence tokens ($p_1 > 0.95$) are preserved with $99.92\%$ fidelity.
