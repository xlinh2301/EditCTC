# Pattern: Height-Progressive Merging vs. Shared High-Resolution Visual Memory

- **Type**: Visual Backbone Strategy
- **Status**: Active (ARCH-4 / ARCH-5 Component)

## 💡 Resolution Trade-offs in Visual Text Recognition

### 1. Standard SVTR / SVTRv2 Height-Progressive Merging
SVTRv2 merges features progressively:
$$\frac{H}{2^i} \times \frac{W}{4} \times D_i \to \frac{H}{2^{i+1}} \times \frac{W}{4} \times D_{i+1}$$
By reducing vertical height $H$ to $1$ (e.g. $1 \times 40$), the visual representation becomes 1D.
- **Problem in Utility Meter Dials**: When height is compressed to $1$, the vertical gap between the top and bottom loops of digits like `8` and `9`, or the horizontal middle bar of `3` vs `8`, collapses into a single averaged vector.

### 2. EditCTC Shared High-Resolution Visual Memory
EditCTC maintains a 2D spatial grid at $1/4$ resolution ($H/4 \times W/4 = 4 \times 96 = 384$ tokens):
- **Conv2D + LayerNorm Projection**: Projects $384$-channel backbone features into $384$-dimensional key/value visual memory.
- **Micro-Stroke Topology Preservation**: The 4 vertical slices preserve the distinct upper vs lower loop closures of ambiguous digits, enabling the 4-layer Transformer decoder to distinguish $8 \leftrightarrow 9$ and $3 \leftrightarrow 8$ under glare.
