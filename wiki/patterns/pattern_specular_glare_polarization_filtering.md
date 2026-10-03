# Pattern: Specular Glare, Droplet Refraction & Geometric Rectification

## 1. Problem Definition: Optical Degradation in AMR
Water meter inspection cameras are deployed in harsh field environments (underground pits, outdoor enclosures, industrial pump stations) and encounter severe optical disturbances:
1. **Specular Glass Reflection & Flare**: Direct sunlight or artificial LED ring flash reflects off the convex/flat protective glass cover, washing out digit stroke contours.
2. **Water Droplets & Internal Condensation**: Droplets act as micro-lenses, producing severe astigmatic distortion, refractive warping, and localized occlusions.
3. **Camera Perspective Tilt**: Non-perpendicular shooting angles introduce trapezoidal skew and out-of-plane rotation.
4. **Sub-Pixel Misalignment**: Coarse pooling operations introduce coordinate rounding errors that distort small 10-15px digit glyphs.

---

## 2. Geometric & Preprocessing Formulations

```
Raw Tilted & Degraded Frame
       │
       ▼
[ WMRR Scale Shrinking ] ──► D = S(1 - r²) / L (Suppress glare & border artifacts)
       │
       ▼
[ 2D Affine Rectification ] ──► M(θ) = [[cos θ, -sin θ, 0], [sin θ, cos θ, 0], [0, 0, 1]]
       │
       ▼
[ Bilinear ROI Align ] ──► Exact floating-point sampling (Eliminates ROI Pooling rounding error)
       │
       ▼
[ Adaptive Contrast & Denoising ] ──► CLAHE + Otsu + Dark-Pit Histogram Equalization
       │
       ▼
Shared High-Res Visual Memory (1/4 scale, 384 tokens)
```

### A. Water Meter Reading Region (WMRR) Scale Shrinking
Directly cropping the bounding box around the counter window captures boundary metal frames, casing screws, and glass edge flares. Applying a scale shrinking factor $D$ derived from DBNet morphological geometry isolates the clean digit region (WMRDR):
\[
D = \frac{S(1 - r^2)}{L}
\]
where $S$ is the polygon area, $L$ is the perimeter, and $r \in [0.35, 0.45]$ (typically $r = 0.4$) is the contraction ratio.

### B. 2D Affine Rotation Rectification
For an oriented bounding box with detected inclination angle $\theta$, horizontal alignment is achieved by applying a 2D affine transformation matrix $\mathbf{M} \in \mathbb{R}^{3 \times 3}$:
\[
\mathbf{M} = \begin{pmatrix} 
\cos\theta & -\sin\theta & (1-\cos\theta)x_c + \sin\theta y_c \\
\sin\theta & \cos\theta & -\sin\theta x_c + (1-\cos\theta)y_c \\
0 & 0 & 1
\end{pmatrix}
\]
where $(x_c, y_c)$ is the geometric center of the counter patch.

### C. Continuous Bilinear ROI Align
Standard ROI Pooling applies integer rounding twice: (1) during candidate box projection onto feature maps $\lfloor x / 16 \rfloor$, and (2) during bin partitioning $\lfloor w / k \rfloor$. On small digit regions ($24 \times 16$ px), this quantization causes a 25-30% spatial shift of micro-strokes.
**ROI Align** avoids coordinate quantization by sampling 4 regular bilinear points per bin:
\[
f(x, y) = \sum_{i,j=1}^2 w_{ij} f(x_i, y_j), \quad w_{ij} = (1 - |x - x_i|)(1 - |y - y_j|)
\]
yielding a **+3.5% mAP boost** in digit detection.

---

## 3. High-Resolution Visual Memory in EditCTC
Rather than relying solely on aggressive input-space thresholding (which destroys subtle half-digit stroke gradients under shadow), EditCTC uses `SharedHighResVisualMemory`:
- Backbone preserves **$1/4$ resolution ($4 \times 96 = 384$ tokens)** at Layer 4 instead of collapsing to $1/8$ ($2 \times 48 = 96$ tokens).
- Cross-attention queries with Gaussian Spatial Bias attend directly to high-frequency stroke details, disambiguating droplet-refracted glyphs ($8 \leftrightarrow 9$, $3 \leftrightarrow 8$, $0 \leftrightarrow 6$).
