# Auxiliary image-reconstruction branch.
#
# Hypothesis: the remaining recognition errors are blur/low-quality crops, and
# forcing the encoder to retain enough information to reconstruct a clean image
# will push it toward more robust, detail-preserving features.
#
# Caution this branch is built to test, not assume: a prior experiment already
# found that DOUBLING input resolution (48x320 -> 64x512) measurably HURT
# accuracy (94.30% vs 96.04% baseline), which is evidence the accuracy ceiling
# here is data-bound (label noise, distribution shift) rather than detail- or
# resolution-bound. If that finding holds, this branch is likely to show little
# or no benefit -- but it tests a distinct hypothesis (denoising/deblurring
# invariance, not raw pixel count) so it is worth the one experiment.
#
# Deliberately a regularizer like the length branch: it reads the backbone's
# own recon_feat (captured before the height-collapsing pool the CTC path
# uses) and is trained alongside CTC with a small weight. It never touches the
# CTC head or its input, so it cannot change what the recognizer outputs by
# construction -- only what the shared conv stages learn to keep.

from __future__ import absolute_import, division, print_function

import paddle
from paddle import nn
from paddle.nn import functional as F


class ReconBranch(nn.Layer):
    """Upsamples an intermediate backbone feature back toward the input
    resolution and predicts a clean reconstruction of it.

    Args:
        in_channels: channel width of the backbone's recon_feat.
        out_shape: (C, H, W) of the normalized input image to reconstruct.
    """

    def __init__(self, in_channels, out_shape=(3, 48, 320), hidden=64):
        super(ReconBranch, self).__init__()
        self.out_shape = out_shape
        self.proj = nn.Conv2D(in_channels, hidden, 1)
        self.deconv = nn.Sequential(
            nn.Conv2DTranspose(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2D(hidden),
            nn.ReLU(),
            nn.Conv2DTranspose(hidden, hidden // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2D(hidden // 2),
            nn.ReLU(),
            nn.Conv2D(hidden // 2, out_shape[0], 3, padding=1),
        )

    def forward(self, feat):
        x = self.proj(feat)
        x = self.deconv(x)
        x = F.interpolate(
            x, size=self.out_shape[1:], mode="bilinear", align_corners=False
        )
        return paddle.tanh(x)
