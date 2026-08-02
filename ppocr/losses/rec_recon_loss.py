# L1 reconstruction loss for the optional auxiliary reconstruction branch.
# Target is the clean (pre-RecAug) image, normalized identically to the
# training image (see ReconTargetCapture / ReconTargetResize in
# ppocr/data/imaug/rec_img_aug.py) so the two are directly comparable.

from __future__ import absolute_import, division, print_function

from paddle import nn


class ReconLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(ReconLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return {"loss": self.l1(pred, target)}
