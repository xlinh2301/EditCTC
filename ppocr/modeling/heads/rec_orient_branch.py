# Auxiliary 0/180 orientation head for a unified recognition backbone.
#
# Part of the unified end-to-end model: one PP-LCNetV4 backbone serves both the
# CTC recognition head and this orientation classifier, so a single forward pass
# replaces the separate PP-LCNet orientation model (~2 ms saved). The head pools
# the shared sequence features and predicts whether the crop is upright (0) or
# inverted (180); at inference the caller rotates and re-reads only when 180 is
# predicted.
from __future__ import absolute_import, division, print_function

import paddle
from paddle import nn


class OrientBranch(nn.Layer):
    def __init__(self, in_channels, hidden=128):
        super(OrientBranch, self).__init__()
        # mean+max pooling: mean captures overall polarity, max the sharpest
        # orientation cue (e.g. the wheel-frame edge)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.LayerNorm(hidden),
            nn.Hardswish(),
            nn.Linear(hidden, 2),
        )

    def forward(self, feats):
        avg = paddle.mean(feats, axis=1)
        mx = paddle.max(feats, axis=1)
        return self.fc(paddle.concat([avg, mx], axis=1))
