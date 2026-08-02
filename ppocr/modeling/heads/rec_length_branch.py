# Auxiliary digit-count branch.
#
# CTC never states how many characters a crop contains -- the count falls out of
# blank collapsing. So when the last odometer wheel is half cut off by the crop
# boundary, the decoder may swallow it or emit it twice and no gradient objects.
# On the current test split that accounts for 5 of PPOCRv6's 17 errors.
#
# This branch predicts the count explicitly and is trained alongside CTC. It is
# deliberately a *regulariser*, not an arbiter: the recogniser already gets the
# length right on 556/561 crops, so letting the branch override the decode would
# need it to exceed 99.10% accuracy just to break even -- the same base-rate trap
# that sank the earlier refine stage. Supervising the encoder carries no such
# downside, because the CTC head still decides the output.

from __future__ import absolute_import, division, print_function

import paddle
from paddle import nn


class LengthBranch(nn.Layer):
    """Predicts the number of characters from the sequence features.

    Args:
        in_channels: width of the incoming (B, T, C) feature sequence.
        max_length: largest count the branch can emit; labels are clipped to it.
        hidden: bottleneck width before the classifier.
    """

    def __init__(self, in_channels, max_length=25, hidden=128):
        super(LengthBranch, self).__init__()
        self.max_length = max_length
        # Mean pooling alone dilutes one extra wheel across the whole sequence;
        # max pooling alone keeps only the single strongest column. Counting
        # needs the overall extent and the sharp per-column evidence, so both
        # are concatenated.
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.LayerNorm(hidden),
            nn.Hardswish(),
            nn.Linear(hidden, max_length + 1),  # index 0 is unused
        )

    def forward(self, feats):
        avg = paddle.mean(feats, axis=1)
        mx = paddle.max(feats, axis=1)
        return self.fc(paddle.concat([avg, mx], axis=1))


class LengthBranchAttn(nn.Layer):
    """Transformer-based digit-count branch (NRTR-style self-attention plus a
    learnable query token that cross-attends to the sequence), in place of
    the mean+max pooling of LengthBranch. Reuses the same building blocks as
    the NRTR recognition head (TransformerBlock, PositionalEncoding).

    Args:
        in_channels: width of the incoming (B, T, C) feature sequence.
        max_length: largest count the branch can emit; labels are clipped to it.
        hidden: bottleneck width before the classifier.
        nhead: attention heads (in_channels must be divisible by nhead).
        num_layers: self-attention TransformerBlocks applied to the sequence
            before the query token reads it.
    """

    def __init__(self, in_channels, max_length=25, hidden=128, nhead=4, num_layers=2):
        super(LengthBranchAttn, self).__init__()
        from ppocr.modeling.heads.rec_nrtr_head import PositionalEncoding, TransformerBlock
        from ppocr.modeling.backbones.rec_svtrnet import zeros_

        self.max_length = max_length
        self.pos_encoding = PositionalEncoding(dropout=0.1, dim=in_channels)
        self.encoder_layers = nn.LayerList(
            [
                TransformerBlock(
                    in_channels,
                    nhead,
                    dim_feedforward=in_channels * 2,
                    attention_dropout_rate=0.0,
                    residual_dropout_rate=0.1,
                    with_self_attn=True,
                    with_cross_attn=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.query = self.create_parameter(
            shape=[1, 1, in_channels], default_initializer=zeros_
        )
        self.pool_block = TransformerBlock(
            in_channels,
            nhead,
            dim_feedforward=in_channels * 2,
            attention_dropout_rate=0.0,
            residual_dropout_rate=0.1,
            with_self_attn=False,
            with_cross_attn=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.Hardswish(),
            nn.Linear(hidden, max_length + 1),
        )

    def forward(self, feats):
        x = self.pos_encoding(feats)
        for layer in self.encoder_layers:
            x = layer(x)
        query = paddle.tile(self.query, [feats.shape[0], 1, 1])
        pooled = self.pool_block(query, memory=x).squeeze(1)
        return self.fc(pooled)


class LengthBranchAttnReg(nn.Layer):
    """Regression variant of LengthBranchAttn: same self-attention encoder
    and query-token pooling, but the final layer emits a single scalar (the
    predicted count as a real number) instead of a max_length+1-way softmax
    classifier. To stay drop-in compatible with the existing eval path and
    CTCLengthGatedDecode (which expects a (B, max_length+1) distribution to
    argmax/max over), the scalar is expanded into a soft distribution peaked
    at its value: logits[b, c] = -(scalar[b] - c)^2 * temperature. The
    regression loss (LengthLossMSE) recovers the scalar from this same
    tensor via a soft-argmax (expected class under softmax), so no change to
    the forward() signature or downstream postprocess is needed.

    Args:
        in_channels: width of the incoming (B, T, C) feature sequence.
        max_length: largest count the branch can emit.
        hidden: bottleneck width before the regression head.
        nhead, num_layers: as in LengthBranchAttn.
        temperature: sharpness of the pseudo-distribution around the scalar.
    """

    def __init__(
        self, in_channels, max_length=25, hidden=128, nhead=4, num_layers=2, temperature=5.0
    ):
        super(LengthBranchAttnReg, self).__init__()
        from ppocr.modeling.heads.rec_nrtr_head import PositionalEncoding, TransformerBlock
        from ppocr.modeling.backbones.rec_svtrnet import zeros_

        self.max_length = max_length
        self.temperature = temperature
        self.pos_encoding = PositionalEncoding(dropout=0.1, dim=in_channels)
        self.encoder_layers = nn.LayerList(
            [
                TransformerBlock(
                    in_channels,
                    nhead,
                    dim_feedforward=in_channels * 2,
                    attention_dropout_rate=0.0,
                    residual_dropout_rate=0.1,
                    with_self_attn=True,
                    with_cross_attn=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.query = self.create_parameter(
            shape=[1, 1, in_channels], default_initializer=zeros_
        )
        self.pool_block = TransformerBlock(
            in_channels,
            nhead,
            dim_feedforward=in_channels * 2,
            attention_dropout_rate=0.0,
            residual_dropout_rate=0.1,
            with_self_attn=False,
            with_cross_attn=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.Hardswish(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer(
            "classes", paddle.arange(0, max_length + 1, dtype="float32")
        )

    def forward(self, feats):
        x = self.pos_encoding(feats)
        for layer in self.encoder_layers:
            x = layer(x)
        query = paddle.tile(self.query, [feats.shape[0], 1, 1])
        pooled = self.pool_block(query, memory=x).squeeze(1)
        scalar = self.fc(pooled)  # (B, 1), unconstrained real-valued count
        logits = -self.temperature * (scalar - self.classes.unsqueeze(0)) ** 2
        return logits
