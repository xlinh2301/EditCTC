"""Independent visual verifier for complete CTC candidate sequences.

The verifier deliberately lives outside ``MultiHeadEditRefineNRTR``.  It sees
the image feature map and a proposed complete string, then returns one scalar
compatibility score.  Keeping this path independent makes it possible to test
whether a proposed edit is visually better than the unchanged CTC seed.
"""
from __future__ import absolute_import, division, print_function

import paddle
from paddle import nn

from .rec_edit_refine_nrtr_head import SharedHighResVisualMemory


class SequenceVisualVerifier(nn.Layer):
    """Score ``(visual feature map, candidate token sequence)`` pairs."""

    def __init__(
        self,
        in_channels=384,
        vocab_size=43,
        hidden=192,
        max_length=25,
        layers=2,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("verifier hidden size must be divisible by heads")
        self.max_length = int(max_length)
        self.visual = SharedHighResVisualMemory(
            in_channels=in_channels,
            dim=hidden,
            max_height=4,
            max_width=96,
            layers=1,
            nhead=heads,
            dropout=dropout,
        )
        self.token_embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos_embed = self.create_parameter(
            [1, self.max_length, hidden],
            default_initializer=nn.initializer.Normal(std=hidden**-0.5),
        )
        from .rec_nrtr_head import TransformerBlock

        self.blocks = nn.LayerList(
            [
                TransformerBlock(
                    d_model=hidden,
                    nhead=heads,
                    dim_feedforward=hidden * 4,
                    attention_dropout_rate=dropout,
                    residual_dropout_rate=dropout,
                    with_self_attn=True,
                    with_cross_attn=True,
                )
                for _ in range(int(layers))
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.score = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feature_map, candidate_ids, candidate_lens):
        """Return one score per candidate sequence.

        Args:
            feature_map: ``[B,C,H,W]`` backbone feature map.
            candidate_ids: ``[B,L]`` CTC-vocabulary ids, padded with zero.
            candidate_lens: ``[B]`` unpadded candidate lengths.
        """
        if len(candidate_ids.shape) != 2:
            raise ValueError("candidate_ids must be [B,L]")
        if candidate_ids.shape[1] > self.max_length:
            raise ValueError("candidate sequence exceeds verifier max_length")
        visual = self.visual(feature_map)
        tokens = self.token_embed(candidate_ids)
        tokens = tokens + self.pos_embed[:, : candidate_ids.shape[1], :]
        valid = (paddle.arange(candidate_ids.shape[1])[None, :] < candidate_lens[:, None]).astype("float32")
        for block in self.blocks:
            tokens = block(tokens, visual)
        tokens = self.norm(tokens)
        denom = paddle.clip(valid.sum(axis=1, keepdim=True), min=1.0)
        cand = (tokens * valid.unsqueeze(-1)).sum(axis=1) / denom
        image = visual.mean(axis=1)
        length = (candidate_lens.astype("float32") / float(self.max_length)).unsqueeze(-1)
        features = paddle.concat([cand, image, cand * image, length], axis=1)
        return self.score(features).squeeze(-1)

