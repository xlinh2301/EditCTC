"""NRTR-initialized EditCTC head.

This variant removes the separate NRTR training branch.  The pretrained NRTR
Transformer decoder is reused as the edit decoder: it consumes the CTC seed
tokens while cross-attending to a memory made from the CTC feature sequence
and a small high-resolution encoder of the original image.  Its hidden states
feed the edit-op and replacement-token heads.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import nn

from .rec_multi_head import MultiHead
from .rec_nrtr_head import TransformerBlock
from .rec_edit_refine_head import (
    EditRefineDecoder,
    apply_edit_ops,
    build_refined_ctc_probs,
)


def ctc_seed_and_margin(ctc_logits, max_seed_len):
    """Greedy CTC collapse plus margin at each emitted token's frame."""
    ids = paddle.argmax(ctc_logits, axis=2).numpy()
    probs = paddle.nn.functional.softmax(ctc_logits, axis=2).numpy()
    bsz = ids.shape[0]
    seeds = np.zeros((bsz, max_seed_len), dtype="int64")
    lens = np.zeros((bsz,), dtype="int64")
    margins = np.zeros((bsz, max_seed_len), dtype="float32")
    alternatives = np.zeros((bsz, max_seed_len), dtype="int64")
    for b in range(bsz):
        prev, n = -1, 0
        for t, tid in enumerate(ids[b].tolist()):
            if tid != 0 and tid != prev and n < max_seed_len:
                seeds[b, n] = tid
                order = np.argsort(probs[b, t])
                margins[b, n] = float(probs[b, t, order[-1]] - probs[b, t, order[-2]])
                alternatives[b, n] = int(order[-2])
                n += 1
            prev = tid
        lens[b] = n
    return seeds, lens, margins, alternatives


class SharedHighResVisualMemory(nn.Layer):
    """Encode the backbone's pre-pooling 2D feature map for the NRTR decoder.

    PPLCNetV4 stores its last recognition feature map as ``recon_feat`` before
    collapsing height for CTC.  E44-A reuses that feature instead of encoding
    raw pixels in a separate branch.  Row/column embeddings preserve the 2D
    layout, while a small self-attention stack supplies local/global context
    before the map is flattened into decoder memory.
    """

    def __init__(
        self,
        in_channels,
        dim,
        max_height=8,
        max_width=256,
        layers=1,
        nhead=8,
        dropout=0.1,
    ):
        super().__init__()
        if dim % nhead != 0:
            raise ValueError("visual memory dim must be divisible by nhead")
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        self.proj = nn.Conv2D(in_channels, dim, kernel_size=1)
        # Depthwise local mixing preserves character strokes before global
        # attention and is cheap at the 3x80 PPLCNetV4 feature resolution.
        self.local_mix = nn.Conv2D(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )
        self.visual_row_embed = self.create_parameter(
            shape=[1, self.max_height, 1, dim],
            default_initializer=nn.initializer.Normal(std=dim**-0.5),
        )
        self.visual_col_embed = self.create_parameter(
            shape=[1, 1, self.max_width, dim],
            default_initializer=nn.initializer.Normal(std=dim**-0.5),
        )
        self.visual_type_embed = self.create_parameter(
            shape=[1, 1, dim],
            default_initializer=nn.initializer.Normal(std=dim**-0.5),
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.LayerList(
            [
                TransformerBlock(
                    d_model=dim,
                    nhead=nhead,
                    dim_feedforward=dim * 4,
                    attention_dropout_rate=dropout,
                    residual_dropout_rate=dropout,
                    with_self_attn=True,
                    with_cross_attn=False,
                )
                for _ in range(int(layers))
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feature_map):
        if feature_map is None or len(feature_map.shape) != 4:
            raise ValueError("high-resolution visual feature must be [B,C,H,W]")
        _, _, height, width = feature_map.shape
        if height > self.max_height or width > self.max_width:
            raise ValueError(
                "visual position range exceeded: got HxW={}x{}, max={}x{}".format(
                    height, width, self.max_height, self.max_width
                )
            )
        x = self.proj(feature_map)
        x = x + self.local_mix(x)
        x = x.transpose([0, 2, 3, 1])
        x = x + self.visual_row_embed[:, :height, :, :]
        x = x + self.visual_col_embed[:, :, :width, :]
        x = x.reshape([0, height * width, x.shape[-1]])
        x = self.dropout(x + self.visual_type_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class MultiHeadEditRefineNRTR(MultiHead):
    """CTC + integrated NRTR-seed edit decoder, without an NRTR loss branch."""

    def __init__(self, in_channels, out_channels_list, **kwargs):
        # The head_list still contains CTCHead + NRTRHead so MultiHead builds
        # the pretrained NRTR module under the stable key `gtc_head.*`.
        super().__init__(in_channels, out_channels_list, **kwargs)
        assert self.use_length_head, "integrated NRTR edit head requires length head"
        self.use_original_image = True
        self.use_highres_visual = kwargs.get("use_highres_visual", False)
        self.backbone_ref = None
        self.vocab_size = out_channels_list["CTCLabelDecode"]
        self.max_seed_len = kwargs.get("length_max", 25)
        self.nrtr_dim = kwargs.get("nrtr_dim", 384)
        self.ctc_mem_proj = (
            nn.Linear(self.ctc_encoder.out_channels, self.nrtr_dim)
            if self.ctc_encoder.out_channels != self.nrtr_dim else None
        )
        # High-resolution image tokens: 48x320 -> 12x80 -> height-pooled 80
        # tokens.  This is intentionally small; the CTC memory remains the
        # main visual sequence and the raw-image path restores local detail.
        self.image_encoder = nn.Sequential(
            nn.Conv2D(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2D(64, self.ctc_encoder.out_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D((1, 80)),
        )
        self.image_mem_proj = nn.Linear(self.ctc_encoder.out_channels, self.nrtr_dim)
        self.ctc_type_embed = self.create_parameter(
            shape=[1, 1, self.nrtr_dim],
            default_initializer=nn.initializer.Normal(std=self.nrtr_dim**-0.5),
        )
        if self.use_highres_visual:
            self.visual_memory = SharedHighResVisualMemory(
                in_channels=kwargs.get("visual_in_channels", in_channels),
                dim=self.nrtr_dim,
                max_height=kwargs.get("visual_pos_max_height", 8),
                max_width=kwargs.get("visual_pos_max_width", 256),
                layers=kwargs.get("visual_encoder_layers", 1),
                nhead=kwargs.get("visual_encoder_heads", self.nrtr_dim // 48),
                dropout=kwargs.get("visual_encoder_dropout", 0.1),
            )
        self.edit_op_head = nn.Linear(self.nrtr_dim, 4)
        self.edit_tok_head = nn.Linear(self.nrtr_dim, self.vocab_size)
        self.edit_head_dropout = nn.Dropout(kwargs.get("edit_head_dropout", 0.2))
        self.edit_allowed_ops = kwargs.get("edit_allowed_ops")
        self.edit_mode = kwargs.get("edit_mode", "four_way")
        self.train_seed_corrupt_prob = kwargs.get("train_seed_corrupt_prob", 0.0)
        self.train_seed_corrupt_mode = kwargs.get("train_seed_corrupt_mode", "random")
        self.edit_gate_threshold = kwargs.get("edit_gate_threshold", 0.5)
        self.edit_delta_threshold = kwargs.get("edit_delta_threshold", 0.05)
        self.edit_candidate_top2_only = kwargs.get("edit_candidate_top2_only", False)
        if self.edit_mode == "factorized":
            # Residual editor: class 0 means KEEP and class 1 means the
            # replacement head should be used. DELETE/INSERT are deliberately
            # absent until substitution has a positive held-out result.
            self.edit_op_head = nn.Linear(self.nrtr_dim, 2)
            self.edit_tok_head = nn.Linear(self.nrtr_dim, self.vocab_size)
        # Inference-only branch tracing used by the test audit.  The normal
        # output remains the refined CTC tensor unless this flag is enabled.
        self.branch_debug = kwargs.get("branch_debug", False)

        # Transfer NRTR's pretrained character projection into the CTC-vocab
        # token head where the index spaces overlap (NRTR chars are CTC ids+3).
        with paddle.no_grad():
            src = self.gtc_head.tgt_word_prj.weight
            rows = [i + 3 for i in range(self.vocab_size)]
            rows = [i for i in rows if i < src.shape[0]]
            if rows:
                weight = self.edit_tok_head.weight.numpy()
                src_weight = src.numpy()
                weight[:, : len(rows)] = src_weight[:, rows]
                self.edit_tok_head.weight.set_value(weight)
                bias = self.edit_tok_head.bias.numpy()
                bias[: len(rows)] = 0.0
                self.edit_tok_head.bias.set_value(bias)

    def _memory(self, feature_map, original_image):
        ctc_memory = self.ctc_encoder(feature_map)
        ctc_part = (
            ctc_memory if self.ctc_mem_proj is None else self.ctc_mem_proj(ctc_memory)
        )
        ctc_part = ctc_part + self.ctc_type_embed
        parts = [ctc_part]
        if self.use_highres_visual:
            backbone = self.backbone_ref
            if callable(backbone):
                backbone = backbone()
            recon_feat = getattr(backbone, "recon_feat", None)
            if recon_feat is None:
                raise RuntimeError(
                    "use_highres_visual requires backbone.recon_feat; "
                    "run the PPLCNetV4 backbone before the head"
                )
            parts.append(self.visual_memory(recon_feat))
        elif original_image is not None:
            raw = self.image_encoder(original_image)
            raw = raw.squeeze(2).transpose([0, 2, 1])
            parts.append(self.image_mem_proj(raw))
        return ctc_memory, paddle.concat(parts, axis=1)

    def _seed_hidden(self, memory, seed_ids):
        # NRTR uses 2 as BOS and CTC character ids map to NRTR ids +3.
        nrtr_seed = paddle.where(seed_ids > 0, seed_ids + 3, paddle.zeros_like(seed_ids))
        bos = paddle.full([seed_ids.shape[0], 1], 2, dtype="int64")
        decoder_input = paddle.concat([bos, nrtr_seed], axis=1)
        tgt = self.gtc_head.embedding(decoder_input)
        tgt = self.gtc_head.positional_encoding(tgt)
        mask = self.gtc_head.generate_square_subsequent_mask(tgt.shape[1])
        for layer in self.gtc_head.decoder:
            tgt = layer(tgt, memory, self_mask=mask)
        return self.edit_head_dropout(tgt[:, 1:, :])

    def _edit_forward(self, ctc_out, memory, length_logits, seed_ids=None, seed_lens=None):
        if seed_ids is None:
            seeds_np, lens_np, _, _ = ctc_seed_and_margin(ctc_out, self.max_seed_len)
            seed_ids = paddle.to_tensor(seeds_np, dtype="int64")
            seed_lens = paddle.to_tensor(lens_np, dtype="int64")
        hidden = self._seed_hidden(memory, seed_ids)
        op_logits = self.edit_op_head(hidden)
        tok_logits = self.edit_tok_head(hidden)
        return {"op_logits": op_logits, "tok_logits": tok_logits,
                "seed_ids": seed_ids, "seed_lens": seed_lens}

    def _corrupt_seed_tokens(self, seeds_np, lens_np, margins_np, alternatives_np=None):
        """Create guaranteed-ish replacement positives for factorized training.

        This is training-only denoising: CTC/backbone outputs remain untouched,
        while the edit decoder sees a one-token wrong seed on a subset of
        samples.  The loss aligns this altered seed against the ground truth.
        """
        if (
            not self.training
            or self.edit_mode not in ("factorized", "token_refine")
            or self.train_seed_corrupt_prob <= 0
        ):
            return seeds_np, lens_np, margins_np
        seeds_np = seeds_np.copy()
        margins_np = margins_np.copy()
        for b, n_raw in enumerate(lens_np):
            n = int(n_raw)
            if n <= 0 or np.random.random() >= self.train_seed_corrupt_prob:
                continue
            pos = int(np.random.randint(n))
            old = int(seeds_np[b, pos])
            if self.vocab_size <= 2:
                continue
            if self.train_seed_corrupt_mode == "ctc_alt" and alternatives_np is not None:
                new = int(alternatives_np[b, pos])
                if new <= 0 or new >= self.vocab_size:
                    new = int(np.random.randint(1, self.vocab_size))
            else:
                new = int(np.random.randint(1, self.vocab_size))
            if new == old:
                new = 1 + (old % (self.vocab_size - 1))
            seeds_np[b, pos] = new
            # The altered token is intentionally uncertain for diagnostics.
            margins_np[b, pos] = 0.0
        return seeds_np, lens_np, margins_np

    def forward(self, x, targets=None, original_image=None):
        ctc_memory, memory = self._memory(x, original_image)
        ctc_out = self.ctc_head(ctc_memory, targets)
        length_logits = self.length_head(ctc_memory) if self.use_length_head else None
        seeds_np, lens_np, margins, alternatives = ctc_seed_and_margin(
            ctc_out, self.max_seed_len
        )
        seeds_np, lens_np, margins = self._corrupt_seed_tokens(
            seeds_np, lens_np, margins, alternatives
        )
        seeds = paddle.to_tensor(seeds_np, dtype="int64")
        lens = paddle.to_tensor(lens_np, dtype="int64")
        edit_out = self._edit_forward(ctc_out, memory, length_logits, seeds, lens)
        edit_out["seed_margin"] = paddle.to_tensor(margins, dtype="float32")
        if self.training:
            out = {"ctc": ctc_out, "ctc_neck": ctc_memory, "edit": edit_out}
            if length_logits is not None:
                out["length"] = length_logits
            return out

        op_logits = edit_out["op_logits"]
        if self.edit_allowed_ops is not None:
            blocked = [i for i in range(4) if i not in self.edit_allowed_ops]
            blocked = [i for i in blocked if i < op_logits.shape[-1]]
            if blocked:
                op_logits[:, :, blocked] = -1e9
        if self.edit_mode == "factorized":
            gate_prob = paddle.nn.functional.softmax(op_logits, axis=2)[:, :, 1]
            op_ids = (gate_prob >= self.edit_gate_threshold).astype("int64").numpy()
        elif self.edit_mode == "token_refine":
            # Direct denoising refinement: the token head is trained on every
            # aligned KEEP/REPLACE slot, so no rare EDIT/KEEP classifier can
            # collapse to the majority KEEP class.  A replacement is accepted
            # only when the predicted token is different, sufficiently
            # confident, and has a margin over the seed token probability.
            tok_probs = paddle.nn.functional.softmax(edit_out["tok_logits"], axis=2)
            tok_ids_t = paddle.argmax(tok_probs, axis=2)
            seed_clip = paddle.clip(seeds, 0, tok_probs.shape[2] - 1)
            seed_prob = paddle.take_along_axis(
                tok_probs, seed_clip.unsqueeze(-1), axis=2
            ).squeeze(-1)
            best_prob = paddle.max(tok_probs, axis=2)
            edit_mask = paddle.logical_and(
                tok_ids_t != seeds,
                paddle.logical_and(
                    best_prob >= self.edit_gate_threshold,
                    best_prob - seed_prob >= self.edit_delta_threshold,
                ),
            )
            if self.edit_candidate_top2_only:
                alt_t = paddle.to_tensor(alternatives, dtype="int64")
                edit_mask = paddle.logical_and(edit_mask, tok_ids_t == alt_t)
            op_ids = edit_mask.astype("int64").numpy()
        else:
            op_ids = paddle.argmax(op_logits, axis=2).numpy()
        tok_ids = paddle.argmax(edit_out["tok_logits"], axis=2).numpy()
        refined = []
        for b, n in enumerate(lens_np):
            refined.append(apply_edit_ops(seeds_np[b, : int(n)].tolist(),
                                          op_ids[b, : int(n)].tolist(),
                                          tok_ids[b, : int(n)].tolist()))
        refined_probs = build_refined_ctc_probs(
            refined, self.vocab_size, EditRefineDecoder.BLANK_ID
        )
        if not self.branch_debug:
            return refined_probs

        # Keep this payload detached and inference-only.  infer_rec.py uses
        # the same schema as the uncertainty head to compute CTC/seed/final
        # exactness plus helped/hurt and operation counts.
        branch_debug = {
            "ctc_probs": ctc_out.numpy(),
            "length_logits": (
                length_logits.numpy() if length_logits is not None else None
            ),
            "nrtr_ids": None,
            "nrtr_probs": None,
            "seed_ids": seeds_np,
            "seed_lens": lens_np,
            "edit_op_logits": edit_out["op_logits"].numpy(),
            "edit_tok_logits": edit_out["tok_logits"].numpy(),
            "edit_op_ids": op_ids,
            "edit_tok_ids": tok_ids,
            "refined_ids": refined,
        }
        return {"ctc": refined_probs, "branch_debug": branch_debug}
