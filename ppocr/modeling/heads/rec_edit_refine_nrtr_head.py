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
from paddle.nn import functional as F

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


def ctc_seed_and_conf(ctc_logits, max_seed_len, conf_dim=2):
    """Greedy CTC collapse plus peak margin, alternatives, continuous confidence, and temporal timesteps.

    Args:
        ctc_logits: [B, T, V] tensor of CTC output logits or eval softmax probabilities
        max_seed_len: int, maximum number of characters in seed
        conf_dim: int, 2 for [p_top1, margin], 4 for [p_top1, p_top2, margin, entropy]
    Returns:
        seeds: [B, max_seed_len] int64 array
        lens: [B] int64 array
        margins: [B, max_seed_len] float32 array
        alternatives: [B, max_seed_len] int64 array
        confs: [B, max_seed_len, conf_dim] float32 array
        timesteps: [B, max_seed_len] int64 array
    """
    if hasattr(ctc_logits, "numpy"):
        c_np = ctc_logits.numpy()
    else:
        c_np = np.asarray(ctc_logits)
    # CTCHead applies softmax when training=False. Avoid catastrophic double-softmax!
    if np.all(c_np >= 0.0) and np.allclose(c_np.sum(axis=-1), 1.0, atol=1e-2):
        probs = c_np
    else:
        probs = paddle.nn.functional.softmax(ctc_logits, axis=2).numpy()
    ids = np.argmax(probs, axis=-1)
    bsz = ids.shape[0]
    seeds = np.zeros((bsz, max_seed_len), dtype="int64")
    lens = np.zeros((bsz,), dtype="int64")
    margins = np.zeros((bsz, max_seed_len), dtype="float32")
    alternatives = np.zeros((bsz, max_seed_len), dtype="int64")
    confs = np.zeros((bsz, max_seed_len, conf_dim), dtype="float32")
    timesteps = np.zeros((bsz, max_seed_len), dtype="int64")
    
    for b in range(bsz):
        seq_ids = ids[b].tolist()
        T = len(seq_ids)
        prev = -1
        seg_start = -1
        cur_tid = 0
        n = 0
        for t, tid in enumerate(seq_ids):
            if tid != prev:
                if cur_tid != 0 and n < max_seed_len:
                    seg_probs = probs[b, seg_start:t, cur_tid]
                    peak_offset = int(np.argmax(seg_probs))
                    peak_t = seg_start + peak_offset
                    seeds[b, n] = cur_tid
                    timesteps[b, n] = peak_t
                    order = np.argsort(probs[b, peak_t])
                    p_top1 = float(probs[b, peak_t, order[-1]])
                    p_top2 = float(probs[b, peak_t, order[-2]])
                    m = float(p_top1 - p_top2)
                    margins[b, n] = m
                    alternatives[b, n] = int(order[-2])
                    if conf_dim == 2:
                        confs[b, n, 0] = p_top1
                        confs[b, n, 1] = m
                    elif conf_dim == 4:
                        p_vec = probs[b, peak_t]
                        entropy = -float(np.sum(p_vec * np.log(np.clip(p_vec, 1e-12, 1.0))))
                        confs[b, n, 0] = p_top1
                        confs[b, n, 1] = p_top2
                        confs[b, n, 2] = m
                        confs[b, n, 3] = entropy
                    n += 1
                cur_tid = tid
                seg_start = t
            prev = tid
        if cur_tid != 0 and n < max_seed_len:
            seg_probs = probs[b, seg_start:T, cur_tid]
            peak_offset = int(np.argmax(seg_probs))
            peak_t = seg_start + peak_offset
            seeds[b, n] = cur_tid
            timesteps[b, n] = peak_t
            order = np.argsort(probs[b, peak_t])
            p_top1 = float(probs[b, peak_t, order[-1]])
            p_top2 = float(probs[b, peak_t, order[-2]])
            m = float(p_top1 - p_top2)
            margins[b, n] = m
            alternatives[b, n] = int(order[-2])
            if conf_dim == 2:
                confs[b, n, 0] = p_top1
                confs[b, n, 1] = m
            elif conf_dim == 4:
                p_vec = probs[b, peak_t]
                entropy = -float(np.sum(p_vec * np.log(np.clip(p_vec, 1e-12, 1.0))))
                confs[b, n, 0] = p_top1
                confs[b, n, 1] = p_top2
                confs[b, n, 2] = m
                confs[b, n, 3] = entropy
            n += 1
        lens[b] = n
    return seeds, lens, margins, alternatives, confs, timesteps



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
        act_type="ReLU",
        use_coord_conv=False,
        use_ms_stroke_mixer=False,
        use_local_refine_block=False,
    ):
        super().__init__()
        if dim % nhead != 0:
            raise ValueError("visual memory dim must be divisible by nhead")
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        self.use_coord_conv = bool(use_coord_conv)
        if self.use_coord_conv:
            self.proj = nn.Conv2D(in_channels + 2, dim, kernel_size=1)
        else:
            self.proj = nn.Conv2D(in_channels, dim, kernel_size=1)
        # Depthwise local mixing preserves character strokes before global
        # attention and is cheap at the 3x80 PPLCNetV4 feature resolution.
        self.local_mix = nn.Conv2D(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )
        self.use_ms_stroke_mixer = bool(use_ms_stroke_mixer)
        if self.use_ms_stroke_mixer:
            self.stroke_mix_h = nn.Conv2D(
                dim, dim, kernel_size=[1, 7], padding=[0, 3], groups=dim
            )
        self.use_local_refine_block = bool(use_local_refine_block)
        if self.use_local_refine_block:
            self.local_refine_dw = nn.Conv2D(dim, dim, kernel_size=3, padding=1, groups=dim)
            self.local_refine_pw = nn.Conv2D(dim, dim, kernel_size=1)
            self.local_refine_norm = nn.BatchNorm2D(dim)
            self.local_refine_act = nn.GELU()
            self.local_refine_scale = self.create_parameter(
                shape=[1],
                default_initializer=nn.initializer.Constant(0.0),
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
                    act_type=act_type,
                )
                for _ in range(int(layers))
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feature_map):
        if feature_map is None or len(feature_map.shape) != 4:
            raise ValueError("high-resolution visual feature must be [B,C,H,W]")
        b, _, height, width = feature_map.shape
        if self.use_coord_conv:
            y_coords = paddle.linspace(-1.0, 1.0, height, dtype=feature_map.dtype).reshape([1, 1, height, 1]).tile([b, 1, 1, width])
            x_coords = paddle.linspace(-1.0, 1.0, width, dtype=feature_map.dtype).reshape([1, 1, 1, width]).tile([b, 1, height, 1])
            coords = paddle.concat([feature_map, x_coords, y_coords], axis=1)
            x = self.proj(coords)
        else:
            x = self.proj(feature_map)
        if self.use_ms_stroke_mixer:
            x = x + self.local_mix(x) + self.stroke_mix_h(x)
        else:
            x = x + self.local_mix(x)
        if self.use_local_refine_block:
            res = x
            feat = self.local_refine_dw(x)
            feat = self.local_refine_act(self.local_refine_norm(feat))
            feat = self.local_refine_pw(feat)
            x = res + paddle.tanh(self.local_refine_scale) * feat
        x = x.transpose([0, 2, 3, 1])
        # Training normally uses HxW <= max_height x max_width.  Dynamic
        # inference resizing can produce a slightly wider feature map; use
        # bilinear interpolation of the learned 2D table instead of failing
        # or silently dropping the extra columns.
        base_h = min(height, self.max_height)
        base_w = min(width, self.max_width)
        pos = self.visual_row_embed[:, :base_h, :, :] + self.visual_col_embed[
            :, :, :base_w, :
        ]
        if base_h != height or base_w != width:
            pos = F.interpolate(
                pos.transpose([0, 3, 1, 2]),
                size=[height, width],
                mode="bilinear",
                align_corners=False,
            ).transpose([0, 2, 3, 1])
        x = x + pos
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
                act_type=kwargs.get("act_type", "ReLU"),
                use_coord_conv=kwargs.get("use_coord_conv", False),
                use_ms_stroke_mixer=kwargs.get("use_ms_stroke_mixer", False),
                use_local_refine_block=kwargs.get("use_local_refine_block", False),
            )
        self.edit_op_head = nn.Linear(self.nrtr_dim, 4)
        self.edit_tok_head = nn.Linear(self.nrtr_dim, self.vocab_size)

        # ARCH-1 & ARCH-2: Explicit Change Head & Confidence Conditioning
        self.use_explicit_change_head = bool(kwargs.get("use_explicit_change_head", False))
        self.use_ctc_conf_embed = bool(kwargs.get("use_ctc_conf_embed", False))
        self.ctc_conf_dim = int(kwargs.get("ctc_conf_dim", 2))
        self.use_ctc_conf_in_change_head = bool(kwargs.get("use_ctc_conf_in_change_head", True))
        self.use_conf_in_decoder_query = bool(kwargs.get("use_conf_in_decoder_query", False))
        if self.use_explicit_change_head:
            change_in_dim = self.nrtr_dim
            if self.use_ctc_conf_embed and self.use_ctc_conf_in_change_head:
                change_in_dim += self.ctc_conf_dim
            self.change_head = nn.Linear(change_in_dim, 1)

        if self.use_ctc_conf_embed and self.use_conf_in_decoder_query:
            self.conf_proj = nn.Sequential(
                nn.Linear(self.ctc_conf_dim, 64),
                nn.GELU(),
                nn.Linear(64, self.nrtr_dim),
                nn.LayerNorm(self.nrtr_dim),
            )
            self.conf_scale = self.create_parameter(
                shape=[1],
                default_initializer=nn.initializer.Constant(0.0),
            )

        # ARCH-3: CTC Temporal Alignment Embedding
        self.use_ctc_align_embed = bool(kwargs.get("use_ctc_align_embed", False))
        self.max_ctc_timesteps = int(kwargs.get("max_ctc_timesteps", 40))
        if self.use_ctc_align_embed:
            self.align_embedding = nn.Embedding(self.max_ctc_timesteps + 1, self.nrtr_dim)
            self.align_scale = self.create_parameter(
                shape=[1],
                default_initializer=nn.initializer.Constant(0.0),
            )

        # ARCH-4: Alignment-Guided High-Res Cross-Attention
        self.use_align_guided_cross_attn = bool(kwargs.get("use_align_guided_cross_attn", False))
        self.align_spatial_weight = float(kwargs.get("align_spatial_weight", 1.0))
        self.align_spatial_sigma = float(kwargs.get("align_spatial_sigma", 0.15))

        # ARCH-7: Gated Memory Fusion (Dual Cross-Attention Routing)
        self.use_gated_memory_fusion = bool(kwargs.get("use_gated_memory_fusion", False))
        if self.use_gated_memory_fusion:
            self.memory_gate_proj = nn.Linear(self.nrtr_dim * 2, self.nrtr_dim)
            self.memory_gate_act = nn.Sigmoid()

        self.edit_head_dropout = nn.Dropout(kwargs.get("edit_head_dropout", 0.2))
        self.edit_allowed_ops = kwargs.get("edit_allowed_ops")
        self.edit_mode = kwargs.get("edit_mode", "four_way")
        self.train_seed_corrupt_prob = kwargs.get("train_seed_corrupt_prob", 0.0)
        self.train_seed_corrupt_mode = kwargs.get("train_seed_corrupt_mode", "random")
        self.edit_gate_threshold = kwargs.get("edit_gate_threshold", 0.5)
        self.edit_delta_threshold = kwargs.get("edit_delta_threshold", 0.05)
        self.edit_candidate_top2_only = kwargs.get("edit_candidate_top2_only", False)
        # Training-only: zero the seed-token embedding at a random subset of
        # decoder input positions.  The decoder output at position i can
        # otherwise read seed[i] straight off its own input through the causal
        # self-attention mask, which makes "copy the seed" a zero-effort
        # solution and leaves the visual/CTC memory unused.  0.0 disables it.
        self.seed_token_dropout = kwargs.get("seed_token_dropout", 0.0)
        # F3: decode with all four levenshtein ops (KEEP/REPLACE/DELETE/
        # INSERT_AFTER) instead of the KEEP/REPLACE-only gate.  The op head is
        # 4-way already; only the eval-side decode collapsed it.  Off by
        # default so existing checkpoints keep their exact behaviour.
        self.edit_ops_enabled = kwargs.get("edit_ops_enabled", False)
        self.edit_op_threshold = kwargs.get("edit_op_threshold", 0.5)
        self.edit_op_delta = kwargs.get("edit_op_delta", 0.05)
        # F4: one extra decoder query after the last seed token.  It is trained
        # to append a trailing character the frozen CTC seed dropped, which is
        # the dominant error direction on the indomain 6-digit bucket.
        self.use_append_head = kwargs.get("use_append_head", False)
        self.append_gate_threshold = kwargs.get("append_gate_threshold", 0.5)
        self.append_delta_threshold = kwargs.get("append_delta_threshold", 0.05)
        if self.use_append_head:
            self.append_query = self.create_parameter(
                shape=[1, 1, self.nrtr_dim],
                default_initializer=nn.initializer.Normal(std=self.nrtr_dim ** -0.5),
            )
            self.append_head = nn.Linear(self.nrtr_dim, 2)
        if self.edit_mode == "factorized":
            # Residual editor: class 0 means KEEP and class 1 means the
            # replacement head should be used. DELETE/INSERT are deliberately
            # absent until substitution has a positive held-out result.
            self.edit_op_head = nn.Linear(self.nrtr_dim, 2)
            self.edit_tok_head = nn.Linear(self.nrtr_dim, self.vocab_size)
        # Inference-only branch tracing used by the test audit.  The normal
        # output remains the refined CTC tensor unless this flag is enabled.
        self.branch_debug = kwargs.get("branch_debug", False)

        # Solution 2: Tail-Guided Spatial Cross-Attention parameters
        self.tail_spatial_attn = kwargs.get("tail_spatial_attn", False)
        self.tail_spatial_bias_weight = float(kwargs.get("tail_spatial_bias_weight", 2.0))
        self.tail_spatial_sigma = float(kwargs.get("tail_spatial_sigma", 0.20))
        self.tail_spatial_mode = kwargs.get("tail_spatial_mode", "tail_only")  # "tail_only" or "progressive"
        self.tail_prog_weight = float(kwargs.get("tail_prog_weight", 1.0))
        self.tail_prog_sigma = float(kwargs.get("tail_prog_sigma", 0.35))

        # Solution 3: Length-Embedding Injection
        self.use_length_embedding = kwargs.get("use_length_embedding", False)
        if self.use_length_embedding:
            self.length_embedding = nn.Embedding(self.max_seed_len + 1, self.nrtr_dim)

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

    def _seed_hidden(
        self, memory, seed_ids, seed_lens=None, length_logits=None, with_append=False, confs=None, timesteps=None
    ):
        # NRTR uses 2 as BOS and CTC character ids map to NRTR ids +3.
        nrtr_seed = paddle.where(seed_ids > 0, seed_ids + 3, paddle.zeros_like(seed_ids))
        tail_mask = None
        if with_append:
            nrtr_seed, tail_mask = self._insert_tail_slot(nrtr_seed, seed_lens)
        bos = paddle.full([seed_ids.shape[0], 1], 2, dtype="int64")
        decoder_input = paddle.concat([bos, nrtr_seed], axis=1)
        tgt = self.gtc_head.embedding(decoder_input)
        tgt = self.gtc_head.positional_encoding(tgt)
        if self.use_ctc_conf_embed and self.use_conf_in_decoder_query:
            if confs is None:
                confs = paddle.zeros([seed_ids.shape[0], seed_ids.shape[1], self.ctc_conf_dim], dtype="float32")
            if with_append:
                append_slot = paddle.zeros([confs.shape[0], 1, self.ctc_conf_dim], dtype=confs.dtype)
                confs = paddle.concat([confs, append_slot], axis=1)
            # BOS slot gets neutral confidence [1.0, 1.0] (or [1.0, 0.0, 1.0, 0.0])
            bos_conf = paddle.ones([seed_ids.shape[0], 1, self.ctc_conf_dim], dtype="float32")
            conf_seq = paddle.concat([bos_conf, confs], axis=1)
            tgt = tgt + paddle.tanh(self.conf_scale) * self.conf_proj(conf_seq)
        if self.use_ctc_align_embed and timesteps is not None:
            bos_t = paddle.zeros([seed_ids.shape[0], 1], dtype="int64")
            time_seq = paddle.concat([bos_t, timesteps], axis=1)
            if with_append:
                append_t = paddle.clip(timesteps[:, -1:] + 2, 0, self.max_ctc_timesteps)
                time_seq = paddle.concat([time_seq, append_t], axis=1)
            time_seq = paddle.clip(time_seq, 0, self.max_ctc_timesteps)
            tgt = tgt + paddle.tanh(self.align_scale) * self.align_embedding(time_seq)
        if self.use_length_embedding and length_logits is not None:
            pred_len = paddle.clip(
                paddle.argmax(length_logits, axis=1), 1, self.max_seed_len
            )
            l_emb = self.length_embedding(pred_len).unsqueeze(1)
            tgt = tgt + l_emb
        if self.training and self.seed_token_dropout > 0.0:
            keep = paddle.cast(
                paddle.rand(tgt.shape[:2]) >= self.seed_token_dropout, tgt.dtype
            ).unsqueeze(-1)
            tgt = tgt * keep
        if tail_mask is not None:
            # The append slot is a free query, not a copied seed token: its
            # embedding is replaced by the learned query vector.  Adding after
            # the positional encoding keeps that query position-agnostic.
            # tail_mask covers the seed columns only, so shift it past the BOS.
            tail_mask = paddle.concat(
                [paddle.zeros_like(tail_mask[:, :1]), tail_mask], axis=1
            )
            tgt = tgt + paddle.cast(self.append_query, tgt.dtype) * tail_mask.unsqueeze(-1)
        mask = self.gtc_head.generate_square_subsequent_mask(tgt.shape[1])
        cross_mask = None
        if self.tail_spatial_attn or self.use_align_guided_cross_attn:
            cross_mask = self._build_spatial_cross_bias(
                tgt_len=tgt.shape[1],
                mem_len=memory.shape[1],
                seed_lens=seed_lens,
                tail_mask=tail_mask,
                bsz=tgt.shape[0],
                dtype=tgt.dtype,
                timesteps=timesteps,
            )
        for layer in self.gtc_head.decoder:
            if self.use_gated_memory_fusion and memory.shape[1] > 40:
                ctc_len = 40
                ctc_mem = memory[:, :ctc_len, :]
                vis_mem = memory[:, ctc_len:, :]
                # 1. Self Attention
                tgt1 = layer.self_attn(tgt, attn_mask=mask)
                tgt = layer.norm1(tgt + layer.dropout1(tgt1))
                # 2. Dual Cross Attention
                h_ctc = layer.cross_attn(tgt, key=ctc_mem)
                vis_mask = cross_mask[:, :, :, ctc_len:] if cross_mask is not None else None
                h_vis = layer.cross_attn(tgt, key=vis_mem, attn_mask=vis_mask)
                # 3. Gated Fusion
                gate = self.memory_gate_act(self.memory_gate_proj(paddle.concat([h_ctc, h_vis], axis=-1)))
                fused_h = gate * h_vis + (1.0 - gate) * h_ctc
                tgt = layer.norm2(tgt + layer.dropout2(fused_h))
                # 4. Feed Forward
                tgt = layer.norm3(tgt + layer.dropout3(layer.mlp(tgt)))
            else:
                tgt = layer(tgt, memory, self_mask=mask, cross_mask=cross_mask)
        return self.edit_head_dropout(tgt[:, 1:, :])

    def _build_spatial_cross_bias(
        self, tgt_len, mem_len, seed_lens=None, tail_mask=None, bsz=1, dtype="float32", timesteps=None
    ):
        """Construct coordinate-aware spatial bias for cross-attention.

        Args:
            tgt_len: sequence length of queries (including BOS), int
            mem_len: sequence length of memory keys (e.g. 280), int
            seed_lens: [B] tensor/array of seed lengths
            tail_mask: [B, tgt_len] tensor with 1.0 at the append query slot (after BOS offset)
            bsz: batch size
            dtype: output tensor dtype
            timesteps: [B, max_seed_len] int64 tensor/array of CTC emitted timesteps
        Returns:
            [B, 1, tgt_len, mem_len] float32 tensor of additive attention biases
        """
        # 1. Compute horizontal normalized coordinate u in [0, 1] for each memory token
        if self.use_highres_visual:
            ctc_w = 40
            vis_len = mem_len - ctc_w
            if vis_len > 0:
                vis_w = 80 if vis_len % 80 == 0 else (vis_len // 3 if vis_len % 3 == 0 else vis_len)
                vis_h = max(1, vis_len // vis_w)
                u_ctc = np.linspace(0.0, 1.0, ctc_w, dtype=np.float32)
                u_col = np.linspace(0.0, 1.0, vis_w, dtype=np.float32)
                u_vis = np.tile(u_col, vis_h)
                u = np.concatenate([u_ctc, u_vis])
            else:
                u = np.linspace(0.0, 1.0, mem_len, dtype=np.float32)
        elif self.use_original_image:
            ctc_w = 40
            raw_w = mem_len - ctc_w
            u_ctc = np.linspace(0.0, 1.0, ctc_w, dtype=np.float32)
            u_raw = np.linspace(0.0, 1.0, max(1, raw_w), dtype=np.float32)
            u = np.concatenate([u_ctc, u_raw])
        else:
            u = np.linspace(0.0, 1.0, mem_len, dtype=np.float32)

        # 2. Build bias tensor of shape [B, 1, tgt_len, mem_len]
        bias_np = np.zeros((bsz, 1, tgt_len, mem_len), dtype=np.float32)
        lens_np = (
            seed_lens.numpy()
            if seed_lens is not None
            else np.full([bsz], tgt_len - 1, dtype="int64")
        )

        tail_weight = self.tail_spatial_bias_weight
        tail_sigma_sq2 = 2.0 * (self.tail_spatial_sigma ** 2)
        prog_weight = self.tail_prog_weight
        prog_sigma_sq2 = 2.0 * (self.tail_prog_sigma ** 2)

        tail_bias_vec = tail_weight * np.exp(-((u - 1.0) ** 2) / tail_sigma_sq2)

        tail_mask_np = tail_mask.numpy() if tail_mask is not None else None

        for b in range(bsz):
            n = int(lens_np[b])
            # For seed positions j = 1..n
            if self.tail_spatial_mode == "progressive":
                for j in range(1, min(n + 1, tgt_len)):
                    c_j = (j - 0.5) / max(1.0, float(n))
                    bias_np[b, 0, j, :] = prog_weight * np.exp(-((u - c_j) ** 2) / prog_sigma_sq2)
            elif self.tail_spatial_mode == "tail_only":
                # Mild boost on the last seed token
                if 1 <= n < tgt_len:
                    bias_np[b, 0, n, :] = 0.5 * tail_bias_vec

            # ARCH-4: Alignment-Guided Spatial Bias on individual seed tokens
            if self.use_align_guided_cross_attn and timesteps is not None:
                timesteps_np = timesteps.numpy() if isinstance(timesteps, paddle.Tensor) else timesteps
                align_sigma_sq2 = 2.0 * (self.align_spatial_sigma ** 2)
                for j in range(1, min(n + 1, tgt_len)):
                    t_val = float(timesteps_np[b, j - 1])
                    c_j = t_val / max(1.0, float(self.max_ctc_timesteps - 1))
                    bias_np[b, 0, j, :] += self.align_spatial_weight * np.exp(-((u - c_j) ** 2) / align_sigma_sq2)

            # For append token (where tail_mask is 1.0 or position n + 1)
            if tail_mask_np is not None:
                append_pos = np.where(tail_mask_np[b] > 0.5)[0]
                for p in append_pos:
                    if p < tgt_len:
                        bias_np[b, 0, p, :] = tail_bias_vec
            elif (n + 1) < tgt_len and self.use_append_head:
                bias_np[b, 0, n + 1, :] = tail_bias_vec

        return paddle.to_tensor(bias_np, dtype=dtype)

    @staticmethod
    def _insert_tail_slot(seed, seed_lens):
        """Move one dummy slot to the first padding position of every row.

        The dummy carries the last seed id so the causal decoder sees a valid
        token; its hidden state is overwritten by the learned append query.
        Returns the padded ids plus a float mask marking the tail column.
        """
        arr = seed.numpy()
        bsz, width = arr.shape
        out = np.zeros((bsz, width + 1), dtype=arr.dtype)
        mask = np.zeros((bsz, width + 1), dtype="float32")
        for b in range(bsz):
            n = width if seed_lens is None else int(seed_lens[b])
            n = max(0, min(n, width))
            out[b, :n] = arr[b, :n]
            if n > 0:
                out[b, n] = arr[b, n - 1]
            mask[b, n] = 1.0
        return paddle.to_tensor(out, dtype=seed.dtype), paddle.to_tensor(mask)

    def _edit_forward(self, ctc_out, memory, length_logits, seed_ids=None, seed_lens=None, confs=None, timesteps=None):
        if seed_ids is None:
            seeds_np, lens_np, _, _, confs_np, timesteps_np = ctc_seed_and_conf(
                ctc_out, self.max_seed_len, conf_dim=self.ctc_conf_dim
            )
            seed_ids = paddle.to_tensor(seeds_np, dtype="int64")
            seed_lens = paddle.to_tensor(lens_np, dtype="int64")
            confs = paddle.to_tensor(confs_np, dtype="float32") if self.use_ctc_conf_embed else None
            timesteps = paddle.to_tensor(timesteps_np, dtype="int64") if (self.use_ctc_align_embed or self.use_align_guided_cross_attn) else None
        elif confs is None and self.use_ctc_conf_embed:
            _, _, _, _, confs_np, _ = ctc_seed_and_conf(
                ctc_out, self.max_seed_len, conf_dim=self.ctc_conf_dim
            )
            confs = paddle.to_tensor(confs_np, dtype="float32")
        elif timesteps is None and (self.use_ctc_align_embed or self.use_align_guided_cross_attn):
            _, _, _, _, _, timesteps_np = ctc_seed_and_conf(
                ctc_out, self.max_seed_len, conf_dim=self.ctc_conf_dim
            )
            timesteps = paddle.to_tensor(timesteps_np, dtype="int64")

        hidden = self._seed_hidden(
            memory,
            seed_ids,
            seed_lens,
            length_logits=length_logits,
            with_append=self.use_append_head,
            confs=confs,
            timesteps=timesteps,
        )
        tail_hidden = None
        if self.use_append_head:
            width = seed_ids.shape[1]
            lens_np = (
                seed_lens.numpy()
                if seed_lens is not None
                else np.full([seed_ids.shape[0]], width, dtype="int64")
            )
            tail_idx = np.clip(lens_np, 0, width).reshape([-1, 1, 1])
            tail_idx = np.repeat(tail_idx, hidden.shape[-1], axis=2)
            tail_hidden = paddle.take_along_axis(
                hidden, paddle.to_tensor(tail_idx, dtype="int64"), axis=1
            )
            hidden = hidden[:, :width, :]
        op_logits = self.edit_op_head(hidden)
        tok_logits = self.edit_tok_head(hidden)
        out = {"op_logits": op_logits, "tok_logits": tok_logits,
               "seed_ids": seed_ids, "seed_lens": seed_lens}
        if self.use_explicit_change_head:
            if self.use_ctc_conf_embed and self.use_ctc_conf_in_change_head and confs is not None:
                ch_in = paddle.concat([hidden, confs[:, :hidden.shape[1], :]], axis=-1)
                out["change_logits"] = self.change_head(ch_in).squeeze(-1)
            else:
                out["change_logits"] = self.change_head(hidden).squeeze(-1)
        if tail_hidden is not None:
            append_raw = self.append_head(tail_hidden).reshape([-1, 2])
            out["append_logits"] = append_raw[:, 1] - append_raw[:, 0]
            out["append_tok_logits"] = self.edit_tok_head(tail_hidden).reshape(
                [-1, self.vocab_size]
            )
        return out

    def _corrupt_seed_tokens(self, seeds_np, lens_np, margins_np, alternatives_np=None, confs_np=None, timesteps_np=None):
        """Create guaranteed-ish replacement and truncation positives for factorized training.

        This is training-only denoising: CTC/backbone outputs remain untouched,
        while the edit decoder sees a corrupted or truncated seed on a subset of
        samples. The loss aligns this altered seed against the ground truth.
        """
        if (
            not self.training
            or self.edit_mode not in ("factorized", "token_refine")
            or self.train_seed_corrupt_prob <= 0
        ):
            return seeds_np, lens_np, margins_np, confs_np, timesteps_np
        seeds_np = seeds_np.copy()
        lens_np = lens_np.copy()
        margins_np = margins_np.copy()
        if confs_np is not None:
            confs_np = confs_np.copy()
        if timesteps_np is not None:
            timesteps_np = timesteps_np.copy()
        # Empirical OCR digit confusion pairs for realistic denoising
        confusion_pairs = {
            33: [34, 41, 35, 42, 57, 45, 39],  # '0' -> '1', '8', '2', '9', 'O', 'C', '6'
            34: [33, 37, 40, 36, 35, 51, 62],  # '1' -> '0', '4', '7', '3', '2', 'I', 'T'
            35: [33, 36, 34, 40, 37, 68],      # '2' -> '0', '3', '1', '7', '4', 'Z'
            36: [38, 33, 34, 39, 40, 41, 44],  # '3' -> '5', '0', '1', '6', '7', '8', 'B'
            37: [33, 34, 40, 42, 35, 43],      # '4' -> '0', '1', '7', '9', '2', 'A'
            38: [33, 34, 39, 42, 35, 61, 36],  # '5' -> '0', '1', '6', '9', '2', 'S', '3'
            39: [33, 41, 34, 35, 37, 49, 38],  # '6' -> '0', '8', '1', '2', '4', 'G', '5'
            40: [33, 34, 35, 38, 36, 62, 68],  # '7' -> '0', '1', '2', '5', '3', 'T', 'Z'
            41: [33, 36, 34, 39, 35, 44, 42],  # '8' -> '0', '3', '1', '6', '2', 'B', '9'
            42: [33, 41, 34, 40, 36, 37],      # '9' -> '0', '8', '1', '7', '3', '4'
            61: [38, 41],                       # 'S' -> '5', '8'
            57: [33, 46],                       # 'O' -> '0', 'D'
            44: [41, 36],                       # 'B' -> '8', '3'
            51: [34, 62],                       # 'I' -> '1', 'T'
        }

        for b, n_raw in enumerate(lens_np):
            n = int(n_raw)
            if n <= 0:
                continue

            if self.train_seed_corrupt_mode == "ctc_aware_confusion":
                # 1. Uncertainty-aware sample-level corruption gating:
                # Higher average uncertainty -> higher chance of corruption
                uncert = 1.0 - margins_np[b, :n]
                mean_uncert = float(np.mean(uncert)) if n > 0 else 0.5
                p_corrupt_sample = np.clip(
                    self.train_seed_corrupt_prob * (0.6 + 0.8 * mean_uncert), 0.25, 0.95
                )
                if np.random.random() >= p_corrupt_sample:
                    continue

                r_action = float(np.random.random())
                # Action 1: 25% tail drop to teach append / edge reconstruction
                if r_action < 0.25 and n > 1:
                    lens_np[b] = n - 1
                    if confs_np is not None:
                        confs_np[b, n - 1] = 0.0
                    if timesteps_np is not None:
                        timesteps_np[b, n - 1] = 0
                    continue
                # Action 2: 10% middle drop to teach insert
                elif r_action < 0.35 and n > 2:
                    drop_pos = int(np.random.randint(1, n - 1))
                    seeds_np[b, drop_pos:n - 1] = seeds_np[b, drop_pos + 1:n]
                    margins_np[b, drop_pos:n - 1] = margins_np[b, drop_pos + 1:n]
                    if confs_np is not None:
                        confs_np[b, drop_pos:n - 1] = confs_np[b, drop_pos + 1:n]
                        confs_np[b, n - 1] = 0.0
                    if timesteps_np is not None:
                        timesteps_np[b, drop_pos:n - 1] = timesteps_np[b, drop_pos + 1:n]
                        timesteps_np[b, n - 1] = 0
                    lens_np[b] = n - 1
                    continue
                else:
                    # Action 3: Uncertainty-weighted position selection
                    weights = (uncert + 0.1) ** 2
                    p_pos = weights / np.sum(weights)
                    pos = int(np.random.choice(n, p=p_pos))

                    old = int(seeds_np[b, pos])
                    r_pick = float(np.random.random())
                    new = old
                    if r_pick < 0.55 and alternatives_np is not None:
                        alt = int(alternatives_np[b, pos])
                        if 0 < alt < self.vocab_size and alt != old:
                            new = alt
                    if new == old and r_pick < 0.90 and old in confusion_pairs:
                        cand = confusion_pairs[old]
                        cand = [c for c in cand if c < self.vocab_size and c != old]
                        if cand:
                            new = int(np.random.choice(cand))
                    if new == old:
                        new = int(np.random.randint(1, self.vocab_size))
                        if new == old:
                            new = 1 + (old % (self.vocab_size - 1))
                    seeds_np[b, pos] = new
                    sim_m = float(np.random.uniform(0.01, 0.18))
                    margins_np[b, pos] = sim_m
                    if confs_np is not None:
                        sim_p = float(np.random.uniform(0.35, 0.65))
                        if confs_np.shape[-1] == 2:
                            confs_np[b, pos, 0] = sim_p
                            confs_np[b, pos, 1] = sim_m
                        elif confs_np.shape[-1] == 4:
                            sim_p2 = sim_p - sim_m
                            sim_ent = float(-sim_p * np.log(max(sim_p, 1e-6)) - sim_p2 * np.log(max(sim_p2, 1e-6)))
                            confs_np[b, pos, 0] = sim_p
                            confs_np[b, pos, 1] = sim_p2
                            confs_np[b, pos, 2] = sim_m
                            confs_np[b, pos, 3] = sim_ent
                continue

            if np.random.random() >= self.train_seed_corrupt_prob:
                continue
            r_action = float(np.random.random())
            if self.train_seed_corrupt_mode == "balanced_tail":
                # Balanced mode: 20% tail drop (10% of total), 10% middle drop, 70% substitution
                if r_action < 0.20 and n > 1:
                    lens_np[b] = n - 1
                    if confs_np is not None:
                        confs_np[b, n - 1] = 0.0
                    if timesteps_np is not None:
                        timesteps_np[b, n - 1] = 0
                    continue
                elif r_action < 0.30 and n > 2:
                    drop_pos = int(np.random.randint(1, n - 1))
                    seeds_np[b, drop_pos:n - 1] = seeds_np[b, drop_pos + 1:n]
                    margins_np[b, drop_pos:n - 1] = margins_np[b, drop_pos + 1:n]
                    if confs_np is not None:
                        confs_np[b, drop_pos:n - 1] = confs_np[b, drop_pos + 1:n]
                        confs_np[b, n - 1] = 0.0
                    if timesteps_np is not None:
                        timesteps_np[b, drop_pos:n - 1] = timesteps_np[b, drop_pos + 1:n]
                        timesteps_np[b, n - 1] = 0
                    lens_np[b] = n - 1
                    continue
                else:
                    r = float(np.random.random())
                    if r < 0.40:
                        pos = 0       # 40% corrupt leading token
                    elif r < 0.80:
                        pos = n - 1   # 40% corrupt trailing token
                    else:
                        pos = int(np.random.randint(n)) # 20% random any position
            elif self.train_seed_corrupt_mode in ("boundary_biased", "full_corruption", "tail_biased"):
                # Action 1: Trailing truncation (drop last character) - teaches append_head and tail insertion
                if r_action < 0.40 and n > 1:
                    lens_np[b] = n - 1
                    if confs_np is not None:
                        confs_np[b, n - 1] = 0.0
                    if timesteps_np is not None:
                        timesteps_np[b, n - 1] = 0
                    continue
                # Action 2: Random middle deletion - teaches INSERT_AFTER in middle
                elif r_action < 0.55 and n > 2:
                    drop_pos = int(np.random.randint(1, n - 1))
                    seeds_np[b, drop_pos:n - 1] = seeds_np[b, drop_pos + 1:n]
                    margins_np[b, drop_pos:n - 1] = margins_np[b, drop_pos + 1:n]
                    if confs_np is not None:
                        confs_np[b, drop_pos:n - 1] = confs_np[b, drop_pos + 1:n]
                        confs_np[b, n - 1] = 0.0
                    if timesteps_np is not None:
                        timesteps_np[b, drop_pos:n - 1] = timesteps_np[b, drop_pos + 1:n]
                        timesteps_np[b, n - 1] = 0
                    lens_np[b] = n - 1
                    continue
                # Action 3: Token substitution (boundary biased)
                else:
                    r = float(np.random.random())
                    if r < 0.50:
                        pos = 0       # 50% chance corrupt leading token
                    elif r < 0.80:
                        pos = n - 1   # 30% chance corrupt trailing token
                    else:
                        pos = int(np.random.randint(n)) # 20% random any position
            else:
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
            sim_m = float(np.random.uniform(0.01, 0.18))
            margins_np[b, pos] = sim_m
            if confs_np is not None:
                sim_p = float(np.random.uniform(0.35, 0.65))
                if confs_np.shape[-1] == 2:
                    confs_np[b, pos, 0] = sim_p
                    confs_np[b, pos, 1] = sim_m
                elif confs_np.shape[-1] == 4:
                    sim_p2 = sim_p - sim_m
                    sim_ent = float(-sim_p * np.log(max(sim_p, 1e-6)) - sim_p2 * np.log(max(sim_p2, 1e-6)))
                    confs_np[b, pos, 0] = sim_p
                    confs_np[b, pos, 1] = sim_p2
                    confs_np[b, pos, 2] = sim_m
                    confs_np[b, pos, 3] = sim_ent
        return seeds_np, lens_np, margins_np, confs_np, timesteps_np

    def forward(self, x, targets=None, original_image=None):
        ctc_memory, memory = self._memory(x, original_image)
        ctc_out = self.ctc_head(ctc_memory, targets)
        length_logits = self.length_head(ctc_memory) if self.use_length_head else None
        
        need_conf = self.use_ctc_conf_embed
        need_align = self.use_ctc_align_embed or self.use_align_guided_cross_attn
        seeds_np, lens_np, margins, alternatives, confs_np, timesteps_np = ctc_seed_and_conf(
            ctc_out, self.max_seed_len, conf_dim=self.ctc_conf_dim
        )
        seeds_np, lens_np, margins, confs_np, timesteps_np = self._corrupt_seed_tokens(
            seeds_np, lens_np, margins, alternatives, confs_np=confs_np, timesteps_np=timesteps_np
        )
        seeds = paddle.to_tensor(seeds_np, dtype="int64")
        lens = paddle.to_tensor(lens_np, dtype="int64")
        confs = paddle.to_tensor(confs_np, dtype="float32") if need_conf else None
        timesteps = paddle.to_tensor(timesteps_np, dtype="int64") if need_align else None
        edit_out = self._edit_forward(ctc_out, memory, length_logits, seeds, lens, confs=confs, timesteps=timesteps)
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
            if self.use_explicit_change_head and "change_logits" in edit_out:
                change_prob = paddle.nn.functional.sigmoid(edit_out["change_logits"])
                edit_mask = paddle.logical_and(
                    tok_ids_t != seeds,
                    paddle.logical_and(
                        change_prob >= self.edit_gate_threshold,
                        best_prob - seed_prob >= self.edit_delta_threshold,
                    ),
                )
            else:
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
            if self.edit_ops_enabled:
                # F3: let the 4-way op head also fire DELETE / INSERT_AFTER.
                # The op head is a softmax over 4 classes, so the same
                # confidence + margin rule applies with KEEP as the reference
                # class instead of the seed token.  REPLACE still needs the
                # token head, because it must also emit a character.
                op_probs = paddle.nn.functional.softmax(op_logits, axis=2)
                op_t = paddle.argmax(op_probs, axis=2)
                op_best = paddle.max(op_probs, axis=2)
                keep_prob = paddle.take_along_axis(
                    op_probs, paddle.zeros_like(op_t).unsqueeze(-1), axis=2
                ).squeeze(-1)
                fire = paddle.logical_and(
                    op_t != 0,
                    paddle.logical_and(
                        op_best >= self.edit_op_threshold,
                        op_best - keep_prob >= self.edit_op_delta,
                    ),
                )
                fire = paddle.logical_and(
                    fire, paddle.logical_or(op_t == 2, edit_mask)
                )
                op_ids = paddle.where(fire, op_t, paddle.zeros_like(op_t)).numpy()
        else:
            op_ids = paddle.argmax(op_logits, axis=2).numpy()
        tok_ids = paddle.argmax(edit_out["tok_logits"], axis=2).numpy()
        append_fire = None
        append_tok_ids = None
        if self.use_append_head and "append_logits" in edit_out:
            # F4: the append query fires when its gate and its character clear
            # the confidence rules. Repeated characters (e.g. 004299, 776000) are
            # explicitly allowed when the append gate and token confidence are high.
            gate_prob = paddle.nn.functional.sigmoid(edit_out["append_logits"])
            ap_probs = paddle.nn.functional.softmax(
                edit_out["append_tok_logits"], axis=1
            )
            ap_ids_t = paddle.argmax(ap_probs, axis=1)
            ap_best = paddle.max(ap_probs, axis=1)
            last_seed = np.zeros([len(lens_np)], dtype="int64")
            for b, n in enumerate(lens_np):
                if int(n) > 0:
                    last_seed[b] = int(seeds_np[b, int(n) - 1])
            last_seed_t = paddle.to_tensor(last_seed, dtype="int64")
            last_prob = paddle.take_along_axis(
                ap_probs, last_seed_t.unsqueeze(-1), axis=1
            ).squeeze(-1)
            
            # For distinct character: require margin over last seed character.
            # For repeated character: require strong gate confidence.
            is_diff = (ap_ids_t != last_seed_t).astype("float32")
            margin_pass = paddle.logical_or(
                is_diff * (ap_best - last_prob) >= self.append_delta_threshold,
                paddle.logical_and(ap_ids_t == last_seed_t, gate_prob >= 0.70)
            )
            
            append_fire = (
                (gate_prob >= self.append_gate_threshold).astype("int64").numpy()
                * (ap_best >= self.edit_gate_threshold).astype("int64").numpy()
                * margin_pass.astype("int64").numpy()
            )
            append_tok_ids = ap_ids_t.numpy()
        refined = []
        for b, n in enumerate(lens_np):
            n = int(n)
            seq = apply_edit_ops(seeds_np[b, :n].tolist(),
                                 op_ids[b, :n].tolist(),
                                 tok_ids[b, :n].tolist())
            if append_fire is not None and n > 0 and append_fire[b] > 0:
                seq = seq + [int(append_tok_ids[b])]
            refined.append(seq)
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
            "change_logits": edit_out["change_logits"].numpy() if "change_logits" in edit_out else None,
            "change_probs": paddle.nn.functional.sigmoid(edit_out["change_logits"]).numpy() if "change_logits" in edit_out else None,
            "ctc_confs": confs.numpy() if confs is not None else None,
            "timesteps": timesteps.numpy() if timesteps is not None else None,
            "edit_append_fire": append_fire,
            "edit_append_tok_ids": append_tok_ids,
            "refined_ids": refined,
        }
        return {"ctc": refined_probs, "branch_debug": branch_debug}
