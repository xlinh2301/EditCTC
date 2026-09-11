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
    for b in range(bsz):
        prev, n = -1, 0
        for t, tid in enumerate(ids[b].tolist()):
            if tid != 0 and tid != prev and n < max_seed_len:
                seeds[b, n] = tid
                top2 = np.partition(probs[b, t], -2)[-2:]
                margins[b, n] = float(top2[-1] - top2[-2])
                n += 1
            prev = tid
        lens[b] = n
    return seeds, lens, margins


class MultiHeadEditRefineNRTR(MultiHead):
    """CTC + integrated NRTR-seed edit decoder, without an NRTR loss branch."""

    def __init__(self, in_channels, out_channels_list, **kwargs):
        # The head_list still contains CTCHead + NRTRHead so MultiHead builds
        # the pretrained NRTR module under the stable key `gtc_head.*`.
        super().__init__(in_channels, out_channels_list, **kwargs)
        assert self.use_length_head, "integrated NRTR edit head requires length head"
        self.use_original_image = True
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
        self.edit_op_head = nn.Linear(self.nrtr_dim, 4)
        self.edit_tok_head = nn.Linear(self.nrtr_dim, self.vocab_size)
        self.edit_head_dropout = nn.Dropout(kwargs.get("edit_head_dropout", 0.2))
        self.edit_allowed_ops = kwargs.get("edit_allowed_ops")

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
        parts = [ctc_memory if self.ctc_mem_proj is None else self.ctc_mem_proj(ctc_memory)]
        if original_image is not None:
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
            seeds_np, lens_np, _ = ctc_seed_and_margin(ctc_out, self.max_seed_len)
            seed_ids = paddle.to_tensor(seeds_np, dtype="int64")
            seed_lens = paddle.to_tensor(lens_np, dtype="int64")
        hidden = self._seed_hidden(memory, seed_ids)
        op_logits = self.edit_op_head(hidden)
        tok_logits = self.edit_tok_head(hidden)
        return {"op_logits": op_logits, "tok_logits": tok_logits,
                "seed_ids": seed_ids, "seed_lens": seed_lens}

    def forward(self, x, targets=None, original_image=None):
        ctc_memory, memory = self._memory(x, original_image)
        ctc_out = self.ctc_head(ctc_memory, targets)
        length_logits = self.length_head(ctc_memory) if self.use_length_head else None
        seeds_np, lens_np, margins = ctc_seed_and_margin(ctc_out, self.max_seed_len)
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
            if blocked:
                op_logits[:, :, blocked] = -1e9
        op_ids = paddle.argmax(op_logits, axis=2).numpy()
        tok_ids = paddle.argmax(edit_out["tok_logits"], axis=2).numpy()
        refined = []
        for b, n in enumerate(lens_np):
            refined.append(apply_edit_ops(seeds_np[b, : int(n)].tolist(),
                                          op_ids[b, : int(n)].tolist(),
                                          tok_ids[b, : int(n)].tolist()))
        return build_refined_ctc_probs(refined, self.vocab_size, EditRefineDecoder.BLANK_ID)
