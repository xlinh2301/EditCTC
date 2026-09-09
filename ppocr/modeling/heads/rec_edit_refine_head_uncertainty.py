# MultiHead + CTC-Seeded Edit Refinement Decoder -- UNCERTAINTY-GATED variant
# (Exp1 of the uncertainty-gating investigation, see
# experiments/editrefine_probe_v6_uncertainty/SPEC.md).
#
# Deliberately a NEW file, not an edit of rec_edit_refine_head.py: the
# existing MultiHeadEditRefine reproduces the reported 96.36% 5-seed
# full-branch result and must stay untouched. This file reuses its
# EditRefineDecoder / apply_edit_ops / build_refined_ctc_probs unchanged
# (imported, not copied) and only adds what Exp1 actually needs: per-seed-
# position margin (top1-top2 softmax prob at that position's source CTC
# frame) and the frame index itself, both computed once here and handed to
# EditLossUncertainty so the loss can build --and later log-- the gating
# mask without recomputing anything.
#
# Exp1 scope reminder: NO cropping, NO change to decoder cross-attention
# (still attends the full ctc_encoder, exactly like the full-branch model),
# NO change at eval/inference (same _eval_refined_output as the parent --
# masking only affects which positions count toward EditLoss during
# training). If this doesn't move 5-seed accuracy, there's no reason to
# build Exp2's ROI cropping.

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


def build_seed_with_frames(decoder: EditRefineDecoder, ctc_logits):
    """Same greedy-decode + blank/repeat-collapse as EditRefineDecoder.build_seed,
    but additionally tracks, for each emitted seed position, (a) the index of
    the CTC frame where that character was FIRST emitted (the frame that
    survives the collapse) and (b) that frame's margin = top1_prob - top2_prob
    over the softmax'd class distribution at that frame.

    Frame index is exactly what Exp2's ROI cropping will need too (crop
    ctc_encoder around this frame) -- computed once here so both experiments
    share the same notion of "which frame does seed position i come from".

    Returns:
        seeds: (B, max_seed_len) int64, same convention as build_seed.
        seed_lens: (B,) int64.
        seed_frame_idx: (B, max_seed_len) int64, 0 where seed slot is unused
            (past seed_lens[b]) -- never read past seed_lens in either loss
            or head, same padding convention as `seeds` itself.
        seed_margin: (B, max_seed_len) float32, top1-top2 prob at
            seed_frame_idx[b, i]; 0.0 for unused slots.
    """
    probs = nn.functional.softmax(ctc_logits, axis=2)
    probs_np = probs.numpy()
    ids_np = np.argmax(probs_np, axis=2)  # (B, T)

    batch_size = ids_np.shape[0]
    max_seed_len = decoder.max_seed_len
    seeds = np.zeros((batch_size, max_seed_len), dtype="int64")
    seed_lens = np.zeros((batch_size,), dtype="int64")
    seed_frame_idx = np.zeros((batch_size, max_seed_len), dtype="int64")
    seed_margin = np.zeros((batch_size, max_seed_len), dtype="float32")

    for b in range(batch_size):
        prev = -1
        out_ids = []
        out_frames = []
        for t, tid in enumerate(ids_np[b].tolist()):
            if tid != EditRefineDecoder.BLANK_ID and tid != prev:
                out_ids.append(tid)
                out_frames.append(t)
                if len(out_ids) >= max_seed_len:
                    break
            prev = tid
        n = len(out_ids)
        seed_lens[b] = n
        if n:
            seeds[b, :n] = out_ids
            seed_frame_idx[b, :n] = out_frames
            # margin at each kept frame: top1 - top2 prob of the full
            # per-frame distribution (not just the two class ids involved
            # in KEEP/REPLACE -- a true top1/top2 margin, matching the
            # "8: 0.51 / 3: 0.48" example from the design discussion).
            frame_probs = probs_np[b, out_frames, :]  # (n, vocab)
            sorted_probs = np.sort(frame_probs, axis=1)
            seed_margin[b, :n] = sorted_probs[:, -1] - sorted_probs[:, -2]

    return seeds, seed_lens, seed_frame_idx, seed_margin


class MultiHeadEditRefineUncertainty(MultiHead):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__(in_channels, out_channels_list, **kwargs)
        # Optional inference-only diagnostics.  The normal output remains the
        # refined CTC tensor; when enabled, _eval_refined_output wraps that
        # tensor with per-branch tensors so infer_rec.py can write a JSONL
        # trace without changing post-processing or training behavior.
        self.branch_debug = kwargs.get("branch_debug", False)
        assert self.use_length_head, (
            "MultiHeadEditRefineUncertainty requires use_length_head: true -- "
            "the length embedding is one of the edit decoder's inputs."
        )
        self.edit_refine_head = EditRefineDecoder(
            in_channels=self.ctc_encoder.out_channels,
            vocab_size=out_channels_list["CTCLabelDecode"],
            edit_dim=kwargs.get("edit_hidden", 128),
            num_layers=kwargs.get("edit_layers", 3),
            nhead=kwargs.get("edit_nhead", 4),
            max_seed_len=kwargs.get("length_max", 25),
            length_max=kwargs.get("length_max", 25),
            dropout=kwargs.get("edit_dropout", 0.1),
            head_dropout=kwargs.get("edit_head_dropout", 0.2),
        )

    def forward(self, x, targets=None):
        if self.use_pool:
            x = self.pool(
                x.reshape([0, 3, -1, self.in_channels]).transpose([0, 3, 1, 2])
            )
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = dict()
        head_out["ctc"] = ctc_out
        head_out["ctc_neck"] = ctc_encoder

        if not self.training:
            # Exp1 makes no eval-time changes: reuse the exact same
            # apply-ops / re-pack path as the full-branch model.
            return self._eval_refined_output(ctc_out, ctc_encoder, x)

        if self.use_length_head:
            head_out["length"] = self.length_head(ctc_encoder)
        if self.gtc_head == "sar":
            head_out["sar"] = self.sar_head(x, targets[1:])
        else:
            head_out["gtc"] = self.gtc_head(self.before_gtc(x), targets[1:])

        # Decoder still forwards on the FULL seed -- Exp1 does not crop or
        # restrict cross-attention. Only the loss (EditLossUncertainty)
        # decides which positions count, via seed_margin below.
        edit_out = self.edit_refine_head(ctc_out, ctc_encoder, head_out.get("length"))

        # Additive: compute margin/frame-idx from this same ctc_out (one
        # extra argmax+sort pass, negligible next to the transformer fwd/bwd
        # already happening). Kept separate from build_seed() inside
        # EditRefineDecoder.forward() above (that call already produced its
        # own seed for the decoder) -- recomputing here is intentional and
        # cheap, avoids threading a new return value through the decoder's
        # public forward() signature that every other head/config also
        # calls unchanged.
        _, _, seed_frame_idx, seed_margin = build_seed_with_frames(
            self.edit_refine_head, ctc_out
        )
        edit_out["seed_margin"] = paddle.to_tensor(seed_margin, dtype="float32")
        head_out["edit"] = edit_out
        return head_out

    def _eval_refined_output(self, ctc_out, ctc_encoder, x):
        # Identical to MultiHeadEditRefine's eval path (see
        # rec_edit_refine_head.py) -- Exp1 makes no inference-time change.
        length_logits = (
            self.length_head(ctc_encoder) if self.use_length_head else None
        )

        # NRTR is a train-time auxiliary branch in the original model.  Run it
        # only for diagnostics so normal inference keeps its original cost.
        nrtr_out = None
        if self.branch_debug and self.gtc_head != "sar":
            nrtr_out = self.gtc_head(self.before_gtc(x))

        seeds_np, seed_lens_np = self.edit_refine_head.build_seed(ctc_out)
        seeds = paddle.to_tensor(seeds_np, dtype="int64")

        edit_out = self.edit_refine_head(
            memory=ctc_encoder,
            length_logits=length_logits,
            seed_ids=seeds,
            seed_lens=paddle.to_tensor(seed_lens_np, dtype="int64"),
        )
        op_ids = paddle.argmax(edit_out["op_logits"], axis=2).numpy()
        tok_ids = paddle.argmax(edit_out["tok_logits"], axis=2).numpy()

        batch_size = seeds_np.shape[0]
        refined_ids_per_sample = []
        for b in range(batch_size):
            n = int(seed_lens_np[b])
            refined = apply_edit_ops(
                seeds_np[b, :n].tolist(),
                op_ids[b, :n].tolist(),
                tok_ids[b, :n].tolist(),
            )
            refined_ids_per_sample.append(refined)

        refined_probs = build_refined_ctc_probs(
            refined_ids_per_sample,
            vocab_size=self.edit_refine_head.vocab_size,
            blank_id=EditRefineDecoder.BLANK_ID,
        )

        if not self.branch_debug:
            return refined_probs

        # Keep this payload deliberately inference-only and detached.  It is
        # consumed by tools/infer_rec.py and never enters a loss or backward
        # pass.  Raw probabilities/logits are retained so a later audit can
        # recompute confidence, length and operation decisions.
        branch_debug = {
            "ctc_probs": ctc_out.numpy(),
            "length_logits": (
                length_logits.numpy() if length_logits is not None else None
            ),
            "nrtr_ids": nrtr_out[0].numpy() if nrtr_out is not None else None,
            "nrtr_probs": nrtr_out[1].numpy() if nrtr_out is not None else None,
            "seed_ids": seeds_np,
            "seed_lens": seed_lens_np,
            "edit_op_logits": edit_out["op_logits"].numpy(),
            "edit_tok_logits": edit_out["tok_logits"].numpy(),
            "edit_op_ids": op_ids,
            "edit_tok_ids": tok_ids,
            "refined_ids": refined_ids_per_sample,
        }
        return {"ctc": refined_probs, "branch_debug": branch_debug}
