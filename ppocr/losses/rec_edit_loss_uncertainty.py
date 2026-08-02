# Uncertainty-gated variant of EditLoss (Exp1 of the uncertainty-gating
# investigation). See experiments/editrefine_probe_v6_uncertainty/SPEC.md.
#
# Deliberately a NEW file: rec_edit_loss.py's EditLoss reproduces the
# reported 96.36% 5-seed full-branch result and stays untouched. Reuses its
# `levenshtein_ops` / `IGNORE_INDEX` / op-id constants unchanged (imported).
#
# What changes vs. EditLoss: op_loss/tok_loss are still built from the exact
# same Wagner-Fischer (op_t, tok_t) targets (the *targets* are not what's
# imbalanced -- KEEP being 94:1 over REPLACE is a property of the data,
# unavoidable at target-construction time). What changes is which positions
# COUNT toward the loss average: instead of averaging op_loss/tok_loss over
# every seed position (where the ~94:1 KEEP:REPLACE ratio drowns the signal
# from ~REPLACE/DELETE/INSERT in gradient-mean terms), we average only over
# a gated subset selected by per-position uncertainty (margin = top1-top2
# CTC frame prob, computed in rec_edit_refine_head_uncertainty.py and passed
# in via predicts["seed_margin"]).
#
# gate_mode="margin": PERCENTILE-based, not a fixed absolute threshold.
#   mask selects the bottom `gate_percentile` fraction of valid seed
#   positions in this batch, ranked by margin (ascending -- lowest margin
#   = least certain = most in need of refinement). Low margin = the two
#   most likely characters were nearly tied at that frame -- exactly the
#   "8: 0.51 / 3: 0.48" case from the design discussion, which a naive
#   confidence threshold (P(top1) > tau) would never catch since P(top1)
#   alone can be high even when top1 is wrong.
#
#   A FIXED absolute margin_tau was tried first and rejected: sanity job
#   55056 showed EditActivationRate decaying monotonically from 25% (epoch
#   1) to ~1.5% (epoch 5) as the model's overall confidence rose during
#   training -- extrapolated to a full 50-epoch run this collapses toward
#   ~0%, exactly the failure mode flagged during design ("nếu cuối cùng
#   0.2% -> gate đang collapse"). Percentile-based selection instead keeps
#   the activation rate pinned at ~gate_percentile for the entire run,
#   independent of how confident the model becomes overall -- it always
#   picks *relatively* the least-certain positions in each batch.
#
# gate_mode="random": a REQUIRED control, not an alternative training mode.
#   Selects the SAME NUMBER of active positions per batch as the margin
#   mask would have (computed internally regardless of gate_mode), but
#   picks them uniformly at random instead of by margin. If margin-gate and
#   random-gate score the same at 5-seed, margin carries no information and
#   the win (if any) comes purely from "fewer KEEP positions drowning the
#   loss", not from the model targeting genuinely uncertain positions. Only
#   a clear margin > random gap is real evidence for uncertainty-gating.
#
# Either mode logs `activation_rate` (fraction of valid seed positions with
# mask==1) back up through MultiLossEditRefineUncertainty so it shows up in
# the training log next to CTCLoss/EditLoss every step -- watch for it
# collapsing to ~0% (gate never fires, no different from full EditLoss with
# an always-KEEP-consistent no-op) or ballooning past ~50% (gate isn't
# selective, no different from ungated EditLoss).

from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import nn

from .rec_edit_loss import IGNORE_INDEX, levenshtein_ops


class EditLossUncertainty(nn.Layer):
    def __init__(
        self,
        max_length=25,
        gate_percentile=0.10,
        gate_mode="margin",
        **kwargs,
    ):
        super().__init__()
        self.max_length = max_length
        assert 0.0 < gate_percentile < 1.0, gate_percentile
        self.gate_percentile = gate_percentile
        assert gate_mode in ("margin", "random"), gate_mode
        self.gate_mode = gate_mode
        # reduction="none" -- masking/averaging is done manually below so
        # the mask can restrict which positions contribute to the mean.
        self.op_loss_func = nn.CrossEntropyLoss(
            reduction="none", ignore_index=IGNORE_INDEX
        )
        self.tok_loss_func = nn.CrossEntropyLoss(
            reduction="none", ignore_index=IGNORE_INDEX
        )

    def build_targets(self, seed_ids, seed_lens, gt_ids, gt_lens):
        # Identical to EditLoss.build_targets -- targets are unaffected by
        # gating, only the loss AVERAGE over them is.
        batch_size, k = seed_ids.shape
        op_t = np.full((batch_size, k), IGNORE_INDEX, dtype="int64")
        tok_t = np.full((batch_size, k), IGNORE_INDEX, dtype="int64")
        for b in range(batch_size):
            n = int(seed_lens[b])
            g = int(gt_lens[b])
            if n == 0:
                continue
            seed = seed_ids[b, :n].tolist()
            gt = gt_ids[b, :g].tolist()
            ops, tok = levenshtein_ops(seed, gt)
            op_t[b, :n] = ops
            tok_t[b, :n] = tok
        return op_t, tok_t

    def forward(self, predicts, batch):
        op_logits = predicts["op_logits"]
        tok_logits = predicts["tok_logits"]
        seed_margin = predicts["seed_margin"]  # (B, K) float32

        seed_ids = predicts["seed_ids"].numpy()
        seed_lens = predicts["seed_lens"].numpy()
        label_ctc, length = batch
        gt_ids = label_ctc.numpy().astype("int64")
        gt_lens = length.numpy().astype("int64")

        op_t_np, tok_t_np = self.build_targets(seed_ids, seed_lens, gt_ids, gt_lens)
        valid_np = (op_t_np != IGNORE_INDEX)  # (B, K) bool -- real seed positions

        if not valid_np.any():
            zero = (op_logits.sum() + tok_logits.sum()) * 0.0
            return {
                "loss": zero,
                "op_loss": zero,
                "tok_loss": zero,
                "activation_rate": zero,
            }

        # --- gating mask -----------------------------------------------
        # Percentile threshold computed fresh every batch over this batch's
        # own pool of valid margins, so the resulting activation rate stays
        # pinned near `gate_percentile` regardless of how the model's
        # overall confidence drifts across training (see module docstring
        # for why a fixed absolute threshold was rejected).
        margin_np = seed_margin.numpy()
        valid_margins = margin_np[valid_np]
        threshold = np.quantile(valid_margins, self.gate_percentile)
        margin_mask_np = np.logical_and(valid_np, margin_np <= threshold)

        if self.gate_mode == "margin":
            mask_np = margin_mask_np
        else:  # "random" control -- match margin's activation COUNT exactly
            mask_np = np.zeros_like(valid_np)
            n_active = int(margin_mask_np.sum())
            valid_flat_idx = np.flatnonzero(valid_np.reshape(-1))
            if n_active > 0 and valid_flat_idx.size > 0:
                n_active = min(n_active, valid_flat_idx.size)
                chosen = np.random.choice(valid_flat_idx, size=n_active, replace=False)
                mask_flat = mask_np.reshape(-1)
                mask_flat[chosen] = True
                mask_np = mask_flat.reshape(mask_np.shape)

        n_valid = int(valid_np.sum())
        n_active = int(mask_np.sum())
        activation_rate = paddle.to_tensor(
            n_active / max(n_valid, 1), dtype="float32"
        )

        if n_active == 0:
            # Gate fired on nothing this batch (can happen early/late in
            # training) -- zero loss this step rather than 0/0 NaN. Still
            # report the real (zero) activation_rate so it's visible in
            # the log, not silently masked by a fallback loss value.
            zero = (op_logits.sum() + tok_logits.sum()) * 0.0
            return {
                "loss": zero,
                "op_loss": zero,
                "tok_loss": zero,
                "activation_rate": activation_rate,
            }

        op_t = paddle.to_tensor(op_t_np, dtype="int64")
        tok_t = paddle.to_tensor(tok_t_np, dtype="int64")
        mask = paddle.to_tensor(mask_np.astype("float32"))

        b, k, c_op = op_logits.shape
        # ignore_index in the per-element loss already zeroes out padded
        # (non-seed) positions' contribution to `.sum()`/mask math below --
        # but paddle's CrossEntropyLoss(reduction="none") with ignore_index
        # still returns a (finite, defined) value at ignored positions, so
        # we rely on `mask` (which is already valid_np-gated) to exclude
        # them, not on the loss function's own ignore behavior.
        op_loss_elem = self.op_loss_func(
            op_logits.reshape([b * k, c_op]), op_t.reshape([b * k])
        ).reshape([b, k])
        c_tok = tok_logits.shape[-1]
        tok_loss_elem = self.tok_loss_func(
            tok_logits.reshape([b * k, c_tok]),
            tok_t.reshape([b * k]),
        ).reshape([b, k])

        denom = mask.sum()
        op_loss = (op_loss_elem * mask).sum() / denom
        tok_loss = (tok_loss_elem * mask).sum() / denom
        loss = op_loss + tok_loss
        return {
            "loss": loss,
            "op_loss": op_loss,
            "tok_loss": tok_loss,
            "activation_rate": activation_rate,
        }
