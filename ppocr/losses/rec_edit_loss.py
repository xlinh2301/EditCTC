# Edit-op + token-prediction loss for the CTC-Seeded Edit Refinement Decoder.
#
# ALIGNMENT / TARGET CONSTRUCTION -- runs here, in the loss forward, not in
# the dataloader/collate step. Reason: the "seed" sequence being aligned is
# the model's own CTC greedy decode for *this* batch at *this* training
# step, which does not exist until the forward pass has produced logits.
# The dataloader only has the ground-truth label; it cannot precompute an
# alignment against a seed it hasn't seen. So target construction is
# necessarily post-hoc, per training step, here.
#
# Implementation: standard Wagner-Fischer edit-distance DP + backtrack,
# done with plain Python/numpy per sample (paddle tensors -> numpy -> back).
# This is O(batch * seed_len * gt_len) with tiny sequences (<=25 tokens), so
# the Python loop is not a real bottleneck next to the conv/transformer
# forward pass.
#
# DOCUMENTED SIMPLIFICATIONS (deliberate, keeps this single-step and scoped
# down -- no iterative refinement):
#   - Each seed slot carries at most ONE op label (keep / replace / delete /
#     insert-after). If the true alignment would need both "keep this token"
#     and "insert a new token right after it", we only keep the insert (it is
#     the more informative signal since keep is already the default).
#   - Insertions are anchored to the *preceding* seed slot ("insert-after").
#     Insertions that occur before the very first seed token are anchored to
#     slot 0 instead (there is no slot -1). If a gap needs more than one
#     inserted token, only the first is kept (one insert slot per anchor).
#   - If the seed is empty (CTC decoded nothing), no targets can be
#     constructed for that sample; it contributes only ignore_index entries.

from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import nn

IGNORE_INDEX = -100

# op ids
KEEP, REPLACE, DELETE, INSERT_AFTER = 0, 1, 2, 3


def levenshtein_ops(seed, gt):
    """Wagner-Fischer alignment between seed (list[int]) and gt (list[int]).

    Returns (ops, tok): both length len(seed) int lists. ops[i] in
    {KEEP, REPLACE, DELETE, INSERT_AFTER}; tok[i] is the vocab id to predict
    for REPLACE (its replacement) or INSERT_AFTER (the token inserted right
    after position i), else IGNORE_INDEX.
    """
    n, m = len(seed), len(gt)
    if n == 0:
        return [], []

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seed[i - 1] == gt[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,  # delete seed[i-1]
                dp[i, j - 1] + 1,  # insert gt[j-1]
                dp[i - 1, j - 1] + cost,  # keep / replace
            )

    events = []  # (kind, seed_idx, tok) in reverse order
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i, j] == dp[i - 1, j - 1] + (0 if seed[i - 1] == gt[j - 1] else 1)
        ):
            if seed[i - 1] == gt[j - 1]:
                events.append(("keep", i - 1, None))
            else:
                events.append(("replace", i - 1, gt[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            events.append(("delete", i - 1, None))
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            # gap right before seed position i (0-indexed anchor = i - 1,
            # clamped to 0 -- see module docstring).
            events.append(("insert", i - 1, gt[j - 1]))
            j -= 1
        else:  # pragma: no cover - defensive, dp construction guarantees a path
            break
    events.reverse()

    ops = [KEEP] * n
    tok = [IGNORE_INDEX] * n
    for kind, idx, t in events:
        if kind == "replace":
            ops[idx] = REPLACE
            tok[idx] = t
        elif kind == "delete":
            ops[idx] = DELETE
        elif kind == "insert":
            anchor = max(idx, 0)
            ops[anchor] = INSERT_AFTER
            tok[anchor] = t
        # "keep" leaves the default KEEP / IGNORE_INDEX in place
    return ops, tok


class EditLoss(nn.Layer):
    def __init__(self, max_length=25, op_class_weights=None, **kwargs):
        super(EditLoss, self).__init__()
        self.max_length = max_length
        # op_class_weights: optional list[float] of length 4, order
        # [KEEP, REPLACE, DELETE, INSERT_AFTER]. Derived empirically from
        # real op-label frequency on this dataset (see
        # workdir_text_rec/experiments/editrefine_probe/BOTTLENECK_ANALYSIS.md
        # for the measurement) -- KEEP outnumbers REPLACE ~94:1 and
        # DELETE/INSERT_AFTER by 3+ orders of magnitude, so plain
        # unweighted CE makes "always predict KEEP" the trivial optimum.
        # Weights are sqrt(inverse-frequency), capped, rather than raw
        # inverse-frequency: DELETE/INSERT_AFTER have only a handful of
        # real examples in the whole dataset, so raw inverse-freq weights
        # (~1000x) would blow up gradient variance on essentially noise-
        # level per-class sample counts.
        op_weight_t = (
            paddle.to_tensor(op_class_weights, dtype="float32")
            if op_class_weights is not None
            else None
        )
        self.op_loss_func = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=IGNORE_INDEX, weight=op_weight_t
        )
        self.tok_loss_func = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=IGNORE_INDEX
        )

    def build_targets(self, seed_ids, seed_lens, gt_ids, gt_lens):
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

        # `batch` may carry precomputed (op_t, tok_t) targets as its 3rd/4th
        # elements (text-only pretraining: seed_ids come straight from the
        # dataset unchanged by the model -- see rec_edit_refine_head.py's
        # pass-through when `seed_ids` is supplied -- so the Wagner-Fischer
        # alignment is identical every epoch and can be computed once
        # up front instead of on every training step. Same `levenshtein_ops`
        # function either way, just called once instead of once/epoch --
        # no change to alignment logic/results, only to when it runs.
        # Joint training (batch has no precomputed targets) is unaffected:
        # the seed there comes from the model's own CTC decode at *this*
        # step, so it cannot be precomputed and still recomputes here.
        if len(batch) >= 4:
            op_t_np, tok_t_np = batch[2], batch[3]
            if hasattr(op_t_np, "numpy"):
                op_t_np = op_t_np.numpy()
            if hasattr(tok_t_np, "numpy"):
                tok_t_np = tok_t_np.numpy()
        else:
            seed_ids = predicts["seed_ids"].numpy()
            seed_lens = predicts["seed_lens"].numpy()

            label_ctc, length = batch
            gt_ids = label_ctc.numpy().astype("int64")
            gt_lens = length.numpy().astype("int64")

            op_t_np, tok_t_np = self.build_targets(seed_ids, seed_lens, gt_ids, gt_lens)

        if not (op_t_np != IGNORE_INDEX).any():
            # Degenerate batch (e.g. every sample's CTC decode is empty).
            # Keep the term in the autograd graph as an exact zero instead
            # of letting CrossEntropyLoss divide 0/0 -> NaN.
            zero = (op_logits.sum() + tok_logits.sum()) * 0.0
            return {"loss": zero, "op_loss": zero, "tok_loss": zero}

        op_t = paddle.to_tensor(op_t_np, dtype="int64")
        tok_t = paddle.to_tensor(tok_t_np, dtype="int64")

        b, k, c_op = op_logits.shape
        op_loss = self.op_loss_func(
            op_logits.reshape([b * k, c_op]), op_t.reshape([b * k])
        )
        c_tok = tok_logits.shape[-1]
        tok_loss = self.tok_loss_func(
            tok_logits.reshape([b * k, c_tok]), tok_t.reshape([b * k])
        )
        loss = op_loss + tok_loss
        return {"loss": loss, "op_loss": op_loss, "tok_loss": tok_loss}
