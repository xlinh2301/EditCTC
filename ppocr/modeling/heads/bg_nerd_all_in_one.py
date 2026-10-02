# -*- coding: utf-8 -*-
"""
BG-NERD ALL-IN-ONE
==================

Single-file version of the redesigned Boundary-Gap NERD.

Contains:
1. MultiHeadEditRefineBGNRTR
2. Levenshtein alignment + target builder
3. EditLossBoundaryGap
4. MultiLossEditRefineBG
5. Round-trip self-test

Expected repository dependencies:
- ppocr/modeling/heads/rec_edit_refine_nrtr_head.py
- ppocr/losses/rec_multi_loss.py

Integration:
- put this file somewhere importable, e.g.
  ppocr/modeling/heads/bg_nerd_all_in_one.py

Then expose:
  MultiHeadEditRefineBGNRTR
  MultiLossEditRefineBG

in the corresponding head/loss registry/import files.

Important:
- up to MAX_GAP_INSERT inserted tokens per gap, as a count head plus one char
  head per slot (count 0..4).  A per-slot binary INSERT/NO_INSERT head was
  rejected: it splits one decision into three, each far more imbalanced than
  the count itself.
- samples whose gap needs more than MAX_GAP_INSERT insertions are dropped from
  the edit loss and counted in `BGGapSkipRatio` instead of raising.  The count
  comes from `diag_bg_seed_counts.py` (job 71521) over the whole curated train
  set: 30 of 2496 samples (1.20%) exceeded K=3, 28 needed 4 and 2 needed 5, and
  every case was an empty or near-empty seed against a 4-6 char GT.
- bidirectional edit decoder
- no text-level random seed corruption in the main experiment
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from ppocr.modeling.heads.rec_edit_refine_nrtr_head import (
    MultiHeadEditRefineNRTR,
    ctc_seed_and_margin,
    build_refined_ctc_probs,
    EditRefineDecoder,
)
from ppocr.losses.rec_multi_loss import MultiLoss


# ============================================================
# CONSTANTS
# ============================================================

IGNORE_INDEX = -100

TOKEN_KEEP = 0
TOKEN_REPLACE = 1
TOKEN_DELETE = 2

# Insertions per gap are a count in 0..MAX_GAP_INSERT, not a binary flag.
# Raise MAX_GAP_INSERT only if an audit of the real error bank shows a gap that
# needs more.
# The number of insertions a gap can need is bounded by the GT length of the
# sample, which is at most 6 on the curated train set.  K=4 covers 2494 of 2496
# measured samples (job 71521); the residual is handled by the skip path in
# build_targets_full and reported as BGGapSkipRatio.
MAX_GAP_INSERT = 4
GAP_COUNT_CLASSES = MAX_GAP_INSERT + 1


# ============================================================
# MODEL
# ============================================================

class MultiHeadEditRefineBGNRTR(MultiHeadEditRefineNRTR):
    """Boundary-Gap CTC-seeded refinement.

    For a seed of N tokens:
        G0, T0, G1, T1, ... , T(N-1), GN

    Token head:
        KEEP / REPLACE / DELETE

    Gap head:
        count in 0..MAX_GAP_INSERT, plus MAX_GAP_INSERT char heads, the k-th of
        which is trained only on gaps whose count is at least k

    GN is the explicit trailing gap, so missing final digits are representable.
    """

    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__(in_channels, out_channels_list, **kwargs)

        self.bg_token_op_head = nn.Linear(self.nrtr_dim, 3)
        self.bg_token_char_head = nn.Linear(self.nrtr_dim, self.vocab_size)

        self.bg_gap_count_head = nn.Linear(self.nrtr_dim, GAP_COUNT_CLASSES)
        self.bg_gap_char_heads = nn.LayerList(
            [
                nn.Linear(self.nrtr_dim, self.vocab_size)
                for _ in range(MAX_GAP_INSERT)
            ]
        )

        self.bg_gap_query = self.create_parameter(
            shape=[1, 1, self.nrtr_dim],
            default_initializer=nn.initializer.Normal(std=self.nrtr_dim ** -0.5),
        )
        self.bg_token_type = self.create_parameter(
            shape=[1, 1, self.nrtr_dim],
            default_initializer=nn.initializer.Normal(std=self.nrtr_dim ** -0.5),
        )
        self.bg_gap_type = self.create_parameter(
            shape=[1, 1, self.nrtr_dim],
            default_initializer=nn.initializer.Normal(std=self.nrtr_dim ** -0.5),
        )

        self.bg_token_op_threshold = float(kwargs.get("bg_token_op_threshold", 0.60))
        self.bg_token_op_delta = float(kwargs.get("bg_token_op_delta", 0.10))
        self.bg_token_char_threshold = float(kwargs.get("bg_token_char_threshold", 0.55))

        self.bg_gap_threshold = float(kwargs.get("bg_gap_threshold", 0.70))
        self.bg_gap_delta = float(kwargs.get("bg_gap_delta", 0.20))
        self.bg_gap_char_threshold = float(kwargs.get("bg_gap_char_threshold", 0.60))

        # Initialize new char heads from the existing edit token head.
        with paddle.no_grad():
            try:
                self.bg_token_char_head.weight.set_value(self.edit_tok_head.weight)
                self.bg_token_char_head.bias.set_value(self.edit_tok_head.bias)

                for head in self.bg_gap_char_heads:
                    head.weight.set_value(self.edit_tok_head.weight)
                    head.bias.set_value(self.edit_tok_head.bias)
            except Exception:
                pass

    def _build_bg_queries(self, memory, seed_ids, seed_lens):
        bsz, width = seed_ids.shape

        # Map CTC ids -> NRTR ids.
        nrtr_seed = paddle.where(
            seed_ids > 0,
            seed_ids + 3,
            paddle.zeros_like(seed_ids),
        )

        token_emb = self.gtc_head.embedding(nrtr_seed) + self.bg_token_type

        gap_emb = self.bg_gap_query.expand(
            [bsz, width + 1, self.nrtr_dim]
        )
        gap_emb = gap_emb + self.bg_gap_type

        # G0,T0,G1,T1,...,T(W-1),GW
        parts = []
        for i in range(width):
            parts.append(gap_emb[:, i:i+1, :])
            parts.append(token_emb[:, i:i+1, :])
        parts.append(gap_emb[:, width:width+1, :])

        tgt = paddle.concat(parts, axis=1)
        tgt = self.gtc_head.positional_encoding(tgt)

        # Valid query count for sample with n seed tokens is 2*n + 1.
        # This is padding-only masking, NOT causal masking.
        q_len = tgt.shape[1]
        valid_q_len = seed_lens.astype("int64") * 2 + 1

        pos = paddle.arange(q_len, dtype="int64").reshape([1, 1, 1, q_len])
        key_valid = pos < valid_q_len.reshape([-1, 1, 1, 1])

        zero = paddle.zeros([1], dtype=tgt.dtype)
        neg = paddle.full([1], -1e4, dtype=tgt.dtype)
        self_mask = paddle.where(key_valid, zero, neg)

        # Bidirectional refinement: no triangular causal mask.
        for layer in self.gtc_head.decoder:
            tgt = layer(tgt, memory, self_mask=self_mask)

        tgt = self.edit_head_dropout(tgt)

        gap_hidden = tgt[:, 0::2, :]
        token_hidden = tgt[:, 1::2, :]

        ar_t = paddle.arange(width, dtype="int64").reshape([1, width])
        token_valid = (
            ar_t < seed_lens.reshape([-1, 1])
        ).astype("float32")

        ar_g = paddle.arange(width + 1, dtype="int64").reshape([1, width + 1])
        gap_valid = (
            ar_g <= seed_lens.reshape([-1, 1])
        ).astype("float32")

        return token_hidden, gap_hidden, token_valid, gap_valid

    def _bg_edit_forward(self, memory, seed_ids, seed_lens):
        token_h, gap_h, token_valid, gap_valid = self._build_bg_queries(
            memory, seed_ids, seed_lens
        )

        out = {
            "token_op_logits": self.bg_token_op_head(token_h),
            "token_char_logits": self.bg_token_char_head(token_h),

            "gap_count_logits": self.bg_gap_count_head(gap_h),

            "token_valid": token_valid,
            "gap_valid": gap_valid,

            "seed_ids": seed_ids,
            "seed_lens": seed_lens,
        }

        for k, head in enumerate(self.bg_gap_char_heads):
            out["gap_char_logits_{}".format(k + 1)] = head(gap_h)

        return out

    @staticmethod
    def _decode_one(seed, token_ops, token_chars, gap_counts, gap_chars):
        """gap_chars is [n+1, MAX_GAP_INSERT]; row g holds the chars for gap g."""
        out = []

        def emit_gap(g):
            for k in range(int(gap_counts[g])):
                out.append(int(gap_chars[g][k]))

        # G0
        emit_gap(0)

        # Ti then Gi+1
        for i, s in enumerate(seed):
            op = int(token_ops[i])

            if op == TOKEN_KEEP:
                out.append(int(s))
            elif op == TOKEN_REPLACE:
                out.append(int(token_chars[i]))
            elif op == TOKEN_DELETE:
                pass
            else:
                raise ValueError("Invalid token op: {}".format(op))

            emit_gap(i + 1)

        return out

    def _decode_bg(self, edit_out):
        seeds = edit_out["seed_ids"]
        seeds_np = seeds.numpy()
        lens_np = edit_out["seed_lens"].numpy()

        # ---------------- token actions ----------------
        op_p = F.softmax(edit_out["token_op_logits"], axis=-1)

        op_id = paddle.argmax(op_p, axis=-1)
        op_best = paddle.max(op_p, axis=-1)
        keep_p = op_p[:, :, TOKEN_KEEP]

        op_fire = (
            (op_id != TOKEN_KEEP)
            & (op_best >= self.bg_token_op_threshold)
            & ((op_best - keep_p) >= self.bg_token_op_delta)
        )

        char_p = F.softmax(edit_out["token_char_logits"], axis=-1)
        char_id = paddle.argmax(char_p, axis=-1)
        char_best = paddle.max(char_p, axis=-1)

        replace_ok = (
            (op_id == TOKEN_REPLACE)
            & (char_best >= self.bg_token_char_threshold)
            & (char_id != seeds)
        )

        delete_ok = (op_id == TOKEN_DELETE)

        op_fire = op_fire & (replace_ok | delete_ok)

        token_ops = paddle.where(
            op_fire,
            op_id,
            paddle.zeros_like(op_id),
        ).numpy()

        token_chars = char_id.numpy()

        # ---------------- gap actions ----------------
        # Same gate shape as the token ops: argmax is not "no insert", the best
        # class clears an absolute threshold and beats the count-0 class.
        count_p = F.softmax(edit_out["gap_count_logits"], axis=-1)

        count_id = paddle.argmax(count_p, axis=-1)
        count_best = paddle.max(count_p, axis=-1)
        p_zero = count_p[:, :, 0]

        # A gap only fires if every char slot it needs also clears its own char
        # threshold; otherwise the count is forced back to 0 rather than
        # emitting an untrusted character.
        count_ok = (
            (count_id != 0)
            & (count_best >= self.bg_gap_threshold)
            & ((count_best - p_zero) >= self.bg_gap_delta)
            & (edit_out["gap_valid"] > 0.5)
        )

        gap_char_ids = []

        for k in range(MAX_GAP_INSERT):
            lp = F.softmax(
                edit_out["gap_char_logits_{}".format(k + 1)], axis=-1
            )
            gap_char_ids.append(paddle.argmax(lp, axis=-1))

            needs_k = count_id > k
            count_ok = count_ok & (
                (count_id <= k)
                | (paddle.max(lp, axis=-1) >= self.bg_gap_char_threshold)
            )

        gap_count_np = paddle.where(
            count_ok,
            count_id,
            paddle.zeros_like(count_id),
        ).numpy()

        gap_chars_np = np.stack(
            [cid.numpy() for cid in gap_char_ids],
            axis=-1,
        )

        refined = []

        for b, n_raw in enumerate(lens_np):
            n = int(n_raw)

            refined.append(
                self._decode_one(
                    seeds_np[b, :n].tolist(),
                    token_ops[b, :n].tolist(),
                    token_chars[b, :n].tolist(),
                    gap_count_np[b, :n+1].tolist(),
                    gap_chars_np[b, :n+1].tolist(),
                )
            )

        return (
            refined,
            token_ops,
            token_chars,
            gap_count_np,
            gap_chars_np,
        )

    def forward(self, x, targets=None, original_image=None):
        # Reuse current CTC + high-res memory.
        ctc_memory, memory = self._memory(x, original_image)

        ctc_out = self.ctc_head(ctc_memory, targets)

        length_logits = (
            self.length_head(ctc_memory)
            if self.use_length_head
            else None
        )

        seeds_np, lens_np, margins, alternatives = ctc_seed_and_margin(
            ctc_out,
            self.max_seed_len,
        )

        # IMPORTANT:
        # No random textual seed corruption here.
        seeds = paddle.to_tensor(seeds_np, dtype="int64")
        lens = paddle.to_tensor(lens_np, dtype="int64")

        edit_out = self._bg_edit_forward(
            memory,
            seeds,
            lens,
        )

        edit_out["seed_margin"] = paddle.to_tensor(
            margins,
            dtype="float32",
        )

        if self.training:
            out = {
                "ctc": ctc_out,
                "ctc_neck": ctc_memory,
                "edit": edit_out,
            }

            if length_logits is not None:
                out["length"] = length_logits

            return out

        refined, token_ops, token_chars, gap_counts, gap_chars = (
            self._decode_bg(edit_out)
        )

        refined_probs = build_refined_ctc_probs(
            refined,
            self.vocab_size,
            EditRefineDecoder.BLANK_ID,
        )

        if not self.branch_debug:
            return refined_probs

        return {
            "ctc": refined_probs,
            "branch_debug": {
                "ctc_probs": ctc_out.numpy(),
                "length_logits": (
                    length_logits.numpy()
                    if length_logits is not None
                    else None
                ),

                "seed_ids": seeds_np,
                "seed_lens": lens_np,

                "bg_token_op_logits":
                    edit_out["token_op_logits"].numpy(),
                "bg_token_char_logits":
                    edit_out["token_char_logits"].numpy(),

                "bg_gap_count_logits":
                    edit_out["gap_count_logits"].numpy(),
                "bg_gap_char_logits":
                    np.stack(
                        [
                            edit_out["gap_char_logits_{}".format(k + 1)].numpy()
                            for k in range(MAX_GAP_INSERT)
                        ],
                        axis=-1,
                    ),

                "bg_token_ops": token_ops,
                "bg_token_chars": token_chars,

                "bg_gap_counts": gap_counts,
                "bg_gap_chars": gap_chars,

                # The gates and the validity mask decide whether a nonzero
                # count survives to the output.  Ship them with the logits so
                # an audit record can say *why* a gap did not fire instead of
                # only that it did not.
                "gap_valid": edit_out["gap_valid"].numpy(),
                "bg_thresholds": {
                    "token_op_threshold": self.bg_token_op_threshold,
                    "token_op_delta": self.bg_token_op_delta,
                    "token_char_threshold": self.bg_token_char_threshold,
                    "gap_threshold": self.bg_gap_threshold,
                    "gap_delta": self.bg_gap_delta,
                    "gap_char_threshold": self.bg_gap_char_threshold,
                },

                "refined_ids": refined,
            },
        }


# ============================================================
# ALIGNMENT / TARGET BUILDER
# ============================================================

def align_seed_to_gt(seed, gt):
    """Standard Levenshtein alignment with deterministic tie-break."""
    n = len(seed)
    m = len(gt)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.empty((n + 1, m + 1), dtype=object)

    for i in range(1, n + 1):
        dp[i, 0] = i
        bt[i, 0] = "del"

    for j in range(1, m + 1):
        dp[0, j] = j
        bt[0, j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if seed[i - 1] == gt[j - 1] else 1

            choices = [
                (dp[i - 1, j - 1] + sub_cost, 0, "diag"),
                (dp[i - 1, j] + 1,          1, "del"),
                (dp[i, j - 1] + 1,          2, "ins"),
            ]

            cost, _, move = min(
                choices,
                key=lambda x: (x[0], x[1]),
            )

            dp[i, j] = cost
            bt[i, j] = move

    steps = []
    i = n
    j = m

    while i > 0 or j > 0:
        move = bt[i, j]

        if move == "diag":
            steps.append(("diag", i - 1, j - 1))
            i -= 1
            j -= 1

        elif move == "del":
            steps.append(("del", i - 1, None))
            i -= 1

        elif move == "ins":
            steps.append(("ins", None, j - 1))
            j -= 1

        else:
            raise RuntimeError(
                "Invalid backtrace at ({},{})".format(i, j)
            )

    steps.reverse()
    return steps


def build_bg_targets_one(seed, gt):
    """Build one-sample Boundary-Gap targets.

    Each gap carries a count in 0..MAX_GAP_INSERT plus one char per slot.  Slot
    k stays IGNORE_INDEX unless the count reaches k+1.
    """

    n = len(seed)

    token_op = np.full(
        (n,),
        TOKEN_KEEP,
        dtype=np.int64,
    )

    token_char = np.full(
        (n,),
        IGNORE_INDEX,
        dtype=np.int64,
    )

    gap_count = np.zeros(
        (n + 1,),
        dtype=np.int64,
    )

    gap_char = np.full(
        (n + 1, MAX_GAP_INSERT),
        IGNORE_INDEX,
        dtype=np.int64,
    )

    consumed_seed = 0

    for move, si, gj in align_seed_to_gt(seed, gt):

        if move == "diag":
            if seed[si] == gt[gj]:
                token_op[consumed_seed] = TOKEN_KEEP

            else:
                token_op[consumed_seed] = TOKEN_REPLACE
                token_char[consumed_seed] = int(gt[gj])

            consumed_seed += 1

        elif move == "del":
            token_op[consumed_seed] = TOKEN_DELETE
            consumed_seed += 1

        elif move == "ins":
            gap = consumed_seed
            k = int(gap_count[gap])

            if k >= MAX_GAP_INSERT:
                raise ValueError(
                    "Gap {} needs {} consecutive insertions, "
                    "MAX_GAP_INSERT is {}."
                    .format(gap, k + 1, MAX_GAP_INSERT)
                )

            gap_char[gap, k] = int(gt[gj])
            gap_count[gap] = k + 1

    return (
        token_op,
        token_char,
        gap_count,
        gap_char,
    )


def apply_bg_targets(
    seed,
    token_op,
    token_char,
    gap_count,
    gap_char,
):
    """Reference decoder for target round-trip validation."""

    out = []

    def emit_gap(g):
        for k in range(int(gap_count[g])):
            out.append(int(gap_char[g, k]))

    emit_gap(0)

    for i, s in enumerate(seed):

        op = int(token_op[i])

        if op == TOKEN_KEEP:
            out.append(int(s))

        elif op == TOKEN_REPLACE:
            out.append(int(token_char[i]))

        elif op == TOKEN_DELETE:
            pass

        else:
            raise ValueError("Bad token op {}".format(op))

        emit_gap(i + 1)

    return out


# ============================================================
# EDIT LOSS
# ============================================================

class EditLossBoundaryGap(nn.Layer):

    def __init__(
        self,
        token_op_weights=(1.0, 2.0, 4.0),
        gap_count_weights=(1.0, 20.0, 30.0, 40.0, 50.0),

        weight_token_op=1.0,
        weight_replace_char=1.0,
        weight_gap_op=1.0,
        weight_insert_char=1.0,

        weight_length_consistency=0.2,

        **kwargs
    ):
        super().__init__()

        self.weight_token_op = float(weight_token_op)
        self.weight_replace_char = float(weight_replace_char)
        self.weight_gap_op = float(weight_gap_op)
        self.weight_insert_char = float(weight_insert_char)

        self.weight_length_consistency = float(
            weight_length_consistency
        )

        self.token_op_ce = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX,
            weight=paddle.to_tensor(
                token_op_weights,
                dtype="float32",
            ),
        )

        # Insertion in a gap runs at roughly 0.3-0.5% of training gaps, so the
        # count classes need a much steeper weight ladder than the token ops.
        if len(gap_count_weights) != GAP_COUNT_CLASSES:
            raise ValueError(
                "gap_count_weights must have {} entries, got {}".format(
                    GAP_COUNT_CLASSES, len(gap_count_weights)
                )
            )

        self.gap_count_ce = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX,
            weight=paddle.to_tensor(
                gap_count_weights,
                dtype="float32",
            ),
        )

        self.char_ce = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX
        )

    @staticmethod
    def _zero(logits):
        return logits.sum() * 0.0

    def build_targets_full(
        self,
        seed_ids,
        seed_lens,
        gt_ids,
        gt_lens,
    ):
        bsz, width = seed_ids.shape

        token_op = np.full(
            (bsz, width),
            IGNORE_INDEX,
            dtype=np.int64,
        )

        token_char = np.full(
            (bsz, width),
            IGNORE_INDEX,
            dtype=np.int64,
        )

        gap_count = np.full(
            (bsz, width + 1),
            IGNORE_INDEX,
            dtype=np.int64,
        )

        gap_char = np.full(
            (bsz, width + 1, MAX_GAP_INSERT),
            IGNORE_INDEX,
            dtype=np.int64,
        )

        token_valid = np.zeros(
            (bsz, width),
            dtype=np.float32,
        )

        gap_valid = np.zeros(
            (bsz, width + 1),
            dtype=np.float32,
        )

        # 1 for samples the edit loss actually trains on, 0 for skipped ones.
        # Every target of a skipped sample keeps IGNORE_INDEX above, so all the
        # cross-entropies ignore it; this flag is what removes it from the
        # length-consistency mean and from the reported skip ratio.
        sample_valid = np.ones(
            (bsz,),
            dtype=np.float32,
        )

        for b in range(bsz):
            n = int(seed_lens[b])
            g = int(gt_lens[b])

            seed = [
                int(x)
                for x in seed_ids[b, :n].tolist()
            ]

            gt = [
                int(x)
                for x in gt_ids[b, :g].tolist()
            ]

            try:
                t_op, t_char, g_count, g_char = (
                    build_bg_targets_one(seed, gt)
                )
            except ValueError as exc:
                # Only the capacity error is recoverable.  Any other ValueError
                # is a bug in the builder and must stay loud.
                if "MAX_GAP_INSERT is" not in str(exc):
                    raise
                sample_valid[b] = 0.0
                continue

            if n > 0:
                token_op[b, :n] = t_op
                token_char[b, :n] = t_char
                token_valid[b, :n] = 1.0

            gap_count[b, :n + 1] = g_count
            gap_char[b, :n + 1] = g_char
            gap_valid[b, :n + 1] = 1.0

            # Structural invariant.
            roundtrip = apply_bg_targets(
                seed,
                t_op,
                t_char,
                g_count,
                g_char,
            )

            if roundtrip != gt:
                raise AssertionError(
                    "BG round-trip failed: seed={} gt={} got={}"
                    .format(seed, gt, roundtrip)
                )

        targets = {
            "token_op": token_op,
            "token_char": token_char,

            "gap_count": gap_count,
            "gap_char": gap_char,

            "token_valid": token_valid,
            "gap_valid": gap_valid,

            "sample_valid": sample_valid,
        }

        for k in range(MAX_GAP_INSERT):
            targets["gap_char_{}".format(k + 1)] = gap_char[:, :, k]

        return targets

    def forward(self, predicts, batch):
        gt_ids, gt_lens = batch

        seed_ids = predicts["seed_ids"].numpy()
        seed_lens = predicts["seed_lens"].numpy()

        gt_ids_np = (
            gt_ids.numpy()
            if paddle.is_tensor(gt_ids)
            else np.asarray(gt_ids)
        )

        gt_lens_np = (
            gt_lens.numpy()
            if paddle.is_tensor(gt_lens)
            else np.asarray(gt_lens)
        )

        targets = self.build_targets_full(
            seed_ids,
            seed_lens,
            gt_ids_np,
            gt_lens_np,
        )

        t_op_t = paddle.to_tensor(
            targets["token_op"],
            dtype="int64",
        )

        t_char_t = paddle.to_tensor(
            targets["token_char"],
            dtype="int64",
        )

        g_count_t = paddle.to_tensor(
            targets["gap_count"],
            dtype="int64",
        )

        token_op_logits = predicts["token_op_logits"]
        token_char_logits = predicts["token_char_logits"]

        gap_count_logits = predicts["gap_count_logits"]

        loss_token_op = self.token_op_ce(
            token_op_logits.reshape([-1, 3]),
            t_op_t.reshape([-1]),
        )

        if bool(
            (targets["token_char"] != IGNORE_INDEX).any()
        ):
            loss_replace_char = self.char_ce(
                token_char_logits.reshape(
                    [-1, token_char_logits.shape[-1]]
                ),
                t_char_t.reshape([-1]),
            )
        else:
            loss_replace_char = self._zero(
                token_char_logits
            )

        loss_gap_count = self.gap_count_ce(
            gap_count_logits.reshape([-1, GAP_COUNT_CLASSES]),
            g_count_t.reshape([-1]),
        )

        # One char CE per insertion slot.  Slot k only has targets in gaps whose
        # count reaches k+1, so it is skipped when a batch has none.  Averaging
        # over the ACTIVE slots keeps the term on the scale of a single char CE
        # instead of growing with MAX_GAP_INSERT.
        slot_losses = []
        n_active_slots = 0

        for k in range(MAX_GAP_INSERT):
            logits_k = predicts["gap_char_logits_{}".format(k + 1)]
            target_k = paddle.to_tensor(
                targets["gap_char_{}".format(k + 1)],
                dtype="int64",
            )

            if bool((target_k != IGNORE_INDEX).any()):
                slot_losses.append(
                    self.char_ce(
                        logits_k.reshape([-1, logits_k.shape[-1]]),
                        target_k.reshape([-1]),
                    )
                )
                n_active_slots += 1
            else:
                slot_losses.append(self._zero(logits_k))

        loss_insert_char = (
            sum(slot_losses) / max(n_active_slots, 1)
        )

        # Expected final length:
        # seed_len - E[#delete] + E[#insert]
        t_prob = F.softmax(
            token_op_logits,
            axis=-1,
        )

        g_prob = F.softmax(
            gap_count_logits,
            axis=-1,
        )

        t_valid = paddle.to_tensor(
            targets["token_valid"],
            dtype="float32",
        )

        g_valid = paddle.to_tensor(
            targets["gap_valid"],
            dtype="float32",
        )

        exp_delete = (
            t_prob[:, :, TOKEN_DELETE] * t_valid
        ).sum(axis=1)

        count_axis = paddle.arange(
            GAP_COUNT_CLASSES, dtype=g_prob.dtype
        ).reshape([1, 1, GAP_COUNT_CLASSES])

        exp_insert = (
            (g_prob * count_axis).sum(axis=-1) * g_valid
        ).sum(axis=1)

        seed_len_t = paddle.to_tensor(
            seed_lens,
            dtype="float32",
        )

        gt_len_t = paddle.to_tensor(
            gt_lens_np,
            dtype="float32",
        ).reshape([-1])

        expected_len = (
            seed_len_t
            - exp_delete
            + exp_insert
        )

        # Skipped samples contribute no gradient anywhere else, so they must not
        # pull this mean either: their expected_len is just seed_len, which is
        # far from gt_len by construction.
        s_valid = paddle.to_tensor(
            targets["sample_valid"],
            dtype="float32",
        )

        loss_len_cons = (
            paddle.abs(expected_len - gt_len_t) * s_valid
        ).sum() / paddle.clip(s_valid.sum(), min=1.0)

        skip_ratio = 1.0 - s_valid.mean()

        total = (
            self.weight_token_op * loss_token_op
            + self.weight_replace_char * loss_replace_char
            + self.weight_gap_op * loss_gap_count
            + self.weight_insert_char * loss_insert_char
            + self.weight_length_consistency * loss_len_cons
        )

        return {
            "loss": total,

            "token_op_loss": loss_token_op,
            "replace_char_loss": loss_replace_char,

            "gap_count_loss": loss_gap_count,
            "insert_char_loss": loss_insert_char,

            "length_consistency_loss": loss_len_cons,

            "skip_ratio": skip_ratio,
        }


# ============================================================
# MULTI-LOSS WRAPPER
# ============================================================

class MultiLossEditRefineBG(MultiLoss):

    def __init__(self, **kwargs):
        self.weight_edit = float(
            kwargs.get("weight_edit", 1.0)
        )

        super().__init__(**kwargs)

        self.edit_loss = EditLossBoundaryGap(
            token_op_weights=kwargs.get(
                "bg_token_op_weights",
                [1.0, 2.0, 4.0],
            ),

            gap_count_weights=kwargs.get(
                "bg_gap_count_weights",
                [1.0, 20.0, 30.0, 40.0, 50.0],
            ),

            weight_token_op=kwargs.get(
                "bg_weight_token_op",
                1.0,
            ),

            weight_replace_char=kwargs.get(
                "bg_weight_replace_char",
                1.0,
            ),

            weight_gap_op=kwargs.get(
                "bg_weight_gap_op",
                1.0,
            ),

            weight_insert_char=kwargs.get(
                "bg_weight_insert_char",
                1.0,
            ),

            weight_length_consistency=kwargs.get(
                "bg_weight_length_consistency",
                0.2,
            ),
        )

    def forward(self, predicts, batch):
        total = super().forward(
            predicts,
            batch,
        )

        result = self.edit_loss(
            predicts["edit"],
            (batch[1], batch[3]),
        )

        edit_loss = (
            result["loss"]
            * self.weight_edit
        )

        total["EditLoss"] = edit_loss

        total["BGTokenOpLoss"] = (
            result["token_op_loss"]
        )

        total["BGReplaceCharLoss"] = (
            result["replace_char_loss"]
        )

        total["BGGapCountLoss"] = (
            result["gap_count_loss"]
        )

        total["BGInsertCharLoss"] = (
            result["insert_char_loss"]
        )

        total["BGLengthConsistencyLoss"] = (
            result["length_consistency_loss"]
        )

        # Fraction of the batch dropped from the edit loss because a gap needed
        # more than MAX_GAP_INSERT insertions.  Must be visible, never silent.
        total["BGGapSkipRatio"] = (
            result["skip_ratio"]
        )

        total["loss"] = (
            total["loss"]
            + edit_loss
        )

        return total


# ============================================================
# SELF TEST
# ============================================================

def _self_test():
    cases = [
        # trailing insert
        (
            [0, 0, 1, 6, 5],
            [0, 0, 1, 6, 5, 7],
        ),

        # leading insert
        (
            [0, 1, 6, 5, 7],
            [0, 0, 1, 6, 5, 7],
        ),

        # internal insert
        (
            [0, 0, 1, 5, 7],
            [0, 0, 1, 6, 5, 7],
        ),

        # all KEEP
        (
            [0, 0, 1, 6, 5, 7],
            [0, 0, 1, 6, 5, 7],
        ),

        # REPLACE
        (
            [0, 0, 1, 6, 5, 7],
            [0, 0, 1, 6, 5, 8],
        ),

        # DELETE
        (
            [0, 0, 1, 6, 5, 7],
            [0, 0, 1, 5, 7],
        ),

        # repeated trailing digit
        (
            [0, 0, 1, 6, 5],
            [0, 0, 1, 6, 5, 5],
        ),

    ]

    # Multi-insert cases.  seed=[A,B], gt=[A,x,y,z,B] has |n-m|=3 and exactly
    # two exact matches, so the only cost-3 alignment is both diagonals plus
    # three insertions in the middle gap.  The same construction with two
    # middle tokens forces count 2, and with the inserts before the last token
    # forces count 2 in gap 0.  These are the shapes the one-insert-per-gap
    # version could not represent.
    multi = [
        ([1, 9], [1, 6, 5, 7, 9], 3, "count 3, middle gap"),
        ([1, 9], [1, 6, 5, 9], 2, "count 2, middle gap"),
        ([9], [6, 5, 9], 2, "count 2, gap 0"),
        ([5, 5, 5], [5, 5, 5, 7, 8, 9], 3, "count 3, last gap"),

        # count == MAX_GAP_INSERT, the deepest gap the head can emit.  Both the
        # measured training requirement (job 71521: 28 samples needed 4) and the
        # over-capacity case below sit on either side of this value.
        ([1, 9], [1, 6, 5, 7, 8, 9], 4, "count 4, middle gap"),
        ([], [6, 5, 7, 8], 4, "count 4, empty seed"),
        ([5, 5, 5], [5, 5, 5, 7, 8, 9, 4], 4, "count 4, last gap"),
    ]

    seen_max = 0

    for seed, gt, want_count, label in multi:
        t_op, t_char, g_count, g_char = build_bg_targets_one(seed, gt)

        got = apply_bg_targets(seed, t_op, t_char, g_count, g_char)

        assert got == gt, ("multi", label, seed, gt, got)
        assert int(g_count.max()) == want_count, (
            "multi",
            label,
            "expected max count",
            want_count,
            "got",
            g_count.tolist(),
        )

        seen_max = max(seen_max, int(g_count.max()))

        print("PASS", label, seed, "->", gt, "counts", g_count.tolist())

    assert seen_max == MAX_GAP_INSERT, (
        "self-test never reached MAX_GAP_INSERT; the K={} path is untested"
        .format(MAX_GAP_INSERT)
    )

    # Over-capacity must fail loudly, not silently truncate.
    try:
        build_bg_targets_one([1, 9], [1, 2, 3, 4, 5, 6, 9])
    except ValueError as exc:
        print("PASS over-capacity raises:", exc)
    else:
        raise AssertionError(
            "5 consecutive insertions should have raised ValueError "
            "(MAX_GAP_INSERT={})".format(MAX_GAP_INSERT)
        )

    # The skip path is the only thing standing between a 5-insertion sample and
    # a dead run, and it does not fire on the curated set (BGGapSkipRatio stayed
    # 0.000000 through the job 71525 smoke), so exercise it here instead of
    # waiting for a real sample to hit it.
    guard = EditLossBoundaryGap()

    width = 8
    seed_ids = np.zeros((2, width), dtype=np.int64)
    gt_ids = np.zeros((2, width + 8), dtype=np.int64)

    # Row 0: gap 2 would need 5 insertions, one past MAX_GAP_INSERT.
    seed_ids[0, :3] = [0, 1, 9]
    gt_ids[0, :8] = [0, 1, 2, 3, 4, 5, 6, 9]

    # Row 1: control, builds cleanly.
    seed_ids[1, :3] = [0, 1, 9]
    gt_ids[1, :3] = [0, 1, 9]

    tg = guard.build_targets_full(
        seed_ids,
        np.array([3, 3], dtype=np.int64),
        gt_ids,
        np.array([8, 3], dtype=np.int64),
    )

    assert list(tg["sample_valid"]) == [0.0, 1.0], tg["sample_valid"]
    assert (tg["gap_valid"][0] == 0.0).all(), tg["gap_valid"][0]
    assert (tg["token_op"][0] == IGNORE_INDEX).all(), tg["token_op"][0]
    assert (tg["token_op"][1, :3] == TOKEN_KEEP).all(), tg["token_op"][1]
    assert tg["gap_valid"][1, :4].sum() == 4.0, tg["gap_valid"][1]

    print(
        "PASS skip path: row 0 skipped (5 insertions), row 1 built; "
        "sample_valid =", list(tg["sample_valid"])
    )

    for seed, gt in cases:
        t_op, t_char, g_count, g_char = (
            build_bg_targets_one(
                seed,
                gt,
            )
        )

        got = apply_bg_targets(
            seed,
            t_op,
            t_char,
            g_count,
            g_char,
        )

        assert got == gt, (
            seed,
            gt,
            got,
        )

        print(
            "PASS",
            seed,
            "->",
            gt,
        )

    print(
        "ALL BG-NERD ROUND-TRIP TESTS PASSED"
    )


if __name__ == "__main__":
    _self_test()
