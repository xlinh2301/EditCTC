"""Balanced binary-gate plus replacement-token loss for EditCTC.

The original four-way operation CE is dominated by KEEP.  This loss trains a
binary EDIT/KEEP gate and a replacement classifier only on aligned
substitution positions.  Delete/insert alignments are ignored until the
substitution branch proves useful.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import nn

from .rec_edit_loss import IGNORE_INDEX, KEEP, REPLACE, levenshtein_ops


class EditLossFactorized(nn.Layer):
    def __init__(self, max_length=25, gate_pos_weight=4.0, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.gate_pos_weight = gate_pos_weight
        self.gate_loss = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=IGNORE_INDEX,
            weight=paddle.to_tensor([1.0, float(gate_pos_weight)], dtype="float32"),
        )
        self.tok_loss = nn.CrossEntropyLoss(
            reduction="none", ignore_index=IGNORE_INDEX
        )

    def build_targets(self, seed_ids, seed_lens, gt_ids, gt_lens):
        bsz, width = seed_ids.shape
        gate = np.full((bsz, width), IGNORE_INDEX, dtype="int64")
        tok = np.full((bsz, width), IGNORE_INDEX, dtype="int64")
        for b in range(bsz):
            n, g = int(seed_lens[b]), int(gt_lens[b])
            if n <= 0:
                continue
            ops, replacement = levenshtein_ops(
                seed_ids[b, :n].tolist(), gt_ids[b, :g].tolist()
            )
            for i, op in enumerate(ops):
                if op == KEEP:
                    gate[b, i] = 0
                elif op == REPLACE:
                    gate[b, i] = 1
                    tok[b, i] = replacement[i]
                # DELETE/INSERT are unsupported in this first factorized
                # experiment and remain ignored rather than mislabeled KEEP.
        return gate, tok

    def forward(self, predicts, batch):
        gate_logits = predicts["op_logits"]
        tok_logits = predicts["tok_logits"]
        seed_ids = predicts["seed_ids"].numpy()
        seed_lens = predicts["seed_lens"].numpy()
        label_ctc, lengths = batch
        gate_t_np, tok_t_np = self.build_targets(
            seed_ids,
            seed_lens,
            label_ctc.numpy().astype("int64"),
            lengths.numpy().astype("int64"),
        )
        valid = gate_t_np != IGNORE_INDEX
        if not valid.any():
            zero = (gate_logits.sum() + tok_logits.sum()) * 0.0
            return self._metrics(zero, zero, zero, zero, zero, zero)

        gate_t = paddle.to_tensor(gate_t_np, dtype="int64")
        tok_t = paddle.to_tensor(tok_t_np, dtype="int64")
        b, k, _ = gate_logits.shape
        gate_elem = self.gate_loss(
            gate_logits.reshape([b * k, 2]), gate_t.reshape([b * k])
        ).reshape([b, k])
        gate_mask = paddle.to_tensor(valid.astype("float32"))
        denom = gate_mask.sum()
        gate_loss = (gate_elem * gate_mask).sum() / denom

        replace_valid = tok_t_np != IGNORE_INDEX
        replace_mask = paddle.to_tensor(replace_valid.astype("float32"))
        tok_elem = self.tok_loss(
            tok_logits.reshape([b * k, tok_logits.shape[-1]]), tok_t.reshape([b * k])
        ).reshape([b, k])
        replace_denom = replace_mask.sum()
        tok_loss = (tok_elem * replace_mask).sum() / paddle.maximum(
            replace_denom, paddle.to_tensor(1.0, dtype="float32")
        )
        loss = gate_loss + tok_loss

        pred_gate = np.argmax(gate_logits.numpy(), axis=2)
        valid_count = max(int(valid.sum()), 1)
        pred_change = float(np.logical_and(valid, pred_gate == 1).sum()) / valid_count
        target_change = float(replace_valid.sum()) / valid_count
        return self._metrics(
            loss,
            gate_loss,
            tok_loss,
            paddle.to_tensor(target_change, dtype="float32"),
            paddle.to_tensor(pred_change, dtype="float32"),
            paddle.to_tensor(pred_change, dtype="float32"),
        )

    @staticmethod
    def _metrics(loss, gate_loss, tok_loss, activation, pred_change, pred_replace):
        zero = loss * 0.0
        return {
            "loss": loss,
            "op_loss": gate_loss,
            "tok_loss": tok_loss,
            "activation_rate": activation,
            "pred_change_rate": pred_change,
            "pred_replace_rate": pred_replace,
            "pred_delete_rate": zero,
            "pred_insert_rate": zero,
        }
