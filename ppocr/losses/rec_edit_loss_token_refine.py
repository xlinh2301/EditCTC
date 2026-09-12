"""Direct token denoising loss for a CTC-seeded editor.

This deliberately removes the imbalanced KEEP/EDIT operation classifier.  The
replacement head learns the target token at every aligned KEEP/REPLACE slot;
on a corrupted seed this is a denoising target, and at inference a separate
confidence/margin rule decides whether to replace the seed.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import nn

from .rec_edit_loss import IGNORE_INDEX, KEEP, REPLACE, levenshtein_ops


class EditLossTokenRefine(nn.Layer):
    def __init__(self, max_length=25, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.tok_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)

    def build_targets(self, seed_ids, seed_lens, gt_ids, gt_lens):
        bsz, width = seed_ids.shape
        tok = np.full((bsz, width), IGNORE_INDEX, dtype="int64")
        op = np.full((bsz, width), IGNORE_INDEX, dtype="int64")
        for b in range(bsz):
            n, g = int(seed_lens[b]), int(gt_lens[b])
            if n <= 0:
                continue
            ops, replacement = levenshtein_ops(
                seed_ids[b, :n].tolist(), gt_ids[b, :g].tolist()
            )
            for i, kind in enumerate(ops):
                if kind == KEEP:
                    op[b, i] = 0
                    tok[b, i] = int(seed_ids[b, i])
                elif kind == REPLACE:
                    op[b, i] = 1
                    tok[b, i] = int(replacement[i])
                # DELETE/INSERT have no one-to-one token slot and are ignored.
        return tok, op

    def forward(self, predicts, batch):
        tok_logits = predicts["tok_logits"]
        seed_ids = predicts["seed_ids"].numpy()
        seed_lens = predicts["seed_lens"].numpy()
        label_ctc, lengths = batch
        tok_t_np, op_t_np = self.build_targets(
            seed_ids,
            seed_lens,
            label_ctc.numpy().astype("int64"),
            lengths.numpy().astype("int64"),
        )
        valid = tok_t_np != IGNORE_INDEX
        if not valid.any():
            zero = tok_logits.sum() * 0.0
            return self._metrics(zero, zero, zero, zero)
        b, k, vocab = tok_logits.shape
        tok_t = paddle.to_tensor(tok_t_np, dtype="int64")
        mask = paddle.to_tensor(valid.astype("float32"))
        elem = self.tok_loss(
            tok_logits.reshape([b * k, vocab]), tok_t.reshape([b * k])
        ).reshape([b, k])
        loss = (elem * mask).sum() / paddle.maximum(
            mask.sum(), paddle.to_tensor(1.0, dtype="float32")
        )
        pred = np.argmax(tok_logits.numpy(), axis=2)
        pred_change = float(np.logical_and(valid, pred != seed_ids).sum()) / max(int(valid.sum()), 1)
        target_replace = float((op_t_np == REPLACE).sum()) / max(int(valid.sum()), 1)
        pred_replace = pred_change
        return self._metrics(
            loss,
            paddle.to_tensor(target_replace, dtype="float32"),
            paddle.to_tensor(pred_change, dtype="float32"),
            paddle.to_tensor(pred_replace, dtype="float32"),
        )

    @staticmethod
    def _metrics(loss, activation, pred_change, pred_replace):
        zero = loss * 0.0
        return {
            "loss": loss,
            "op_loss": zero,
            "tok_loss": loss,
            "activation_rate": activation,
            "pred_change_rate": pred_change,
            "pred_replace_rate": pred_replace,
            "pred_delete_rate": zero,
            "pred_insert_rate": zero,
        }
