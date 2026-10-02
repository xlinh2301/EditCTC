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

from .rec_edit_loss import (
    IGNORE_INDEX,
    KEEP,
    REPLACE,
    DELETE,
    INSERT_AFTER,
    levenshtein_ops,
)


class EditLossTokenRefine(nn.Layer):
    def __init__(self, max_length=25, keep_loss_weight=1.0, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.keep_loss_weight = float(keep_loss_weight)
        self.lambda_keep = float(kwargs.get("lambda_keep", self.keep_loss_weight))
        self.use_decoupled_loss = bool(kwargs.get("use_decoupled_loss", False))
        self.weight_gate = float(kwargs.get("weight_gate", 0.0))
        self.gate_pos_weight = float(kwargs.get("gate_pos_weight", 4.0))
        self.label_smoothing = float(kwargs.get("label_smoothing", 0.0))
        self.tok_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        
        # F3: full 4-way op supervision
        self.weight_op = float(kwargs.get("weight_op", 0.0))
        op_class_weights = kwargs.get("op_class_weights")
        self.op_loss_func = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=IGNORE_INDEX,
            weight=(
                paddle.to_tensor(op_class_weights, dtype="float32")
                if op_class_weights is not None
                else None
            ),
        )
        # F4: append gate (binary, class-balanced) + appended character
        self.weight_append = float(kwargs.get("weight_append", 0.0))
        pos_weight = kwargs.get("append_pos_weight", 4.0)
        self.append_loss_func = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=paddle.to_tensor([pos_weight], dtype="float32"),
        )
        self.change_loss_func = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=paddle.to_tensor([self.gate_pos_weight], dtype="float32"),
        )
        self.tail_focal_gamma = float(kwargs.get("tail_focal_gamma", 0.0))
        self.lambda_fix = float(kwargs.get("lambda_fix", 1.0))
        self.focal_gate_gamma = float(kwargs.get("focal_gate_gamma", 0.0))
        self.focal_gate_alpha_wrong = float(kwargs.get("focal_gate_alpha_wrong", 0.75))
        self.focal_gate_alpha_correct = float(kwargs.get("focal_gate_alpha_correct", 0.25))
        self.confidence_aware_keep = bool(kwargs.get("confidence_aware_keep", False))
        self.decoupled_norm = kwargs.get("decoupled_norm", "valid")
        self.keep_conf_alpha = float(kwargs.get("keep_conf_alpha", 0.0))
        self.keep_conf_step = bool(kwargs.get("keep_conf_step", False))
        self.gate_brier_weight = float(kwargs.get("gate_brier_weight", 0.0))

    def build_targets(self, seed_ids, seed_lens, gt_ids, gt_lens):
        """Back-compatible (tok, op) view of :meth:`build_targets_full`."""
        full = self.build_targets_full(seed_ids, seed_lens, gt_ids, gt_lens)
        return full["tok"], full["op"]

    def build_targets_full(self, seed_ids, seed_lens, gt_ids, gt_lens):
        bsz, width = seed_ids.shape
        tok = np.full((bsz, width), IGNORE_INDEX, dtype="int64")
        op = np.full((bsz, width), IGNORE_INDEX, dtype="int64")
        append_gate = np.zeros((bsz,), dtype="float32")
        append_tok = np.zeros((bsz,), dtype="int64")
        append_valid = np.zeros((bsz,), dtype="float32")
        for b in range(bsz):
            n, g = int(seed_lens[b]), int(gt_lens[b])
            if n <= 0 or g <= 0:
                continue
            seed = seed_ids[b, :n].tolist()
            gt = gt_ids[b, :g].tolist()
            ops, replacement = levenshtein_ops(seed, gt)
            for i, kind in enumerate(ops):
                op[b, i] = kind
                if kind == KEEP:
                    tok[b, i] = int(seed_ids[b, i])
                elif kind in (REPLACE, INSERT_AFTER):
                    tok[b, i] = int(replacement[i])
            append_valid[b] = 1.0
            if g > n and gt[:n] == seed:
                append_gate[b] = 1.0
                append_tok[b] = int(gt[n])
                op[b, n - 1] = KEEP
                tok[b, n - 1] = int(seed_ids[b, n - 1])
        return {
            "tok": tok,
            "op": op,
            "append_gate": append_gate,
            "append_tok": append_tok,
            "append_valid": append_valid,
        }

    def forward(self, predicts, batch):
        tok_logits = predicts["tok_logits"]
        seed_ids = predicts["seed_ids"].numpy()
        seed_lens = predicts["seed_lens"].numpy()
        label_ctc, lengths = batch
        targets = self.build_targets_full(
            seed_ids,
            seed_lens,
            label_ctc.numpy().astype("int64"),
            lengths.numpy().astype("int64"),
        )
        tok_t_np, op_t_np = targets["tok"], targets["op"]
        valid = tok_t_np != IGNORE_INDEX
        if not valid.any():
            zero = tok_logits.sum() * 0.0
            return self._metrics(zero, zero, zero, zero)
        b, k, vocab = tok_logits.shape
        tok_t = paddle.to_tensor(tok_t_np, dtype="int64")
        
        elem = self.tok_loss(
            tok_logits.reshape([b * k, vocab]), tok_t.reshape([b * k])
        ).reshape([b, k])
        if self.label_smoothing > 0.0:
            log_probs = paddle.nn.functional.log_softmax(tok_logits, axis=-1)
            smooth_loss = -log_probs.mean(axis=-1)
            elem = (1.0 - self.label_smoothing) * elem + self.label_smoothing * smooth_loss

        # Masking & Sets: Wrong seeds vs Correct seeds
        # A seed position is "wrong" iff the optimal op is NOT KEEP.
        # INSERT_AFTER means the seed token itself is correct but a token
        # must be inserted right after it — it should NOT be counted as wrong.
        # The previous condition `seed_ids != tok_t_np` was incorrect because
        # tok_t_np at INSERT_AFTER positions holds the inserted token (≠ seed),
        # causing INSERT_AFTER positions to be mis-labelled as wrong seeds.
        wrong_np = np.logical_and(valid, op_t_np != KEEP)
        correct_np = np.logical_and(valid, np.logical_not(wrong_np))
        
        wrong_mask = paddle.to_tensor(wrong_np.astype("float32"))
        correct_mask = paddle.to_tensor(correct_np.astype("float32"))
        valid_mask = paddle.to_tensor(valid.astype("float32"))

        # EXP-5: Confidence-aware KEEP weighting
        if self.confidence_aware_keep and "seed_margin" in predicts:
            conf_ctc = predicts["seed_margin"]  # margin is in [0, 1]
            if self.keep_conf_step:
                w_keep = paddle.where(conf_ctc >= 0.90, paddle.full_like(correct_mask, 2.0), paddle.ones_like(correct_mask))
            else:
                w_keep = 1.0 + self.keep_conf_alpha * conf_ctc
            correct_mask = correct_mask * w_keep

        num_wrong = float(wrong_np.sum())
        num_correct = float(correct_np.sum())
        num_total = max(float(valid.sum()), 1.0)

        if self.decoupled_norm == "valid":
            norm_denom = paddle.maximum(valid_mask.sum(), paddle.to_tensor(1.0, dtype="float32"))
            loss_fix = (elem * wrong_mask).sum() / norm_denom
            loss_keep = (elem * correct_mask).sum() / norm_denom
        else:
            loss_fix = (elem * wrong_mask).sum() / paddle.maximum(
                wrong_mask.sum(), paddle.to_tensor(1.0, dtype="float32")
            )
            loss_keep = (elem * correct_mask).sum() / paddle.maximum(
                correct_mask.sum(), paddle.to_tensor(1.0, dtype="float32")
            )

        loss_gate = tok_logits.sum() * 0.0
        if self.weight_gate > 0.0:
            if "change_logits" in predicts:
                # ARCH-1: Explicit Change Head
                change_logits = predicts["change_logits"]
                bce_elem = self.change_loss_func(change_logits, wrong_mask)
                q_i = paddle.nn.functional.sigmoid(change_logits)
                z_i = wrong_mask
                if self.gate_brier_weight > 0.0:
                    brier_elem = paddle.square(q_i - z_i)
                    bce_elem = bce_elem + self.gate_brier_weight * brier_elem
                loss_gate = (bce_elem * valid_mask).sum() / paddle.maximum(
                    valid_mask.sum(), paddle.to_tensor(1.0, dtype="float32")
                )
            else:
                # Gating probability: q_i = 1 - P_edit(s_i)
                probs = paddle.nn.functional.softmax(tok_logits, axis=-1)
                seed_clip = paddle.clip(paddle.to_tensor(seed_ids, dtype="int64"), 0, vocab - 1)
                p_seed = paddle.take_along_axis(probs, seed_clip.unsqueeze(-1), axis=-1).squeeze(-1)
                q_i = paddle.clip(1.0 - p_seed, 1e-6, 1.0 - 1e-6)
                z_i = wrong_mask
                
                if self.focal_gate_gamma > 0.0:
                    # EXP-4: Focal Loss for KEEP/CHANGE
                    # p_t: probability of true class (q_i for z=1, 1-q_i for z=0)
                    p_t = z_i * q_i + (1.0 - z_i) * (1.0 - q_i)
                    alpha_t = z_i * self.focal_gate_alpha_wrong + (1.0 - z_i) * self.focal_gate_alpha_correct
                    focal_weight = alpha_t * paddle.pow(1.0 - p_t, self.focal_gate_gamma)
                    bce_elem = -focal_weight * paddle.log(p_t)
                else:
                    bce_elem = -(z_i * paddle.log(q_i) + (1.0 - z_i) * paddle.log(1.0 - q_i))

                if self.gate_brier_weight > 0.0:
                    # Brier score regularization: (p_change - y_change)^2 = (q_i - z_i)^2
                    brier_elem = paddle.square(q_i - z_i)
                    bce_elem = bce_elem + self.gate_brier_weight * brier_elem
                    
                loss_gate = (bce_elem * valid_mask).sum() / paddle.maximum(
                    valid_mask.sum(), paddle.to_tensor(1.0, dtype="float32")
                )

        if self.use_decoupled_loss:
            # EXP-1/EXP-2/EXP-8: lambda_fix * L_fix + lambda_keep * L_keep + weight_gate * L_gate
            tok_loss = self.lambda_fix * loss_fix + self.lambda_keep * loss_keep + self.weight_gate * loss_gate
        else:
            # EXP-0: Baseline weighted CE with gate_pos_weight
            combined_mask = wrong_mask * self.gate_pos_weight + correct_mask * self.keep_loss_weight
            tok_loss = (elem * combined_mask).sum() / paddle.maximum(
                combined_mask.sum(), paddle.to_tensor(1.0, dtype="float32")
            )
        if self.weight_gate > 0.0 and not self.use_decoupled_loss:
            tok_loss = tok_loss + self.weight_gate * loss_gate

        # Raw unweighted CE (over valid positions) for clean logging
        raw_tok_loss = (elem * valid_mask).sum() / paddle.maximum(
            valid_mask.sum(), paddle.to_tensor(1.0, dtype="float32")
        )

        # Diagnostic Metrics Calculation
        pred_tok = np.argmax(tok_logits.numpy(), axis=2)
        pred_changed = np.logical_and(valid, pred_tok != seed_ids)
        num_changed = float(pred_changed.sum())

        # repaired_correct: only count REPLACE positions where pred matches GT.
        # DELETE/INSERT_AFTER positions are structural edits the token head can't
        # directly "fix" (the target tok is an insert/delete marker, not the seed).
        replace_positions = np.logical_and(valid, op_t_np == REPLACE)
        repaired_correct = np.logical_and(replace_positions, pred_tok == tok_t_np)
        over_corrected = np.logical_and(correct_np, pred_tok != seed_ids)
        preserved_correct = np.logical_and(correct_np, pred_tok == seed_ids)
        num_wrong_replace = max(float(replace_positions.sum()), 1.0)

        wrong_seed_rate = num_wrong / num_total
        correction_recall = float(repaired_correct.sum()) / num_wrong_replace
        correction_prec = float(repaired_correct.sum()) / max(num_changed, 1.0)
        over_corr_rate = float(over_corrected.sum()) / max(num_correct, 1.0)
        preserve_rate = float(preserved_correct.sum()) / max(num_correct, 1.0)
        pred_change_rate = num_changed / num_total

        # F3: 4-way op cross entropy
        op_loss = tok_loss * 0.0
        op_logits = predicts.get("op_logits")
        op_valid = op_t_np != IGNORE_INDEX
        if op_logits is not None and self.weight_op > 0.0 and bool(op_valid.any()):
            op_t = paddle.to_tensor(op_t_np, dtype="int64")
            op_loss = self.op_loss_func(
                op_logits.reshape([b * k, op_logits.shape[-1]]), op_t.reshape([b * k])
            )

        # F4: append gate + appended character
        append_loss = tok_loss * 0.0
        append_tok_loss = tok_loss * 0.0
        append_pred_rate = tok_loss * 0.0
        if "append_logits" in predicts and self.weight_append > 0.0:
            gate_t = paddle.to_tensor(targets["append_gate"], dtype="float32")
            valid_t = paddle.to_tensor(targets["append_valid"], dtype="float32")
            gate_elem = self.append_loss_func(
                predicts["append_logits"].reshape([-1]), gate_t
            )
            append_loss = (gate_elem * valid_t).sum() / paddle.maximum(
                valid_t.sum(), paddle.to_tensor(1.0, dtype="float32")
            )
            gate_pred = (paddle.nn.functional.sigmoid(
                predicts["append_logits"].reshape([-1])) >= 0.5).astype("float32")
            append_pred_rate = (gate_pred * valid_t).sum() / paddle.maximum(
                valid_t.sum(), paddle.to_tensor(1.0, dtype="float32")
            )
            append_target_rate = (gate_t * valid_t).sum() / paddle.maximum(
                valid_t.sum(), paddle.to_tensor(1.0, dtype="float32")
            )
            if "append_tok_logits" in predicts:
                tok_t_append = np.where(
                    targets["append_gate"] > 0.5, targets["append_tok"], IGNORE_INDEX
                )
                if bool((tok_t_append != IGNORE_INDEX).any()):
                    elem_ap = self.tok_loss(
                        predicts["append_tok_logits"],
                        paddle.to_tensor(tok_t_append, dtype="int64"),
                    )
                    tok_valid = paddle.to_tensor((tok_t_append != IGNORE_INDEX).astype("float32"))
                    append_tok_loss = (elem_ap * tok_valid).sum() / paddle.maximum(
                        tok_valid.sum(), paddle.to_tensor(1.0, dtype="float32")
                    )
        else:
            append_target_rate = tok_loss * 0.0

        op_pred = np.argmax(op_logits.numpy(), axis=2) if op_logits is not None else None
        if op_pred is not None:
            pred_delete = float(((op_pred == DELETE) & op_valid).sum()) / max(int(op_valid.sum()), 1)
            pred_insert = float(((op_pred == INSERT_AFTER) & op_valid).sum()) / max(int(op_valid.sum()), 1)
        else:
            pred_delete = pred_insert = 0.0

        loss_brier = tok_logits.sum() * 0.0
        if self.weight_gate > 0.0:
            loss_brier = (paddle.square(q_i - z_i) * valid_mask).sum() / paddle.maximum(
                valid_mask.sum(), paddle.to_tensor(1.0, dtype="float32")
            )

        return self._metrics(
            tok_loss,
            paddle.to_tensor(wrong_seed_rate, dtype="float32"),
            paddle.to_tensor(pred_change_rate, dtype="float32"),
            paddle.to_tensor(pred_change_rate, dtype="float32"),
            op_loss=op_loss,
            append_loss=append_loss,
            append_tok_loss=append_tok_loss,
            append_pred_rate=append_pred_rate,
            append_target_rate=append_target_rate,
            target_delete=paddle.to_tensor(float((op_t_np == DELETE).sum()) / max(int(op_valid.sum()), 1), dtype="float32"),
            target_insert=paddle.to_tensor(float((op_t_np == INSERT_AFTER).sum()) / max(int(op_valid.sum()), 1), dtype="float32"),
            pred_delete=paddle.to_tensor(pred_delete, dtype="float32"),
            pred_insert=paddle.to_tensor(pred_insert, dtype="float32"),
            loss_fix=loss_fix,
            loss_keep=loss_keep,
            loss_gate=loss_gate,
            loss_brier=loss_brier,
            raw_tok_loss=raw_tok_loss,
            wrong_seed_rate=paddle.to_tensor(wrong_seed_rate, dtype="float32"),
            correction_recall=paddle.to_tensor(correction_recall, dtype="float32"),
            correction_precision=paddle.to_tensor(correction_prec, dtype="float32"),
            over_correction_rate=paddle.to_tensor(over_corr_rate, dtype="float32"),
            preserve_rate=paddle.to_tensor(preserve_rate, dtype="float32"),
        )

    @staticmethod
    def _metrics(loss, activation, pred_change, pred_replace, **extra):
        zero = loss * 0.0
        out = {
            "loss": loss,
            "op_loss": extra.get("op_loss", zero),
            "tok_loss": extra.get("raw_tok_loss", loss),
            "loss_fix": extra.get("loss_fix", zero),
            "loss_keep": extra.get("loss_keep", zero),
            "loss_gate": extra.get("loss_gate", zero),
            "loss_brier": extra.get("loss_brier", zero),
            "EditGateBrierLoss": extra.get("loss_brier", zero),
            "WrongSeedRate": extra.get("wrong_seed_rate", zero),
            "CorrectionRecall": extra.get("correction_recall", zero),
            "CorrectionPrecision": extra.get("correction_precision", zero),
            "OverCorrectionRate": extra.get("over_correction_rate", zero),
            "CorrectSeedPreserveRate": extra.get("preserve_rate", zero),
            "activation_rate": activation,
            "EditActivationRate": activation,
            "pred_change_rate": pred_change,
            "pred_replace_rate": pred_replace,
            "pred_delete_rate": extra.get("pred_delete", zero),
            "pred_insert_rate": extra.get("pred_insert", zero),
            "target_delete_rate": extra.get("target_delete", zero),
            "target_insert_rate": extra.get("target_insert", zero),
            "append_loss": extra.get("append_loss", zero),
            "append_tok_loss": extra.get("append_tok_loss", zero),
            "append_pred_rate": extra.get("append_pred_rate", zero),
            "append_target_rate": extra.get("append_target_rate", zero),
        }
        return out
