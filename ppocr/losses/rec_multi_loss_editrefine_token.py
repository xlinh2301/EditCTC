"""Multi-task loss wiring for direct token refinement."""
from __future__ import absolute_import, division, print_function

from .rec_multi_loss import MultiLoss
from .rec_edit_loss_token_refine import EditLossTokenRefine


class MultiLossEditRefineToken(MultiLoss):
    def __init__(self, **kwargs):
        self.weight_edit = kwargs.get("weight_edit", 1.0)
        self.weight_op = kwargs.get("weight_op", 0.0)
        self.weight_append = kwargs.get("weight_append", 0.0)
        super().__init__(**kwargs)
        self.edit_loss = EditLossTokenRefine(
            max_length=kwargs.get("length_max", 25),
            keep_loss_weight=kwargs.get("keep_loss_weight", 1.0),
            lambda_keep=kwargs.get("lambda_keep", kwargs.get("keep_loss_weight", 1.0)),
            use_decoupled_loss=kwargs.get("use_decoupled_loss", False),
            weight_gate=kwargs.get("weight_gate", 0.0),
            gate_pos_weight=kwargs.get("gate_pos_weight", 4.0),
            weight_op=self.weight_op,
            op_class_weights=kwargs.get("op_class_weights"),
            weight_append=self.weight_append,
            append_pos_weight=kwargs.get("append_pos_weight", 4.0),
            label_smoothing=kwargs.get("label_smoothing", 0.0),
            tail_focal_gamma=kwargs.get("tail_focal_gamma", 0.0),
            lambda_fix=kwargs.get("lambda_fix", 1.0),
            focal_gate_gamma=kwargs.get("focal_gate_gamma", 0.0),
            focal_gate_alpha_wrong=kwargs.get("focal_gate_alpha_wrong", 0.75),
            focal_gate_alpha_correct=kwargs.get("focal_gate_alpha_correct", 0.25),
            confidence_aware_keep=kwargs.get("confidence_aware_keep", False),
            decoupled_norm=kwargs.get("decoupled_norm", "valid"),
            gate_brier_weight=kwargs.get("gate_brier_weight", 0.0),
        )

    def forward(self, predicts, batch):
        total = super().forward(predicts, batch)
        result = self.edit_loss(predicts["edit"], (batch[1], batch[3]))
        edit_loss = result["loss"] * self.weight_edit
        total["EditLoss"] = edit_loss
        total["EditTokLoss"] = result["tok_loss"] * self.weight_edit
        total["EditFixLoss"] = result.get("loss_fix", edit_loss * 0.0)
        total["EditKeepLoss"] = result.get("loss_keep", edit_loss * 0.0)
        total["EditGateLoss"] = result.get("loss_gate", edit_loss * 0.0)
        total["EditGateBrierLoss"] = result.get("loss_brier", edit_loss * 0.0)
        total["EditOpLoss"] = result["op_loss"] * self.weight_op
        total["EditAppendLoss"] = result["append_loss"] * self.weight_append
        total["EditAppendTokLoss"] = result["append_tok_loss"] * self.weight_append
        
        # Diagnostic metrics
        total["WrongSeedRate"] = result.get("WrongSeedRate", edit_loss * 0.0)
        total["CorrectionRecall"] = result.get("CorrectionRecall", edit_loss * 0.0)
        total["CorrectionPrecision"] = result.get("CorrectionPrecision", edit_loss * 0.0)
        total["OverCorrectionRate"] = result.get("OverCorrectionRate", edit_loss * 0.0)
        total["CorrectSeedPreserveRate"] = result.get("CorrectSeedPreserveRate", edit_loss * 0.0)
        total["EditActivationRate"] = result["activation_rate"]
        total["EditPredChangeRate"] = result["pred_change_rate"]
        total["EditPredReplaceRate"] = result["pred_replace_rate"]
        total["EditPredDeleteRate"] = result["pred_delete_rate"]
        total["EditPredInsertRate"] = result["pred_insert_rate"]
        total["EditTargetDeleteRate"] = result["target_delete_rate"]
        total["EditTargetInsertRate"] = result["target_insert_rate"]
        total["EditAppendPredRate"] = result["append_pred_rate"]
        total["EditAppendTargetRate"] = result["append_target_rate"]
        
        total["loss"] = (
            total["loss"]
            + edit_loss
            + total["EditOpLoss"]
            + total["EditAppendLoss"]
            + total["EditAppendTokLoss"]
        )
        return total
