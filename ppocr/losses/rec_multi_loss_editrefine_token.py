"""Multi-task loss wiring for direct token refinement."""
from __future__ import absolute_import, division, print_function

from .rec_multi_loss import MultiLoss
from .rec_edit_loss_token_refine import EditLossTokenRefine


class MultiLossEditRefineToken(MultiLoss):
    def __init__(self, **kwargs):
        self.weight_edit = kwargs.get("weight_edit", 1.0)
        super().__init__(**kwargs)
        self.edit_loss = EditLossTokenRefine(max_length=kwargs.get("length_max", 25))

    def forward(self, predicts, batch):
        total = super().forward(predicts, batch)
        result = self.edit_loss(predicts["edit"], (batch[1], batch[3]))
        edit_loss = result["loss"] * self.weight_edit
        total["EditLoss"] = edit_loss
        total["EditOpLoss"] = result["op_loss"] * self.weight_edit
        total["EditTokLoss"] = result["tok_loss"] * self.weight_edit
        total["EditActivationRate"] = result["activation_rate"]
        total["EditPredChangeRate"] = result["pred_change_rate"]
        total["EditPredReplaceRate"] = result["pred_replace_rate"]
        total["EditPredDeleteRate"] = result["pred_delete_rate"]
        total["EditPredInsertRate"] = result["pred_insert_rate"]
        total["loss"] = total["loss"] + edit_loss
        return total
