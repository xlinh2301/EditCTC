# MultiLoss + uncertainty-gated CTC-Seeded Edit Refinement Decoder term.
# Exp1 of the uncertainty-gating investigation -- see
# experiments/editrefine_probe_v6_uncertainty/SPEC.md.
#
# Subclasses MultiLoss exactly like MultiLossEditRefine (reused unchanged
# for CTCLoss/NRTRLoss/LengthLoss dispatch); the only difference is which
# EditLoss implementation it wires up, and that it surfaces
# `EditActivationRate` as its own logged key so it appears in the training
# log next to CTCLoss/EditLoss every step without needing any change to
# tools/train.py's logging (any key in the returned dict gets printed the
# same way the existing "EditLoss" key already does).

from __future__ import absolute_import, division, print_function

from .rec_multi_loss import MultiLoss
from .rec_edit_loss_uncertainty import EditLossUncertainty


class MultiLossEditRefineUncertainty(MultiLoss):
    def __init__(self, **kwargs):
        self.weight_edit = kwargs.get("weight_edit", 0.15)
        super().__init__(**kwargs)
        self.edit_loss = EditLossUncertainty(
            max_length=kwargs.get("length_max", 25),
            gate_percentile=kwargs.get("gate_percentile", 0.10),
            gate_mode=kwargs.get("gate_mode", "margin"),
        )

    def forward(self, predicts, batch):
        total = super().forward(predicts, batch)

        # batch = [image, label_ctc, label_gtc, length, valid_ratio]
        edit_result = self.edit_loss(predicts["edit"], (batch[1], batch[3]))
        edit_loss = edit_result["loss"] * self.weight_edit

        total["EditLoss"] = edit_loss
        # Not multiplied by weight_edit -- this is a diagnostic rate
        # (fraction of seed positions selected for loss), not a loss term,
        # and must not be added into total["loss"].
        total["EditActivationRate"] = edit_result["activation_rate"]
        total["loss"] = total["loss"] + edit_loss
        return total
