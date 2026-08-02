# Cross-entropy on the auxiliary digit count.
#
# The target is free -- MultiLoss already receives each label's length so the CTC
# loss can be computed, so no new annotation is required.
#
# Weight it small. Counting wheels is far easier than reading them, so this term
# converges within a few epochs and would otherwise start competing with the CTC
# objective for capacity while contributing no new information.

from __future__ import absolute_import, division, print_function

import paddle
from paddle import nn


class LengthLoss(nn.Layer):
    def __init__(self, max_length=25, **kwargs):
        super(LengthLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.max_length = max_length

    def forward(self, predicts, batch):
        # batch here is the raw length tensor handed over by MultiLoss
        target = batch if paddle.is_tensor(batch) else batch[0]
        target = paddle.clip(
            target.astype("int64").reshape([-1]), min=0, max=self.max_length
        )
        return {"loss": self.loss_func(predicts, target)}


class LengthLossMSE(nn.Layer):
    """Regression variant: (predicted_count - true_count)^2.

    predicts is the (B, max_length+1) pseudo-distribution produced by
    LengthBranchAttnReg (logits[b, c] = -(scalar[b]-c)^2 * temperature). The
    scalar is recovered as a soft-argmax (expected class under softmax of
    those logits) so no separate raw-scalar channel is needed.
    """

    def __init__(self, max_length=25, **kwargs):
        super(LengthLossMSE, self).__init__()
        self.max_length = max_length
        self.register_buffer(
            "classes", paddle.arange(0, max_length + 1, dtype="float32")
        )

    def forward(self, predicts, batch):
        target = batch if paddle.is_tensor(batch) else batch[0]
        target = paddle.clip(target.astype("float32").reshape([-1]), min=0, max=self.max_length)
        probs = paddle.nn.functional.softmax(predicts, axis=1)
        pred_count = paddle.sum(probs * self.classes.unsqueeze(0), axis=1)
        loss = paddle.mean((pred_count - target) ** 2)
        return {"loss": loss}


class LengthCountFusionLoss(nn.Layer):
    """Trains the learnable count_beta used by CTCCountAwareDecode.

    count_beta is NOT a parameter of this loss module (build_optimizer only
    tracks model.parameters(), never loss_class parameters -- a parameter
    placed here would silently never be updated). It lives on MultiHead
    (learn_count_beta=True) and is forwarded through predicts["count_beta"]
    so gradients flow into the real trainable parameter.

    For each training example we compute, for every candidate length n in
    [0, max_length], a DIFFERENTIABLE forward-algorithm log-probability
    logZ(n) = log sum over all CTC paths whose blank-collapsed output has
    exactly n symbols (restricted to blank + digit candidates, same
    restriction as CTCLengthGatedDecode/CTCCountAwareDecode, for
    tractability). This is the log-sum-exp analogue of the exact-decode DP
    in rec_postprocess.py (which uses max instead of log-sum-exp and is
    therefore not differentiable).

    Fusion score(n) = logZ(n) + count_beta * log_softmax(length_logits)[n].
    We turn scores over n into a soft distribution and minimise the expected
    squared error to the true length -- a small, scoped MWER-style
    objective whose only new trainable input is count_beta (CTC/length
    logits feeding into it are detached, so this pathway can only shape
    count_beta itself, not the encoder/CTC/length heads -- consistent with
    the detach-by-default policy used everywhere else in this project).
    """

    def __init__(
        self,
        max_length=25,
        character_dict_path=None,
        use_space_char=False,
        candidate_chars="0123456789",
        temperature=1.0,
        **kwargs,
    ):
        super(LengthCountFusionLoss, self).__init__()
        self.max_length = max_length
        self.temperature = temperature

        character_str = []
        if character_dict_path is None:
            character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
        else:
            with open(character_dict_path, "rb") as fin:
                for line in fin.readlines():
                    character_str.append(line.decode("utf-8").strip("\n").strip("\r\n"))
            if use_space_char:
                character_str.append(" ")
        dict_character = ["blank"] + character_str
        self.candidate_ids = [0] + [
            i for i, c in enumerate(dict_character) if c in candidate_chars
        ]
        self.register_buffer(
            "length_classes", paddle.arange(0, max_length + 1, dtype="float32")
        )

    def _logsumexp(self, a, b):
        m = paddle.maximum(a, b)
        m_safe = paddle.where(paddle.isinf(m), paddle.zeros_like(m), m)
        return m_safe + paddle.log(
            paddle.exp(a - m_safe) + paddle.exp(b - m_safe)
        )

    def _log_z(self, logp):
        """logp: (B, T, C) log-probabilities (already log_softmax'd).

        Returns (B, max_length+1) logZ(n) for n = 0..max_length, via a
        log-domain forward algorithm restricted to self.candidate_ids.
        State (k, r): k = symbols emitted so far (0..max_length), r = index
        into candidate_ids of the last raw (pre-collapse) symbol, 0 = blank.
        """
        B, T, _ = logp.shape
        K = self.max_length
        R = len(self.candidate_ids)
        NEG_INF = -1e9

        cand_logp = paddle.stack(
            [logp[:, :, c] for c in self.candidate_ids], axis=2
        )  # (B, T, R), column 0 = blank

        dp = paddle.full([B, K + 1, R], NEG_INF)
        dp[:, 0, 0] = 0.0

        for t in range(T):
            row = cand_logp[:, t, :]  # (B, R)
            new_dp = paddle.full([B, K + 1, R], NEG_INF)

            # blank: any (k, r) -> (k, 0), collapses all r via logsumexp
            collapsed = dp[:, :, 0]
            for r in range(1, R):
                collapsed = self._logsumexp(collapsed, dp[:, :, r])
            new_dp[:, :, 0] = collapsed + row[:, 0:1]

            for ci in range(1, R):
                sym_logp = row[:, ci : ci + 1]  # (B, 1)
                # new symbol: from any r != ci -> (k+1, ci)
                from_other = dp[:, :, 0]
                for r in range(1, R):
                    if r != ci:
                        from_other = self._logsumexp(from_other, dp[:, :, r])
                shifted_new = paddle.full([B, K + 1], NEG_INF)
                shifted_new[:, 1:] = from_other[:, :-1] + sym_logp
                new_dp[:, :, ci] = self._logsumexp(new_dp[:, :, ci], shifted_new)

                # repeat of same symbol arriving via blank (r=0): extends, k+1
                shifted_rep = paddle.full([B, K + 1], NEG_INF)
                shifted_rep[:, 1:] = dp[:, :-1, 0] + sym_logp
                new_dp[:, :, ci] = self._logsumexp(new_dp[:, :, ci], shifted_rep)

                # repeat of same symbol arriving via same non-blank (r=ci): merges, k unchanged
                merge = dp[:, :, ci] + sym_logp
                new_dp[:, :, ci] = self._logsumexp(new_dp[:, :, ci], merge)

            dp = new_dp

        log_z = dp[:, :, 0]
        for r in range(1, R):
            log_z = self._logsumexp(log_z, dp[:, :, r])
        return log_z  # (B, K+1)

    def forward(self, predicts, batch):
        ctc_logits = predicts["ctc"].detach()
        length_logits = predicts["length"].detach()
        count_beta = predicts["count_beta"]
        target = paddle.clip(
            batch.astype("float32").reshape([-1]), min=0, max=self.max_length
        )

        logp = paddle.nn.functional.log_softmax(ctc_logits, axis=2)
        log_z = self._log_z(logp)  # (B, K+1)
        log_p_len = paddle.nn.functional.log_softmax(length_logits, axis=1)

        score = log_z + count_beta * log_p_len  # (B, K+1)
        weights = paddle.nn.functional.softmax(score / self.temperature, axis=1)
        expected_n = paddle.sum(weights * self.length_classes.unsqueeze(0), axis=1)
        loss = paddle.mean((expected_n - target) ** 2)
        return {"loss": loss}
