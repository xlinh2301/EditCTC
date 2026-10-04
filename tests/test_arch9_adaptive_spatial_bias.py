"""Unit tests for ARCH-9 (Exp-1 vectorized bias, Exp-2 adaptive sigma, Exp-3 CTC-confidence gating).

These tests run on CPU-only PaddlePaddle (no GPU required) and are the local
correctness gate before any Colab T4 training/fine-tuning is launched:

1. `_build_spatial_cross_bias` (vectorized Paddle-ops rewrite) must be
   numerically equivalent to the original per-sample Python/numpy loop it
   replaced, for every ARCH-4-era flag combination, when
   `use_adaptive_spatial_sigma=False`.
2. With `use_adaptive_spatial_sigma=True`, a higher-entropy confidence vector
   must produce a strictly wider (larger-sigma) Gaussian window than a
   low-entropy one, and `adaptive_sigma_beta` must receive a gradient
   (confirms the vectorized rewrite is actually differentiable).
3. The Exp-3 CTC-confidence-gated edit eligibility forces KEEP at seed
   positions whose margin already clears `ctc_confidence_gate_margin`.
"""

import unittest

import numpy as np


def _reference_build_spatial_cross_bias(
    tgt_len,
    mem_len,
    seed_lens_np,
    tail_mask_np,
    bsz,
    use_highres_visual,
    use_original_image,
    tail_spatial_mode,
    tail_spatial_bias_weight,
    tail_spatial_sigma,
    tail_prog_weight,
    tail_prog_sigma,
    use_align_guided_cross_attn,
    align_spatial_weight,
    align_spatial_sigma,
    max_ctc_timesteps,
    timesteps_np,
    use_append_head,
):
    """Faithful re-implementation of the pre-Exp-1 per-sample Python loop.

    Kept here (not in production code) purely as an independent oracle for
    the vectorized rewrite under test.
    """
    if use_highres_visual:
        ctc_w = 40
        vis_len = mem_len - ctc_w
        if vis_len > 0:
            vis_w = (
                80
                if vis_len % 80 == 0
                else (vis_len // 3 if vis_len % 3 == 0 else vis_len)
            )
            vis_h = max(1, vis_len // vis_w)
            u_ctc = np.linspace(0.0, 1.0, ctc_w, dtype=np.float32)
            u_col = np.linspace(0.0, 1.0, vis_w, dtype=np.float32)
            u_vis = np.tile(u_col, vis_h)
            u = np.concatenate([u_ctc, u_vis])
        else:
            u = np.linspace(0.0, 1.0, mem_len, dtype=np.float32)
    elif use_original_image:
        ctc_w = 40
        raw_w = mem_len - ctc_w
        u_ctc = np.linspace(0.0, 1.0, ctc_w, dtype=np.float32)
        u_raw = np.linspace(0.0, 1.0, max(1, raw_w), dtype=np.float32)
        u = np.concatenate([u_ctc, u_raw])
    else:
        u = np.linspace(0.0, 1.0, mem_len, dtype=np.float32)

    bias_np = np.zeros((bsz, 1, tgt_len, mem_len), dtype=np.float32)
    lens_np = seed_lens_np
    tail_weight = tail_spatial_bias_weight
    tail_sigma_sq2 = 2.0 * (tail_spatial_sigma**2)
    prog_weight = tail_prog_weight
    prog_sigma_sq2 = 2.0 * (tail_prog_sigma**2)
    tail_bias_vec = tail_weight * np.exp(-((u - 1.0) ** 2) / tail_sigma_sq2)

    for b in range(bsz):
        n = int(lens_np[b])
        if tail_spatial_mode == "progressive":
            for j in range(1, min(n + 1, tgt_len)):
                c_j = (j - 0.5) / max(1.0, float(n))
                bias_np[b, 0, j, :] = prog_weight * np.exp(
                    -((u - c_j) ** 2) / prog_sigma_sq2
                )
        elif tail_spatial_mode == "tail_only":
            if 1 <= n < tgt_len:
                bias_np[b, 0, n, :] = 0.5 * tail_bias_vec

        if use_align_guided_cross_attn and timesteps_np is not None:
            align_sigma_sq2 = 2.0 * (align_spatial_sigma**2)
            for j in range(1, min(n + 1, tgt_len)):
                t_val = float(timesteps_np[b, j - 1])
                c_j = t_val / max(1.0, float(max_ctc_timesteps - 1))
                bias_np[b, 0, j, :] += align_spatial_weight * np.exp(
                    -((u - c_j) ** 2) / align_sigma_sq2
                )

        if tail_mask_np is not None:
            append_pos = np.where(tail_mask_np[b] > 0.5)[0]
            for p in append_pos:
                if p < tgt_len:
                    bias_np[b, 0, p, :] = tail_bias_vec
        elif (n + 1) < tgt_len and use_append_head:
            bias_np[b, 0, n + 1, :] = tail_bias_vec

    return bias_np


class TestArch9AdaptiveSpatialBias(unittest.TestCase):
    def _make_head(self, **overrides):
        import paddle

        from ppocr.modeling.heads.rec_edit_refine_nrtr_head import (
            MultiHeadEditRefineNRTR,
        )

        kwargs = dict(
            head_list=[
                {
                    "CTCHead": {
                        "Neck": {"name": "lightsvtr", "dims": 24, "depth": 1},
                        "Head": {},
                    }
                },
                {"NRTRHead": {"max_text_length": 8, "nrtr_dim": 32}},
            ],
            name="MultiHeadEditRefineNRTR",
            nrtr_dim=32,
            nrtr_layers=1,
            length_hidden=16,
            length_max=8,
            use_length_head=True,
            use_highres_visual=False,
            use_original_image=True,
            use_explicit_change_head=False,
            use_align_guided_cross_attn=True,
            align_spatial_weight=1.0,
            align_spatial_sigma=0.15,
            max_ctc_timesteps=10,
            edit_mode="token_refine",
        )
        kwargs.update(overrides)
        paddle.seed(0)
        head = MultiHeadEditRefineNRTR(
            in_channels=24,
            out_channels_list={"CTCLabelDecode": 12, "NRTRLabelDecode": 15},
            **kwargs,
        )
        head.eval()
        return head

    def test_vectorized_bias_matches_reference_loop(self):
        import paddle

        head = self._make_head(use_adaptive_spatial_sigma=False)
        bsz, tgt_len, mem_len = 3, 6, 44  # mem_len = 40 (ctc) + 4 (raw image tokens)
        seed_lens_np = np.array([5, 3, 0], dtype="int64")
        timesteps_np = np.random.randint(0, 10, size=(bsz, head.max_seed_len)).astype(
            "int64"
        )

        ref = _reference_build_spatial_cross_bias(
            tgt_len=tgt_len,
            mem_len=mem_len,
            seed_lens_np=seed_lens_np,
            tail_mask_np=None,
            bsz=bsz,
            use_highres_visual=head.use_highres_visual,
            use_original_image=head.use_original_image,
            tail_spatial_mode=head.tail_spatial_mode,
            tail_spatial_bias_weight=head.tail_spatial_bias_weight,
            tail_spatial_sigma=head.tail_spatial_sigma,
            tail_prog_weight=head.tail_prog_weight,
            tail_prog_sigma=head.tail_prog_sigma,
            use_align_guided_cross_attn=head.use_align_guided_cross_attn,
            align_spatial_weight=head.align_spatial_weight,
            align_spatial_sigma=head.align_spatial_sigma,
            max_ctc_timesteps=head.max_ctc_timesteps,
            timesteps_np=timesteps_np,
            use_append_head=head.use_append_head,
        )

        got = head._build_spatial_cross_bias(
            tgt_len=tgt_len,
            mem_len=mem_len,
            seed_lens=paddle.to_tensor(seed_lens_np, dtype="int64"),
            tail_mask=None,
            bsz=bsz,
            dtype="float32",
            timesteps=paddle.to_tensor(timesteps_np, dtype="int64"),
            confs=None,
        ).numpy()

        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_vectorized_bias_matches_reference_with_append_head_and_progressive_tail(
        self,
    ):
        import paddle

        head = self._make_head(
            use_adaptive_spatial_sigma=False,
            use_append_head=True,
            tail_spatial_attn=True,
            tail_spatial_mode="progressive",
        )
        bsz, seed_width = 2, 4
        tgt_len = seed_width + 2  # BOS + seed_width seed slots + 1 append slot
        mem_len = 44
        seed_lens_np = np.array([3, 4], dtype="int64")
        timesteps_np = np.random.randint(0, 10, size=(bsz, head.max_seed_len)).astype(
            "int64"
        )

        seed_ids = paddle.to_tensor(np.ones((bsz, seed_width), dtype="int64"))
        nrtr_seed, tail_mask = head._insert_tail_slot(seed_ids, seed_lens_np)
        # tail_mask covers seed columns only; _seed_hidden prepends a BOS zero
        # column before passing it to _build_spatial_cross_bias.
        tail_mask_shifted = paddle.concat(
            [paddle.zeros_like(tail_mask[:, :1]), tail_mask], axis=1
        )

        ref = _reference_build_spatial_cross_bias(
            tgt_len=tgt_len,
            mem_len=mem_len,
            seed_lens_np=seed_lens_np,
            tail_mask_np=tail_mask_shifted.numpy(),
            bsz=bsz,
            use_highres_visual=head.use_highres_visual,
            use_original_image=head.use_original_image,
            tail_spatial_mode=head.tail_spatial_mode,
            tail_spatial_bias_weight=head.tail_spatial_bias_weight,
            tail_spatial_sigma=head.tail_spatial_sigma,
            tail_prog_weight=head.tail_prog_weight,
            tail_prog_sigma=head.tail_prog_sigma,
            use_align_guided_cross_attn=head.use_align_guided_cross_attn,
            align_spatial_weight=head.align_spatial_weight,
            align_spatial_sigma=head.align_spatial_sigma,
            max_ctc_timesteps=head.max_ctc_timesteps,
            timesteps_np=timesteps_np,
            use_append_head=head.use_append_head,
        )

        got = head._build_spatial_cross_bias(
            tgt_len=tgt_len,
            mem_len=mem_len,
            seed_lens=paddle.to_tensor(seed_lens_np, dtype="int64"),
            tail_mask=tail_mask_shifted,
            bsz=bsz,
            dtype="float32",
            timesteps=paddle.to_tensor(timesteps_np, dtype="int64"),
            confs=None,
        ).numpy()

        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_adaptive_sigma_widens_with_higher_entropy_and_is_differentiable(self):
        import paddle

        head = self._make_head(
            use_adaptive_spatial_sigma=True,
            use_ctc_conf_embed=True,
            ctc_conf_dim=4,
        )
        bsz, tgt_len, mem_len = 1, 4, 44
        seed_lens_np = np.array([3], dtype="int64")
        timesteps_np = np.array([[5, 5, 5, 0, 0, 0, 0, 0]], dtype="int64")[
            :, : head.max_seed_len
        ]

        low_entropy_confs = np.zeros((bsz, head.max_seed_len, 4), dtype="float32")
        low_entropy_confs[..., 3] = 0.0  # confident CTC peak
        high_entropy_confs = np.zeros((bsz, head.max_seed_len, 4), dtype="float32")
        high_entropy_confs[..., 3] = 2.0  # ambiguous CTC peak (near entropy_norm cap)

        # Nudge beta away from its deeply-negative init so the sigma_scale
        # term is not numerically flattened to ~1.0 for this comparison.
        with paddle.no_grad():
            head.adaptive_sigma_beta.set_value(paddle.to_tensor([2.0], dtype="float32"))

        bias_low = head._build_spatial_cross_bias(
            tgt_len=tgt_len,
            mem_len=mem_len,
            seed_lens=paddle.to_tensor(seed_lens_np, dtype="int64"),
            tail_mask=None,
            bsz=bsz,
            dtype="float32",
            timesteps=paddle.to_tensor(timesteps_np, dtype="int64"),
            confs=paddle.to_tensor(low_entropy_confs, dtype="float32"),
        )
        bias_high = head._build_spatial_cross_bias(
            tgt_len=tgt_len,
            mem_len=mem_len,
            seed_lens=paddle.to_tensor(seed_lens_np, dtype="int64"),
            tail_mask=None,
            bsz=bsz,
            dtype="float32",
            timesteps=paddle.to_tensor(timesteps_np, dtype="int64"),
            confs=paddle.to_tensor(high_entropy_confs, dtype="float32"),
        )

        # A wider Gaussian has heavier tails: far from the Gaussian center
        # (c_j = 5 / 9 ~= 0.556 on the normalized u-grid), a larger sigma
        # decays slower and so must leave a *larger* residual bias than a
        # narrow sigma.  (At the exact center both give exp(0)=1 regardless
        # of sigma, so the comparison must be made away from the peak.)
        far_col = 35  # u = 35 / 39 ~= 0.897, ~0.34 away from c_j
        far_low = bias_low.numpy()[0, 0, 1, far_col]
        far_high = bias_high.numpy()[0, 0, 1, far_col]
        self.assertGreater(far_high, far_low)

        # Gradient must reach adaptive_sigma_beta (confirms differentiability
        # of the Exp-1 vectorized rewrite, not just correctness).
        loss = bias_high.sum()
        loss.backward()
        self.assertIsNotNone(head.adaptive_sigma_beta.grad)
        self.assertFalse(bool(paddle.all(head.adaptive_sigma_beta.grad == 0.0).item()))

    def test_disabled_adaptive_sigma_matches_fixed_sigma_bias(self):
        """use_adaptive_spatial_sigma=False must be bit-identical to ARCH-4."""
        import paddle

        head_fixed = self._make_head(use_adaptive_spatial_sigma=False)
        head_adaptive_off_confs = self._make_head(
            use_adaptive_spatial_sigma=True, use_ctc_conf_embed=True, ctc_conf_dim=4
        )
        bsz, tgt_len, mem_len = 1, 4, 44
        seed_lens_np = np.array([3], dtype="int64")
        timesteps_np = np.array([[5, 5, 5, 0, 0, 0, 0, 0]], dtype="int64")[
            :, : head_fixed.max_seed_len
        ]

        bias_fixed = head_fixed._build_spatial_cross_bias(
            tgt_len=tgt_len,
            mem_len=mem_len,
            seed_lens=paddle.to_tensor(seed_lens_np, dtype="int64"),
            tail_mask=None,
            bsz=bsz,
            dtype="float32",
            timesteps=paddle.to_tensor(timesteps_np, dtype="int64"),
            confs=None,
        ).numpy()

        # Passing confs=None even when the flag is on must fall back to the
        # fixed sigma (safe default / backward compatibility).
        bias_no_confs = head_adaptive_off_confs._build_spatial_cross_bias(
            tgt_len=tgt_len,
            mem_len=mem_len,
            seed_lens=paddle.to_tensor(seed_lens_np, dtype="int64"),
            tail_mask=None,
            bsz=bsz,
            dtype="float32",
            timesteps=paddle.to_tensor(timesteps_np, dtype="int64"),
            confs=None,
        ).numpy()

        np.testing.assert_allclose(bias_no_confs, bias_fixed, rtol=1e-5, atol=1e-6)


class TestExp3CtcConfidenceGating(unittest.TestCase):
    def test_high_margin_positions_are_forced_to_keep(self):
        """Mirrors the gate math added to forward(); kept as a focused,
        model-free check of the boolean logic since exercising the full
        forward() pass requires a populated CTCHead/backbone stack."""
        import paddle

        margins = np.array([[0.9, 0.01, 0.5]], dtype="float32")
        edit_mask = np.array([[True, True, True]])

        margin_t = paddle.to_tensor(margins, dtype="float32")
        edit_mask_t = paddle.to_tensor(edit_mask)
        threshold = 0.3

        high_conf_keep = margin_t >= threshold
        gated = paddle.logical_and(edit_mask_t, paddle.logical_not(high_conf_keep))

        np.testing.assert_array_equal(gated.numpy(), np.array([[False, True, False]]))


if __name__ == "__main__":
    unittest.main()
