"""Unit tests for the E44-A shared high-resolution visual memory path.

The runtime tests are skipped when PaddlePaddle is not installed locally.  The
static contract tests still run in the lightweight development environment and
protect the model/config wiring before an Arbor training job is submitted.
"""

from pathlib import Path

import unittest


ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / "ppocr/modeling/heads/rec_edit_refine_nrtr_head.py"
BASE_MODEL = ROOT / "ppocr/modeling/architectures/base_model.py"
CONFIG = ROOT / "config/PP-OCRv6_small_rec_s1024_e44_highres.yml"


class TestE44Architecture(unittest.TestCase):
    def test_e44_config_enables_shared_highres_visual_memory(self):
        text = CONFIG.read_text()
        self.assertIn("name: MultiHeadEditRefineNRTR", text)
        self.assertIn("edit_mode: token_refine", text)
        self.assertIn("use_highres_visual: true", text)
        self.assertIn("visual_pos_max_height:", text)
        self.assertIn("visual_pos_max_width:", text)

    def test_backbone_reference_is_wired_for_highres_head(self):
        text = BASE_MODEL.read_text()
        self.assertIn("use_highres_visual", text)
        self.assertIn("weakref.ref(self.backbone)", text)
        self.assertIn("self.head.backbone_ref = self.backbone", text)

    def test_head_contains_explicit_2d_visual_memory_contract(self):
        text = HEAD.read_text()
        self.assertIn("class SharedHighResVisualMemory", text)
        self.assertIn("visual_type_embed", text)
        self.assertIn("visual_row_embed", text)
        self.assertIn("visual_col_embed", text)
        self.assertIn("recon_feat", text)

    def test_shared_visual_memory_preserves_2d_tokens_and_shape(self):
        try:
            import paddle
        except ImportError:
            self.skipTest("PaddlePaddle is available on the Arbor runtime")
        from ppocr.modeling.heads.rec_edit_refine_nrtr_head import (
            SharedHighResVisualMemory,
        )

        paddle.seed(7)
        module = SharedHighResVisualMemory(
            in_channels=8,
            dim=16,
            max_height=4,
            max_width=8,
            layers=1,
            nhead=4,
            dropout=0.0,
        )
        module.eval()
        x = paddle.randn([2, 8, 3, 5])
        y = module(x)
        self.assertEqual(list(y.shape), [2, 15, 16])
        self.assertTrue(bool(paddle.all(paddle.isfinite(y)).item()))

    def test_shared_visual_memory_interpolates_dynamic_position_range(self):
        try:
            import paddle
        except ImportError:
            self.skipTest("PaddlePaddle is available on the Arbor runtime")
        from ppocr.modeling.heads.rec_edit_refine_nrtr_head import (
            SharedHighResVisualMemory,
        )

        module = SharedHighResVisualMemory(
            in_channels=8,
            dim=16,
            max_height=2,
            max_width=4,
            layers=0,
            nhead=4,
            dropout=0.0,
        )
        y = module(paddle.zeros([1, 8, 3, 6]))
        self.assertEqual(list(y.shape), [1, 18, 16])
        self.assertTrue(bool(paddle.all(paddle.isfinite(y)).item()))


if __name__ == "__main__":
    unittest.main()
