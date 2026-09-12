"""Contract and runtime tests for the independent E37 sequence verifier."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "ppocr/modeling/heads/sequence_verifier.py"


class TestE37Verifier(unittest.TestCase):
    def test_verifier_is_independent_and_sequence_level(self):
        text = VERIFIER.read_text()
        self.assertIn("class SequenceVisualVerifier", text)
        self.assertIn("candidate_ids", text)
        self.assertIn("candidate_lens", text)
        self.assertIn("cand * image", text)

    def test_verifier_rejects_invalid_shapes(self):
        try:
            import paddle
            from ppocr.modeling.heads.sequence_verifier import SequenceVisualVerifier
        except ImportError:
            self.skipTest("PaddlePaddle is available on the Arbor runtime")
        model = SequenceVisualVerifier(
            in_channels=8, vocab_size=20, hidden=16, max_length=6,
            layers=0, heads=4, dropout=0.0
        )
        with self.assertRaises(ValueError):
            model(
                paddle.zeros([1, 8, 3, 4]),
                paddle.zeros([1, 2, 6], dtype="int64"),
                paddle.ones([1, 2], dtype="int64"),
            )

    def test_verifier_output_is_finite(self):
        try:
            import paddle
            from ppocr.modeling.heads.sequence_verifier import SequenceVisualVerifier
        except ImportError:
            self.skipTest("PaddlePaddle is available on the Arbor runtime")
        paddle.seed(11)
        model = SequenceVisualVerifier(
            in_channels=8, vocab_size=20, hidden=16, max_length=6,
            layers=1, heads=4, dropout=0.0
        )
        model.eval()
        score = model(
            paddle.randn([2, 8, 3, 5]),
            paddle.to_tensor([[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0]], dtype="int64"),
            paddle.to_tensor([3, 2], dtype="int64"),
        )
        self.assertEqual(list(score.shape), [2])
        self.assertTrue(bool(paddle.all(paddle.isfinite(score)).item()))


if __name__ == "__main__":
    unittest.main()
