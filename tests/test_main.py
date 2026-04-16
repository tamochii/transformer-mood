from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import transformer_mood.main as main


class LoadModelStateTests(unittest.TestCase):
    def test_missing_model_path_returns_error_instead_of_raising(self):
        missing_model = Path("/tmp/transformer-mood-missing-model.pth")

        self.assertTrue(
            hasattr(main, "_load_model_state"),
            "transformer_mood.main should expose _load_model_state",
        )
        model, error = main._load_model_state(missing_model)

        self.assertIsNone(model)
        self.assertIsNotNone(error)
        self.assertIn("模型文件不存在", error)
        self.assertIn(str(missing_model), error)


if __name__ == "__main__":
    unittest.main()
