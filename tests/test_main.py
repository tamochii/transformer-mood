from pathlib import Path
import sys
import tempfile
import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

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

    def test_load_model_state_accepts_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model_complete.pth"
            torch.save(
                {
                    "model_state_dict": main.SpeechEmotionClassifier(input_dim=128, num_classes=8).state_dict(),
                    "config": {
                        "input_dim": 128,
                        "d_model": 128,
                        "nhead": 8,
                        "num_layers": 4,
                        "dim_feedforward": 512,
                        "dropout": 0.2,
                        "num_classes": 8,
                    },
                    "emotion_map": {i: name for i, name in enumerate(main.IDX_TO_EMOTION.values())},
                },
                model_path,
            )

            model, error = main._load_model_state(model_path)

        self.assertIsNotNone(model)
        self.assertIsNone(error)

    def test_format_prediction_uses_checkpoint_emotion_labels(self):
        result = {
            "predicted_emotion": "surprised",
            "confidence": 0.91,
            "all_probabilities": {
                "angry": 0.01,
                "disgust": 0.01,
                "fearful": 0.01,
                "happy": 0.01,
                "neutral": 0.01,
                "sad": 0.05,
                "surprised": 0.91,
            },
        }

        payload = main._format_prediction(result, "sample.wav", "audio/wav")

        self.assertEqual(payload["predicted_emotion"], "surprised")
        self.assertEqual(payload["probabilities"][0]["emotion"], "surprised")

    def test_predict_audio_passes_checkpoint_emotion_map(self):
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    model=object(),
                    model_error=None,
                    ffmpeg_path=None,
                    emotion_map={0: "angry", 1: "surprised"},
                )
            )
        )
        audio = SimpleNamespace(
            filename="sample.wav",
            content_type="audio/wav",
            read=AsyncMock(side_effect=[b"fake-audio", b""]),
            close=AsyncMock(),
        )

        async def run_case():
            with patch.object(
                main,
                "run_in_threadpool",
                new=AsyncMock(
                    side_effect=[
                        None,
                        {
                            "predicted_emotion": "surprised",
                            "confidence": 0.9,
                            "all_probabilities": {"angry": 0.1, "surprised": 0.9},
                        },
                    ]
                ),
            ) as threadpool_mock:
                response = await main.predict_audio(request, audio)
                predict_call = threadpool_mock.await_args_list[1]
                self.assertIs(predict_call.args[0], main.predict_single)
                self.assertEqual(predict_call.args[4], {0: "angry", 1: "surprised"})
                self.assertEqual(response["predicted_emotion"], "surprised")

        asyncio.run(run_case())

    def test_health_uses_loaded_emotion_map_labels(self):
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ffmpeg_path=None,
                    model_path="/tmp/model_complete.pth",
                    model=object(),
                    model_error=None,
                    emotion_map={0: "angry", 1: "surprised"},
                )
            )
        )

        async def run_case():
            payload = await main.health(request)
            self.assertEqual(payload["labels"], ["angry", "surprised"])

        asyncio.run(run_case())


if __name__ == "__main__":
    unittest.main()
