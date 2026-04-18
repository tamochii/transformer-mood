import importlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_run_module(test_case: unittest.TestCase):
    spec = importlib.util.find_spec("run")
    test_case.assertIsNotNone(spec, "run.py should be importable as module 'run'")
    return importlib.import_module("run")


class RunArgParsingTests(unittest.TestCase):
    def test_default_subcommand_is_webui(self):
        run = load_run_module(self)

        args = run.parse_args([])

        self.assertEqual(args.command, "webui")

    def test_train_passthrough_strips_separator(self):
        run = load_run_module(self)

        args = run.parse_args(["train", "--", "--epochs", "1"])

        self.assertEqual(args.command, "train")
        self.assertEqual(args.extra_args, ["--epochs", "1"])

    def test_train_passthrough_keeps_dataset_and_cache_flags(self):
        run = load_run_module(self)

        args = run.parse_args(["train", "--", "--dataset", "tess", "--cache-features"])

        self.assertEqual(args.command, "train")
        self.assertEqual(args.extra_args, ["--dataset", "tess", "--cache-features"])


class RunEnvironmentPathTests(unittest.TestCase):
    def test_linux_venv_python_path(self):
        run = load_run_module(self)

        path = run.get_venv_python_path(Path("/tmp/project"), platform_name="linux")

        self.assertEqual(path, Path("/tmp/project/.venv/bin/python"))

    def test_windows_venv_python_path(self):
        run = load_run_module(self)

        path = run.get_venv_python_path(Path("C:/tmp/project"), platform_name="win32")

        self.assertEqual(path, Path("C:/tmp/project/.venv/Scripts/python.exe"))


class RunValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        (self.repo_root / "output").mkdir()
        (self.repo_root / "data").mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_predict_requires_existing_model(self):
        run = load_run_module(self)

        result = run.validate_command_requirements(
            command="predict",
            repo_root=self.repo_root,
            explicit_model=None,
            ffmpeg_path="/usr/bin/ffmpeg",
        )

        self.assertTrue(result.errors)
        self.assertIn("best_model.pth", result.errors[0])

    def test_webui_allows_missing_model_with_warning(self):
        run = load_run_module(self)

        result = run.validate_command_requirements(
            command="webui",
            repo_root=self.repo_root,
            explicit_model=None,
            ffmpeg_path=None,
        )

        self.assertFalse(result.errors)
        self.assertTrue(result.warnings)
        self.assertIn("best_model.pth", result.warnings[0])

    def test_train_requires_dataset_directory(self):
        run = load_run_module(self)

        result = run.validate_command_requirements(
            command="train",
            repo_root=self.repo_root,
            explicit_model=None,
            ffmpeg_path="/usr/bin/ffmpeg",
        )

        self.assertTrue(result.errors)
        self.assertIn("data/ravdess", result.errors[0])

    def test_train_requires_vec_directory_when_tess_dataset_requested(self):
        run = load_run_module(self)

        result = run.validate_command_requirements(
            command="train",
            repo_root=self.repo_root,
            explicit_model=None,
            ffmpeg_path="/usr/bin/ffmpeg",
            train_args=["--dataset", "tess"],
        )

        self.assertTrue(result.errors)
        self.assertIn("data/vec", result.errors[0])

    def test_webui_dispatch_uses_resolved_default_model_path(self):
        run = load_run_module(self)
        model_path = self.repo_root / "output" / "model_complete.pth"
        model_path.write_bytes(b"checkpoint")

        args = run.parse_args(["webui"])
        with patch.object(run.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            result = run.dispatch_command(
                args=args,
                repo_root=self.repo_root,
                venv_python=self.repo_root / ".venv" / "bin" / "python",
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            run_mock.call_args.kwargs["env"]["EMOTION_MODEL_PATH"],
            str(model_path),
        )

    def test_predict_dispatch_uses_resolved_default_model_path(self):
        run = load_run_module(self)
        model_path = self.repo_root / "output" / "model_complete.pth"
        model_path.write_bytes(b"checkpoint")
        audio_path = self.repo_root / "sample.wav"
        audio_path.write_bytes(b"fake")

        args = run.parse_args(["predict", "--audio", str(audio_path)])
        with patch.object(run.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            result = run.dispatch_command(
                args=args,
                repo_root=self.repo_root,
                venv_python=self.repo_root / ".venv" / "bin" / "python",
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            run_mock.call_args.kwargs["env"]["EMOTION_MODEL_PATH"],
            str(model_path),
        )


if __name__ == "__main__":
    unittest.main()
