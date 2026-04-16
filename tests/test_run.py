import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_run_module(test_case: unittest.TestCase):
    spec = importlib.util.find_spec("run")
    test_case.assertIsNotNone(spec, "run.py should be importable as module 'run'")
    return importlib.import_module("run")


class RunArgParsingTests(unittest.TestCase):
    def test_default_subcommand_is_webui(self):
        run = load_run_module(self)

        args = run.parse_args([])

        self.assertEqual(args.command, "webui")


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


if __name__ == "__main__":
    unittest.main()
