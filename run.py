from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_IMPORTS = ["fastapi", "uvicorn", "torch", "torchaudio", "librosa"]


@dataclass
class ValidationResult:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    model_path: Path | None = None
    dataset_path: Path | None = None
    ffmpeg_path: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformer Mood project launcher")
    subparsers = parser.add_subparsers(dest="command")

    webui = subparsers.add_parser("webui", help="Start the FastAPI WebUI")
    webui.add_argument("--host", default="127.0.0.1")
    webui.add_argument("--port", type=int, default=8000)
    webui.add_argument("--model", default=None)

    train = subparsers.add_parser("train", help="Run model training")
    train.add_argument("extra_args", nargs=argparse.REMAINDER)

    predict = subparsers.add_parser("predict", help="Run single-file prediction")
    predict.add_argument("--audio", required=True)
    predict.add_argument("--model", default=None)

    subparsers.add_parser("doctor", help="Print environment diagnostics")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["webui"]
    return build_parser().parse_args(argv)


def get_venv_python_path(repo_root: Path, platform_name: str | None = None) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


def resolve_model_path(repo_root: Path, explicit_model: str | None) -> Path:
    if explicit_model:
        return Path(explicit_model).expanduser().resolve()
    return repo_root / "output" / "best_model.pth"


def validate_command_requirements(
    command: str,
    repo_root: Path,
    explicit_model: str | None,
    ffmpeg_path: str | None,
) -> ValidationResult:
    result = ValidationResult(ffmpeg_path=ffmpeg_path)

    if command in {"webui", "predict"}:
        result.model_path = resolve_model_path(repo_root, explicit_model)
        if not result.model_path.exists():
            message = f"Model file not found: {result.model_path}. Expected output/best_model.pth or use --model."
            if command == "predict":
                result.errors.append(message)
            else:
                result.warnings.append(message)

    if command == "train":
        result.dataset_path = repo_root / "data" / "ravdess"
        if not result.dataset_path.is_dir():
            result.errors.append(
                f"Dataset not found: {result.dataset_path}. Place RAVDESS under data/ravdess/."
            )

    if command in {"webui", "predict"} and ffmpeg_path is None:
        result.warnings.append("ffmpeg not found in PATH.")

    return result


def run_checked(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, env=env)


def ensure_venv(repo_root: Path) -> Path:
    venv_python = get_venv_python_path(repo_root)
    if venv_python.exists():
        print(f"[INFO] Using virtual environment: {venv_python}")
        return venv_python

    print("[INFO] Creating virtual environment...")
    run_checked([sys.executable, "-m", "venv", str(repo_root / ".venv")])
    return venv_python


def build_child_env(repo_root: Path, model_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current_pythonpath else os.pathsep.join([src_path, current_pythonpath])
    if model_path is not None:
        env["EMOTION_MODEL_PATH"] = str(model_path)
    return env


def dependencies_ready(venv_python: Path, repo_root: Path) -> bool:
    try:
        run_checked(
            [str(venv_python), "-c", "import fastapi, uvicorn, torch, torchaudio, librosa"],
            env=build_child_env(repo_root),
        )
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_dependencies(venv_python: Path, repo_root: Path) -> None:
    if dependencies_ready(venv_python, repo_root):
        print("[INFO] Required Python packages already available.")
        return

    print("[INFO] Installing dependencies...")
    run_checked([str(venv_python), "-m", "pip", "install", "torch", "torchaudio"])
    run_checked([str(venv_python), "-m", "pip", "install", "-r", str(repo_root / "requirements.txt")])


def print_validation(result: ValidationResult) -> None:
    for warning in result.warnings:
        print(f"[WARN] {warning}")
    for error in result.errors:
        print(f"[ERROR] {error}")


def print_doctor_report(
    repo_root: Path,
    venv_python: Path,
    dependencies_ok: bool,
    ffmpeg_path: str | None,
) -> None:
    print(f"[INFO] Detected platform: {sys.platform}")
    print(f"[INFO] Launcher Python: {sys.executable}")
    print(f"[INFO] Using virtual environment: {venv_python}")
    print(f"[INFO] .venv exists: {(repo_root / '.venv').exists()}")
    print(f"[INFO] Required imports available: {dependencies_ok}")
    print(f"[INFO] ffmpeg: {ffmpeg_path or 'missing'}")
    print(f"[INFO] model: {resolve_model_path(repo_root, None)}")
    print(f"[INFO] model exists: {resolve_model_path(repo_root, None).exists()}")
    print(f"[INFO] dataset exists: {(repo_root / 'data' / 'ravdess').is_dir()}")


def dispatch_command(args: argparse.Namespace, repo_root: Path, venv_python: Path) -> int:
    ffmpeg_path = shutil.which("ffmpeg")
    command = args.command
    explicit_model = getattr(args, "model", None)
    validation = validate_command_requirements(command, repo_root, explicit_model, ffmpeg_path)
    print_validation(validation)
    if validation.errors:
        return 1

    child_env = build_child_env(repo_root, validation.model_path if explicit_model else None)

    if command == "doctor":
        print_doctor_report(repo_root, venv_python, dependencies_ready(venv_python, repo_root), ffmpeg_path)
        return 0

    if command == "webui":
        proc = [
            str(venv_python),
            "-m",
            "uvicorn",
            "transformer_mood.main:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    elif command == "train":
        proc = [str(venv_python), "-m", "transformer_mood.speech_emotion_classifier", "--mode", "train"] + args.extra_args
    elif command == "predict":
        proc = [
            str(venv_python),
            "-m",
            "transformer_mood.speech_emotion_classifier",
            "--mode",
            "predict",
            "--audio",
            args.audio,
        ]
    else:
        raise ValueError(f"Unsupported command: {command}")

    return subprocess.run(proc, env=child_env).returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    venv_python = ensure_venv(repo_root)
    ensure_dependencies(venv_python, repo_root)
    return dispatch_command(args, repo_root, venv_python)


if __name__ == "__main__":
    raise SystemExit(main())
