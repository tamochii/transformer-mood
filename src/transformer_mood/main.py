from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from transformer_mood.speech_emotion_classifier import (
    DEVICE,
    FEATURE_DIM,
    IDX_TO_EMOTION,
    OUTPUT_DIR,
    SpeechEmotionClassifier,
    predict_single,
)


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
TEMPLATES = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
DEFAULT_MODEL_PATH = Path(OUTPUT_DIR) / "best_model.pth"
EMOTION_TO_ZH = {
    "neutral": "中性",
    "calm": "平静",
    "happy": "快乐",
    "sad": "悲伤",
    "angry": "愤怒",
    "fearful": "恐惧",
    "disgust": "厌恶",
    "surprised": "惊讶",
}


def _get_model_path() -> Path:
    configured_path = os.environ.get("EMOTION_MODEL_PATH")
    return Path(configured_path) if configured_path else DEFAULT_MODEL_PATH


def _load_model(model_path: Path) -> SpeechEmotionClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = SpeechEmotionClassifier(input_dim=FEATURE_DIM).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_model_state(model_path: Path) -> tuple[SpeechEmotionClassifier | None, str | None]:
    try:
        return _load_model(model_path), None
    except FileNotFoundError as exc:
        return None, str(exc)


def _normalize_audio(source_path: Path, normalized_path: Path, ffmpeg_path: str | None) -> None:
    if ffmpeg_path is None:
        if source_path.suffix.lower() != ".wav":
            raise RuntimeError("未找到 ffmpeg，当前仅支持直接上传 wav 文件。")
        shutil.copyfile(source_path, normalized_path)
        return

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(normalized_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg 无法解析该音频文件。"
        raise RuntimeError(detail)


def _format_prediction(result: dict, filename: str, content_type: str | None) -> dict:
    predicted = result["predicted_emotion"]
    probabilities = []
    for emotion, probability in sorted(
        result["all_probabilities"].items(), key=lambda item: item[1], reverse=True
    ):
        probabilities.append(
            {
                "emotion": emotion,
                "emotion_zh": EMOTION_TO_ZH.get(emotion, emotion),
                "probability": probability,
            }
        )

    return {
        "filename": filename,
        "content_type": content_type,
        "predicted_emotion": predicted,
        "predicted_emotion_zh": EMOTION_TO_ZH.get(predicted, predicted),
        "confidence": result["confidence"],
        "probabilities": probabilities,
        "device": str(DEVICE),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = _get_model_path()
    app.state.ffmpeg_path = shutil.which("ffmpeg")
    app.state.model_path = str(model_path)
    app.state.model, app.state.model_error = _load_model_state(model_path)
    yield


app = FastAPI(title="Transformer Mood WebUI", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "device": str(DEVICE),
            "ffmpeg_ready": bool(request.app.state.ffmpeg_path),
            "model_path": request.app.state.model_path,
        },
    )


@app.get("/api/health")
async def health(request: Request):
    return {
        "status": "ok",
        "device": str(DEVICE),
        "ffmpeg": request.app.state.ffmpeg_path,
        "model_path": request.app.state.model_path,
        "model_loaded": request.app.state.model is not None,
        "model_error": request.app.state.model_error,
        "labels": [IDX_TO_EMOTION[idx] for idx in sorted(IDX_TO_EMOTION)],
    }


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.post("/api/predict")
async def predict_audio(request: Request, audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="没有收到音频文件。")

    suffix = Path(audio.filename).suffix or ".bin"
    model = request.app.state.model
    ffmpeg_path = request.app.state.ffmpeg_path

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.model_error or "模型尚未加载。",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="emotion-webui-") as temp_dir:
            temp_dir_path = Path(temp_dir)
            source_path = temp_dir_path / f"upload{suffix}"
            normalized_path = temp_dir_path / "normalized.wav"

            with source_path.open("wb") as output_file:
                while chunk := await audio.read(1024 * 1024):
                    output_file.write(chunk)

            await run_in_threadpool(_normalize_audio, source_path, normalized_path, ffmpeg_path)
            result = await run_in_threadpool(predict_single, model, str(normalized_path), DEVICE)
            return _format_prediction(result, audio.filename, audio.content_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - last-resort safeguard for API clients
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc
    finally:
        await audio.close()
