# Transformer Mood

[中文](README.zh.md) | English

Transformer-based speech emotion recognition project with training scripts, CLI inference, and a local WebUI that supports both file uploads and browser microphone recording.

## Features

- RAVDESS dataset preprocessing and training
- Transformer Encoder speech emotion classification model
- Single-file CLI inference
- FastAPI WebUI
- Browser microphone recording, audio uploads, and probability display

## Repository Layout

```text
README.zh.md                    # Chinese project README
speech_emotion_classifier.py   # Training and CLI inference entry point
app_fastapi.py                 # FastAPI WebUI server
templates/index.html           # WebUI template
output/                        # Generated models and visual artifacts
data/README.md                 # Dataset placement notes
data/README.zh.md              # Chinese dataset placement notes
transformer-md/                # Reference materials
requirements.txt               # Non-PyTorch Python dependencies
requirements-webui.txt         # Minimal extra dependencies for the WebUI
ENVIRONMENT.md                 # Environment and runtime notes
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install `torch` and `torchaudio` first, then install the remaining dependencies:

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

Install `ffmpeg`:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Dataset

This repository does not include the RAVDESS dataset itself. Download it manually and place it under:

```text
data/ravdess/
```

See `data/README.md` for details.

## Training

```bash
python speech_emotion_classifier.py --mode train
```

## CLI Prediction

```bash
python speech_emotion_classifier.py --mode predict --audio path/to/example.wav
```

## WebUI

```bash
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The WebUI supports:

- Uploading local audio files
- Recording from the browser microphone
- Displaying predicted emotion, confidence, and full probability distribution

## Notes

- `data/ravdess/` is in `.gitignore` so the raw dataset will not be committed accidentally
- Legacy root-level model and image artifacts are ignored; the current expected outputs live in `output/`
- Files in `output/` can be kept as project artifacts or regenerated as needed
