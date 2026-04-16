<p align="right">
  English | <a href="./README.zh.md">简体中文</a>
</p>

<h1 align="center">Transformer Mood</h1>

<p align="center">
  A local-first speech emotion recognition toolkit with training, CLI inference, and a FastAPI WebUI.
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB.svg">
  <img alt="Model" src="https://img.shields.io/badge/Model-Transformer-7A3FFF.svg">
  <img alt="WebUI" src="https://img.shields.io/badge/WebUI-FastAPI-009688.svg">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-RAVDESS-EA4C89.svg">
</p>

## Features

- RAVDESS dataset preprocessing and training
- Transformer Encoder speech emotion classification model
- Single-file CLI inference
- FastAPI WebUI
- Browser microphone recording, audio uploads, and probability display

## Repository Layout

```text
src/
  transformer_mood/
    __init__.py
    main.py
    speech_emotion_classifier.py
    templates/
      index.html
    static/
      .gitkeep
README.zh.md                    # Chinese project README
output/                        # Runtime output directory (ignored except .gitkeep)
data/README.md                 # Dataset placement notes
data/README.zh.md              # Chinese dataset placement notes
transformer-md/                # Reference materials
requirements.txt               # Non-PyTorch Python dependencies
requirements-webui.txt         # Minimal extra dependencies for the WebUI
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
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode train
```

## CLI Prediction

```bash
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode predict --audio path/to/example.wav
```

## WebUI

```bash
PYTHONPATH=src python -m uvicorn transformer_mood.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The WebUI supports:

- Uploading local audio files
- Recording from the browser microphone
- Displaying predicted emotion, confidence, and full probability distribution

Prediction requires a local checkpoint at `output/best_model.pth`, or an explicit `EMOTION_MODEL_PATH` environment variable.

## Notes

- This project is released under the MIT License. See `LICENSE` for details.
- `data/ravdess/` is in `.gitignore` so the raw dataset will not be committed accidentally
- Legacy root-level model and image artifacts are ignored; the current expected outputs live in `output/`
- `output/` is kept as a directory boundary, but generated model files and figures are ignored for the public repository
