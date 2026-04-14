# Environment Setup

This repository is meant to run from a project-local virtual environment.

## 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

## 2. Install PyTorch first

Install `torch` and `torchaudio` using the official PyTorch instructions that match your CPU or CUDA runtime.

Example:

```bash
pip install torch torchaudio
```

## 3. Install the remaining Python dependencies

```bash
pip install -r requirements.txt
```

If you only need the WebUI layer on top of an existing ML environment, you can install just:

```bash
pip install -r requirements-webui.txt
```

## 4. Install ffmpeg

`ffmpeg` is required so the WebUI can accept browser recordings and common compressed audio formats.

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## 5. Place the dataset

Download RAVDESS and extract it to:

```text
data/ravdess/
```

The expected structure is:

```text
data/ravdess/Actor_01/*.wav
data/ravdess/Actor_02/*.wav
...
data/ravdess/Actor_24/*.wav
```

## 6. Verify the CLI

```bash
python speech_emotion_classifier.py --help
python speech_emotion_classifier.py --mode predict --audio data/ravdess/Actor_01/03-01-08-02-01-01-01.wav
```

## 7. Run the WebUI

```bash
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

For microphone recording, prefer opening the page from `localhost` or `127.0.0.1`.
