# Src Layout Public Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the project into a `src/transformer_mood/` package layout, update entry points and docs, and remove generated artifacts from the public-facing repository state.

**Architecture:** Keep the existing behavior and large training script intact, but move the FastAPI app, classifier module, and templates into a single package under `src/transformer_mood/`. Resolve repository-root paths explicitly from the package, then tighten `.gitignore` and remove tracked output artifacts so the repo is safe to publish.

**Tech Stack:** Python, FastAPI, Jinja2, uvicorn, Git, Markdown

---

## File Structure

- `src/transformer_mood/__init__.py` — package marker for the new src layout
- `src/transformer_mood/main.py` — FastAPI app entry point, template loading, API routes
- `src/transformer_mood/speech_emotion_classifier.py` — model, training, inference, dataset/output path configuration
- `src/transformer_mood/templates/index.html` — WebUI template
- `src/transformer_mood/static/.gitkeep` — placeholder for standard static asset layout
- `output/.gitkeep` — keeps the output directory while ignoring generated contents
- `.gitignore` — public-repo ignore policy for local envs, caches, dataset, and generated artifacts
- `README.md` — English docs with new layout and package-based commands
- `README.zh.md` — Chinese docs with new layout and package-based commands
- `ENVIRONMENT.md` — environment setup instructions aligned with the src layout

### Task 1: Create the src package layout and move the WebUI entrypoint

**Files:**
- Create: `src/transformer_mood/__init__.py`
- Create: `src/transformer_mood/static/.gitkeep`
- Create: `src/transformer_mood/main.py`
- Modify: `src/transformer_mood/main.py`
- Move: `app_fastapi.py` -> `src/transformer_mood/main.py`
- Move: `templates/index.html` -> `src/transformer_mood/templates/index.html`

- [ ] **Step 1: Create the src package directories**

Run:

```bash
mkdir -p src/transformer_mood/templates src/transformer_mood/static
```

Expected: command exits with code 0 and both directories exist.

- [ ] **Step 2: Create the package marker file**

Write `src/transformer_mood/__init__.py`:

```python
"""Transformer Mood package."""
```

- [ ] **Step 3: Move the FastAPI entrypoint and template into the package**

Run:

```bash
git mv app_fastapi.py src/transformer_mood/main.py
git mv templates/index.html src/transformer_mood/templates/index.html
touch src/transformer_mood/static/.gitkeep
```

Expected: `git status --short` shows `R  app_fastapi.py -> src/transformer_mood/main.py`, `R  templates/index.html -> src/transformer_mood/templates/index.html`, and `A  src/transformer_mood/static/.gitkeep`.

- [ ] **Step 4: Update package imports and template path handling in `src/transformer_mood/main.py`**

Replace the import and path setup block with:

```python
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from transformer_mood.speech_emotion_classifier import (
    DEVICE,
    IDX_TO_EMOTION,
    N_MELS,
    OUTPUT_DIR,
    SpeechEmotionClassifier,
    predict_single,
)


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
TEMPLATES = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
DEFAULT_MODEL_PATH = Path(OUTPUT_DIR) / "best_model.pth"
```

This is the only structural change needed in `main.py`; keep the route handlers and inference logic intact.

- [ ] **Step 5: Run an import smoke check for the moved FastAPI module**

Run:

```bash
PYTHONPATH=src python -c "from transformer_mood.main import app; print(app.title)"
```

Expected output:

```text
Transformer Mood WebUI
```

### Task 2: Move the classifier module and repair repo-root paths

**Files:**
- Move: `speech_emotion_classifier.py` -> `src/transformer_mood/speech_emotion_classifier.py`
- Modify: `src/transformer_mood/speech_emotion_classifier.py`

- [ ] **Step 1: Move the classifier module into the package**

Run:

```bash
git mv speech_emotion_classifier.py src/transformer_mood/speech_emotion_classifier.py
```

Expected: `git status --short` shows the rename under `src/transformer_mood/`.

- [ ] **Step 2: Replace the root-relative data and output constants**

In `src/transformer_mood/speech_emotion_classifier.py`, replace:

```python
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ravdess")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
```

with:

```python
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = str(PROJECT_ROOT / "data" / "ravdess")
OUTPUT_DIR = str(PROJECT_ROOT / "output")
```

and add this import near the top:

```python
from pathlib import Path
```

- [ ] **Step 3: Update the usage text so the script advertises the package entrypoint**

In the module docstring, replace:

```text
训练:   python speech_emotion_classifier.py --mode train
推理:   python speech_emotion_classifier.py --mode predict --audio <path_to_wav>
```

with:

```text
训练:   PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode train
推理:   PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode predict --audio <path_to_wav>
```

At the end of `main()`, replace:

```python
print(f"    python speech_emotion_classifier.py --mode predict --audio <your.wav>")
```

with:

```python
print(
    "    PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier "
    "--mode predict --audio <your.wav>"
)
```

- [ ] **Step 4: Run a module import and CLI help smoke check**

Run:

```bash
PYTHONPATH=src python -c "from transformer_mood.speech_emotion_classifier import DATA_DIR, OUTPUT_DIR; print(DATA_DIR); print(OUTPUT_DIR)"
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --help
```

Expected output:

```text
.../data/ravdess
.../output
usage: ...
```

The exact absolute paths can vary, but they must end in `/data/ravdess` and `/output`.

### Task 3: Clean generated artifacts for the public repository and tighten `.gitignore`

**Files:**
- Create: `output/.gitkeep`
- Modify: `.gitignore`
- Delete: `output/best_model.pth`
- Delete: `output/model_complete.pth`
- Delete: `output/confusion_matrix.png`
- Delete: `output/training_curves.png`
- Delete: `output/emotion_distribution.png`

- [ ] **Step 1: Replace `.gitignore` with a public-repo-safe version**

Set `.gitignore` to:

```gitignore
.venv/
__pycache__/
.playwright-mcp/
.pytest_cache/
.mypy_cache/
*.pyc
*.pyo
*.log
.DS_Store
.idea/
.vscode/

# Local dataset cache
data/ravdess/

# Legacy root-level artifacts
/best_model.pth
/training_curves.png
/confusion_matrix.png

# Generated training outputs
output/*
!output/.gitkeep
```

- [ ] **Step 2: Add the output placeholder and remove tracked generated artifacts**

Run:

```bash
touch output/.gitkeep
git rm output/best_model.pth output/model_complete.pth output/confusion_matrix.png output/training_curves.png output/emotion_distribution.png
rm -f best_model.pth training_curves.png confusion_matrix.png
```

Expected: tracked `output/` artifacts are staged for deletion, the root legacy artifacts are removed locally if present, and `output/.gitkeep` exists.

- [ ] **Step 3: Verify the ignore rules on representative paths**

Run:

```bash
git check-ignore .venv data/ravdess output/best_model.pth best_model.pth .vscode/settings.json
git check-ignore -v output/.gitkeep || true
```

Expected:

```text
.gitignore:...:.venv/ .venv
.gitignore:...:data/ravdess/ data/ravdess
.gitignore:...:output/* output/best_model.pth
.gitignore:...:/best_model.pth best_model.pth
.gitignore:...:.vscode/ .vscode/settings.json
```

and `output/.gitkeep` must not be ignored.

### Task 4: Update the docs to match the src layout and package commands

**Files:**
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `ENVIRONMENT.md`

- [ ] **Step 1: Update the repository layout block in `README.md`**

Replace the current layout block with:

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
output/                        # Runtime output directory (ignored except .gitkeep)
data/README.md                 # Dataset placement notes
data/README.zh.md              # Chinese dataset placement notes
```

- [ ] **Step 2: Update the English startup commands in `README.md`**

Replace the CLI and WebUI commands with:

```bash
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode train
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode predict --audio path/to/example.wav
PYTHONPATH=src python -m uvicorn transformer_mood.main:app --host 127.0.0.1 --port 8000
```

Add a note under `Notes`:

```md
- `output/` is kept as a directory boundary, but generated model files and figures are ignored for the public repository
```

- [ ] **Step 3: Apply the same command and layout updates in `README.zh.md`**

Replace the corresponding Chinese command blocks with:

```bash
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode train
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode predict --audio path/to/example.wav
PYTHONPATH=src python -m uvicorn transformer_mood.main:app --host 127.0.0.1 --port 8000
```

and update the Chinese notes section to mention:

```md
- `output/` 目录仅保留结构，生成的模型和训练图不会提交到公开仓库
```

- [ ] **Step 4: Update `ENVIRONMENT.md` to use the package entrypoints**

Replace the CLI verification and WebUI run section with:

```bash
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --help
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode predict --audio data/ravdess/Actor_01/03-01-08-02-01-01-01.wav
PYTHONPATH=src python -m uvicorn transformer_mood.main:app --host 127.0.0.1 --port 8000
```

- [ ] **Step 5: Read the docs back and verify the commands all point to `src/transformer_mood/`**

Run:

```bash
rg -n "app_fastapi|speech_emotion_classifier.py --mode|python -m uvicorn app_fastapi:app|templates/index.html" README.md README.zh.md ENVIRONMENT.md
```

Expected: no matches.

### Task 5: Run final repo-structure and application verification

**Files:**
- Verify: `src/transformer_mood/main.py`
- Verify: `src/transformer_mood/speech_emotion_classifier.py`
- Verify: `src/transformer_mood/templates/index.html`
- Verify: `.gitignore`
- Verify: `README.md`
- Verify: `README.zh.md`
- Verify: `ENVIRONMENT.md`

- [ ] **Step 1: Verify the new repository structure exists**

Run:

```bash
test -f src/transformer_mood/__init__.py
test -f src/transformer_mood/main.py
test -f src/transformer_mood/speech_emotion_classifier.py
test -f src/transformer_mood/templates/index.html
test -f src/transformer_mood/static/.gitkeep
test -f output/.gitkeep
```

Expected: all commands exit 0.

- [ ] **Step 2: Start the FastAPI app with the new entrypoint and hit the health endpoint**

Run:

```bash
PYTHONPATH=src python -m uvicorn transformer_mood.main:app --host 127.0.0.1 --port 8000 >/tmp/transformer-mood.log 2>&1 &
server_pid=$!
sleep 2
curl -sS http://127.0.0.1:8000/api/health
kill $server_pid
```

Expected JSON contains:

```json
{"status":"ok"}
```

and the command exits cleanly.

- [ ] **Step 3: Confirm there are no root-level business scripts left behind**

Run:

```bash
test ! -f app_fastapi.py
test ! -f speech_emotion_classifier.py
```

Expected: both commands exit 0.

- [ ] **Step 4: Review the final repo state before committing**

Run:

```bash
git status --short
```

Expected: only the intended moved source files, doc updates, `.gitignore` changes, placeholder files, and artifact deletions appear.
