# Run.py Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single `run.py` launcher that bootstraps the local environment and dispatches WebUI, training, prediction, and diagnostics across Windows and Linux.

**Architecture:** Keep all application logic in the existing `src/transformer_mood/` modules and make `run.py` an orchestration layer only. The launcher will choose or create `.venv`, verify/install dependencies, validate model and dataset requirements per subcommand, and then execute the existing modules with the correct environment.

**Tech Stack:** Python standard library, argparse, subprocess, venv, unittest, FastAPI/uvicorn module entrypoints

---

## File Structure

- `run.py` — top-level launcher with CLI parsing, OS detection, `.venv` selection/bootstrap, dependency checks, diagnostics, and subcommand dispatch
- `tests/test_run.py` — unit tests for launcher argument parsing, platform-specific venv path selection, and prerequisite validation
- `README.md` — update recommended commands to use `python run.py ...`
- `README.zh.md` — same documentation updates in Chinese

### Task 1: Add the launcher tests first

**Files:**
- Create: `tests/test_run.py`

- [ ] **Step 1: Write a failing test for the default subcommand**

Write a unit test asserting that parsing no subcommand maps to `webui`.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m unittest tests.test_run.RunArgParsingTests.test_default_subcommand_is_webui -v
```

Expected: FAIL because `run.py` or its parser helper does not exist yet.

- [ ] **Step 3: Write failing tests for platform-specific venv paths and command validation**

Add tests for:

- Linux venv interpreter path resolves to `.venv/bin/python`
- Windows venv interpreter path resolves to `.venv/Scripts/python.exe`
- `predict` rejects missing model paths
- `webui` allows missing model paths but reports a warning state
- `train` rejects a missing `data/ravdess/` directory

- [ ] **Step 4: Run the launcher test module to verify the expected failures**

Run:

```bash
python -m unittest tests.test_run -v
```

Expected: FAIL because launcher functions are not implemented yet.

### Task 2: Implement `run.py` with minimal orchestration logic

**Files:**
- Create: `run.py`
- Modify: `run.py`

- [ ] **Step 1: Add CLI parsing and default `webui` behavior**

Implement parser helpers that support:

- default no-subcommand execution as `webui`
- `webui --host --port --model`
- `train` with passthrough args
- `predict --audio --model`
- `doctor`

- [ ] **Step 2: Add platform-aware `.venv` interpreter resolution**

Implement a helper that returns:

- `.venv/bin/python` on Linux
- `.venv/Scripts/python.exe` on Windows

- [ ] **Step 3: Add environment bootstrap helpers**

Implement helpers that:

- create `.venv` with the current interpreter if missing
- verify required imports inside the chosen interpreter
- install dependencies in order with:

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

- [ ] **Step 4: Add prerequisite validation per subcommand**

Implement minimal checks for:

- missing model behavior split between `webui` and `predict`
- missing dataset behavior for `train`
- `ffmpeg` presence reporting

- [ ] **Step 5: Add subprocess dispatch to existing modules**

Implement launcher execution through:

```bash
<venv-python> -m uvicorn transformer_mood.main:app ...
<venv-python> -m transformer_mood.speech_emotion_classifier --mode train ...
<venv-python> -m transformer_mood.speech_emotion_classifier --mode predict ...
```

with `PYTHONPATH=src` in the child environment.

- [ ] **Step 6: Run the launcher tests again**

Run:

```bash
python -m unittest tests.test_run -v
```

Expected: PASS.

### Task 3: Update README usage to promote `run.py`

**Files:**
- Modify: `README.md`
- Modify: `README.zh.md`

- [ ] **Step 1: Replace the primary command examples in `README.md`**

Update the English README so the recommended commands are:

```bash
python run.py
python run.py webui --host 127.0.0.1 --port 8000
python run.py train
python run.py predict --audio path/to/example.wav
python run.py doctor
```

- [ ] **Step 2: Apply the same update in `README.zh.md`**

Update the Chinese README with the same command set, localized prose, and `run.py` as the primary entrypoint.

- [ ] **Step 3: Verify no stale “recommended” direct commands remain**

Run:

```bash
python - <<'PY'
from pathlib import Path
for path in [Path('README.md'), Path('README.zh.md')]:
    text = path.read_text(encoding='utf-8')
    assert 'python run.py' in text
PY
```

Expected: command exits 0.

### Task 4: Verify the launcher in the real repository environment

**Files:**
- Verify: `run.py`
- Verify: `tests/test_run.py`
- Verify: `README.md`
- Verify: `README.zh.md`

- [ ] **Step 1: Run diagnostics**

Run:

```bash
python run.py doctor
```

Expected: prints OS, interpreter, venv, dependency, ffmpeg, model, and dataset status without crashing.

- [ ] **Step 2: Verify the default command path**

Run:

```bash
python run.py --help
```

Expected: shows the launcher CLI and available subcommands.

- [ ] **Step 3: Verify WebUI startup through the launcher**

Run:

```bash
python run.py webui --host 127.0.0.1 --port 18010
```

Then probe:

```bash
python - <<'PY'
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
print(opener.open('http://127.0.0.1:18010/api/health').read().decode())
PY
```

Expected: `/api/health` returns JSON with `"status":"ok"`.

- [ ] **Step 4: Verify missing-model handling for prediction**

Run with a nonexistent model path:

```bash
python run.py predict --audio missing.wav --model definitely-missing-model.pth
```

Expected: exits non-zero with a clear model-path error before trying inference.

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_run.py README.md README.zh.md
git commit -m "feat: add a cross-platform project launcher"
```
