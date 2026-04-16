# Run.py Launcher Design

## Goal

Add a single cross-platform startup entrypoint, `run.py`, that works on both freshly cloned machines and already-configured local machines, with predictable behavior for environment setup, dependency checks, model selection, and common project actions.

## Problem

The repository currently expects users to know several separate commands:

- how to create and activate `.venv`
- how to install dependencies in the correct order
- how to set `PYTHONPATH=src`
- how to start the WebUI
- how to run training or prediction
- how to deal with a missing checkpoint or missing `ffmpeg`

This is acceptable for a developer already familiar with the project, but it is not a good out-of-box experience for a newly cloned machine or for a public repository where users expect one obvious entrypoint.

## Core Decision

The repository will expose one user-facing launcher:

- `run.py`

Users will run the project through:

- `python run.py`
- `python run.py webui`
- `python run.py train`
- `python run.py predict --audio path/to/file.wav`
- `python run.py doctor`

This is intentionally a single Python file instead of separate `run.sh` and `run.bat` scripts, because Python already exists as the project runtime dependency and can make the cross-platform decisions internally.

## Scope

This design covers:

- platform detection for Windows vs Linux
- virtual environment detection and bootstrap
- dependency checks and installation
- `ffmpeg` checks
- model path selection for WebUI and prediction
- command dispatch for common tasks
- user-facing logging and failure messages

This design does not cover:

- GPU-specific PyTorch installation logic
- automatic model downloading
- packaging or publishing to PyPI
- replacing the existing underlying modules or training logic

## File Layout

The change will introduce one new top-level file:

- `run.py`

The launcher will orchestrate existing modules:

- `src/transformer_mood/main.py`
- `src/transformer_mood/speech_emotion_classifier.py`

No shell or batch wrapper files are part of this design.

## Command Model

### Default Behavior

Running `python run.py` with no subcommand will be equivalent to:

```bash
python run.py webui
```

This matches the most common project use case and makes the repository easier to try immediately after cloning.

### Supported Subcommands

The launcher will provide these subcommands:

#### `webui`

Starts the FastAPI WebUI.

Supported options:

- `--host`
- `--port`
- `--model PATH`

Execution target:

- `transformer_mood.main:app`

#### `train`

Runs the training flow through the existing classifier module.

The launcher should allow relevant arguments to pass through to the classifier module rather than re-implementing training arguments.

Execution target:

- `python -m transformer_mood.speech_emotion_classifier --mode train ...`

#### `predict`

Runs single-file prediction.

Minimum required option:

- `--audio PATH`

Optional:

- `--model PATH`

Execution target:

- `python -m transformer_mood.speech_emotion_classifier --mode predict --audio ...`

#### `doctor`

Performs environment checks and prints a readable report without starting anything.

The report should include:

- operating system
- Python executable in use
- whether `.venv` exists
- whether required imports work
- whether `ffmpeg` is available
- whether `output/best_model.pth` exists
- whether `data/ravdess/` exists

## Platform Detection

`run.py` will detect the current platform with Python standard library facilities.

Required behavior:

- Windows uses `.venv\\Scripts\\python.exe`
- Linux uses `.venv/bin/python`

This is only for choosing the repository-managed virtual environment executable path. The launcher itself will still run from whatever Python was used to execute `run.py`.

## Environment Strategy

### Preferred Interpreter

The launcher will prefer the repository-local `.venv`.

Selection order:

1. If `.venv` exists and contains a usable Python interpreter, use it.
2. If `.venv` does not exist, create it using the current Python interpreter.
3. After creation, switch to the `.venv` interpreter for all project commands.

The launcher should not try to guess or adopt arbitrary global Python environments. This keeps behavior predictable.

### Dependency Checks

The launcher will verify the availability of these imports inside the chosen environment:

- `fastapi`
- `uvicorn`
- `torch`
- `torchaudio`
- `librosa`

If they are missing, the launcher will bootstrap them automatically.

### Installation Order

When dependencies are missing, the launcher will install in this order:

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

This matches the current documented setup flow and avoids a mismatch between the launcher and README.

The design intentionally avoids CUDA-specific installer branching. This version prioritizes reliability and simplicity over environment-specific optimization.

## Runtime Command Construction

The launcher will not duplicate application logic. It will construct subprocess commands that call the existing package modules with the correct environment.

Common subprocess rules:

- set `PYTHONPATH=src`
- use the selected `.venv` Python executable
- pass through arguments to the underlying module where possible

Examples of the intended internal command shape:

```bash
<venv-python> -m uvicorn transformer_mood.main:app --host 127.0.0.1 --port 8000
<venv-python> -m transformer_mood.speech_emotion_classifier --mode train
<venv-python> -m transformer_mood.speech_emotion_classifier --mode predict --audio path/to/file.wav
```

## Model Handling

### Default Model Path

The default checkpoint path remains:

- `output/best_model.pth`

### Override

The launcher will accept:

- `--model PATH`

If provided, it will set:

- `EMOTION_MODEL_PATH=<PATH>`

before starting the subprocess.

### Missing Model Behavior

The design distinguishes between commands that can degrade gracefully and commands that cannot.

#### For `webui`

If no checkpoint exists and `--model` is not provided:

- allow WebUI startup
- print a warning that the interface can open but inference requires a model

This matches the current application behavior, where health checks and page loading can still work without a model.

#### For `predict`

If no checkpoint exists and `--model` is not provided:

- exit with a clear error message

Prediction cannot proceed without a model, so silent fallback would be misleading.

## Dataset Handling

For `train`, the launcher will check that the dataset root exists:

- `data/ravdess/`

If it is missing:

- exit with a clear error
- tell the user where the dataset should be placed

The launcher will not attempt to download datasets automatically.

## ffmpeg Handling

The launcher will check whether `ffmpeg` is available in `PATH`.

Behavior:

- if `ffmpeg` exists, proceed normally
- if `ffmpeg` is missing, print a clear warning or error depending on the command

Command expectations:

- `webui`: warn, because the UI can still start but recorded/compressed audio handling may be limited
- `doctor`: report missing `ffmpeg`
- `train`: no hard dependency on `ffmpeg`
- `predict`: no hard dependency if the input is already a supported wav file, but the launcher should still report availability status rather than attempting to infer every audio case

The launcher will not try to install `ffmpeg` automatically.

## Logging And User Experience

The launcher will print concise, explicit status lines, for example:

- `[INFO] Detected platform: Linux`
- `[INFO] Using virtual environment: .venv`
- `[INFO] Creating virtual environment...`
- `[INFO] Installing dependencies...`
- `[WARN] ffmpeg not found in PATH`
- `[WARN] Model not found, WebUI will start in no-model mode`
- `[ERROR] Dataset not found: data/ravdess/`

The design goal is to make automated steps visible and failures actionable.

## Failure Policy

### Missing `.venv`

- create it automatically

### Missing Python packages

- install automatically

### Missing `ffmpeg`

- do not install automatically
- print clear instructions or warning

### Missing model for `webui`

- continue with warning

### Missing model for `predict`

- fail with error

### Missing dataset for `train`

- fail with error

### Any bootstrap or subprocess failure

- return a non-zero exit code
- show the failing command or reason

## Documentation Changes

After implementation, README usage examples should be updated so the primary documented entrypoint is `run.py`.

Examples that should appear after implementation:

```bash
python run.py
python run.py webui --host 127.0.0.1 --port 8000
python run.py train
python run.py predict --audio path/to/example.wav
python run.py doctor
```

The older direct module commands may still be mentioned for advanced use, but `run.py` should become the recommended path.

## Verification Requirements

Implementation is only complete if the following are verified:

1. `python run.py doctor` reports the environment correctly
2. `python run.py` launches the WebUI path by default
3. `python run.py webui` works on a machine with a valid `.venv`
4. `python run.py` can create `.venv` when it does not exist
5. missing model handling differs correctly between `webui` and `predict`
6. `python run.py train` fails clearly when `data/ravdess/` is missing
7. the launcher works on both Windows and Linux path conventions at the code level

## Non-Goals

This design deliberately does not include:

- downloading pretrained checkpoints
- downloading datasets
- GPU/CUDA autodetection for package index selection
- replacing the existing CLI module interface
- multiple wrapper scripts

## Success Criteria

- The project has one obvious entrypoint: `run.py`
- New users can run the project from a freshly cloned machine with fewer manual steps
- Existing users with a working `.venv` can keep using the local environment without disruption
- The launcher behaves predictably on both Windows and Linux
- WebUI, training, prediction, and diagnostics are accessible through one interface
