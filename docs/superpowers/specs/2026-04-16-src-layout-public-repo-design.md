# Src Layout And Public Repo Cleanup Design

## Goal

Reorganize the repository into a more standard public-facing Python project layout by moving business code into `src/transformer_mood/`, preserving current behavior, and tightening `.gitignore` so private, local, and generated artifacts are not prepared for the public GitHub repository.

## Current State

The repository is currently organized as a script-first project:

- `app_fastapi.py` is the FastAPI entry point at the repository root
- `speech_emotion_classifier.py` contains training, inference, model definitions, and path configuration at the repository root
- `templates/index.html` lives outside the Python code layout
- Root-level generated artifacts such as `best_model.pth`, `training_curves.png`, and `confusion_matrix.png` still exist
- `output/` exists as a mixed-purpose result directory
- `.gitignore` is minimal and does not yet reflect a public-repository cleanup policy

This works locally, but the shape is closer to a personal script repository than a polished public project.

## Target Layout

The repository will be reorganized into this shape:

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

data/
output/
docs/
README.md
README.zh.md
ENVIRONMENT.md
requirements.txt
requirements-webui.txt
.gitignore
```

Key principles:

- `src/transformer_mood/` becomes the single business-code package
- Root stays focused on project-level files, documentation, data directories, and outputs
- The reorganization is structural, not a logic rewrite
- Existing features continue to work after path and import updates

## File Move Strategy

The implementation will make these file-level changes:

- Move `app_fastapi.py` to `src/transformer_mood/main.py`
- Move `speech_emotion_classifier.py` to `src/transformer_mood/speech_emotion_classifier.py`
- Move `templates/index.html` to `src/transformer_mood/templates/index.html`
- Create `src/transformer_mood/__init__.py`
- Create `src/transformer_mood/static/.gitkeep`

This pass will not split `speech_emotion_classifier.py` into multiple modules. The file is large, but separating training, inference, and model definitions is a second-phase refactor and is intentionally out of scope here.

## Python Package Design

### Package Name

The package name will be `transformer_mood`, located at `src/transformer_mood/`.

Rationale:

- Avoids a generic package name like `app`
- Matches the project name already used in documentation and UI
- Looks more polished in a public repository

### Web Entry Point

The FastAPI application will move from a root script to a package entry point:

- Old import target: `app_fastapi:app`
- New import target: `transformer_mood.main:app`

Recommended startup command after the reorganization:

```bash
PYTHONPATH=src python -m uvicorn transformer_mood.main:app --host 127.0.0.1 --port 8000
```

### CLI Entry Point

The training and prediction script will remain module-driven, but invoked from the package:

```bash
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode train
PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode predict --audio path/to/example.wav
```

This keeps current behavior while making the repo structure more package-oriented.

## Import And Path Adjustments

### Intra-Package Imports

`main.py` will no longer import from a root-level sibling script. Imports will be updated to package-relative or package-qualified imports that work under `src/transformer_mood/`.

Expected direction:

- `main.py` imports model and inference helpers from `transformer_mood.speech_emotion_classifier`

### Filesystem Paths

Current code relies on `__file__` relative to the repository root. After the move, path resolution must still target the repository-level `data/` and `output/` directories.

The design requirement is:

- Package-local assets resolve inside `src/transformer_mood/`:
  - `templates/`
  - `static/`
- Repository-level artifacts resolve from the package back to the repo root:
  - `data/`
  - `output/`

This means path logic in `speech_emotion_classifier.py` and `main.py` must be updated carefully so the move is transparent to users.

## Template And Static Asset Layout

The HTML template will move into the package:

- From: `templates/index.html`
- To: `src/transformer_mood/templates/index.html`

`main.py` will load templates from the package-local `templates/` directory.

`src/transformer_mood/static/` will be created now as an empty standard directory, even if there are no static files yet. It exists to make future frontend additions fit the layout without further structural churn.

## Public Repository Cleanup Policy

The repository is being prepared for a public GitHub upload. The cleanup policy is:

- Keep source code and docs
- Keep data placement instructions
- Ignore local environments and caches
- Ignore raw datasets
- Ignore models, training visualizations, and generated outputs
- Keep output directory intent visible without committing generated artifacts

## `.gitignore` Policy

The updated `.gitignore` will cover these categories.

### Local Environment And Python Cache

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `*.pyc`
- `*.pyo`
- `*.log`

### Local Tooling

- `.playwright-mcp/`

### Dataset

- `data/ravdess/`

### Generated Model And Training Outputs

- Root-level legacy artifacts:
  - `best_model.pth`
  - `training_curves.png`
  - `confusion_matrix.png`
- Generated content inside `output/`

The design expectation is that `output/` remains in the repo as a directory boundary, but its generated contents are ignored. A placeholder file such as `.gitkeep` may be used if needed to retain the directory.

### Editor And OS Noise

- `.DS_Store`
- `.idea/`
- `.vscode/`

## Documentation Changes

Documentation must be updated to match the new package layout and public-repo posture.

### README Files

These files must be updated:

- `README.md`
- `README.zh.md`

Required updates:

- Repository layout snippet reflects `src/transformer_mood/`
- Web startup command uses `PYTHONPATH=src` and `transformer_mood.main:app`
- CLI examples use `python -m transformer_mood.speech_emotion_classifier`
- References to template location or root-level scripts are corrected

### ENVIRONMENT.md

`ENVIRONMENT.md` must also be updated so setup instructions do not contradict the new structure.

### Data README Files

`data/README.md` and `data/README.zh.md` do not need structural changes unless a path reference becomes inconsistent. Since `data/ravdess/` remains unchanged, these files are expected to stay nearly the same.

## Verification Requirements

The reorganization is only successful if these checks pass after implementation:

1. Python package imports work from `src/transformer_mood/`
2. FastAPI can start using:

```bash
PYTHONPATH=src python -m uvicorn transformer_mood.main:app --host 127.0.0.1 --port 8000
```

3. README commands and repository layout snippets match the actual repo state
4. Template loading works from the new package-local template directory
5. Model, `data/`, and `output/` paths still resolve correctly after the move
6. `.gitignore` excludes the intended dataset, caches, models, training images, and generated outputs

## Non-Goals

This design does not include:

- Splitting `speech_emotion_classifier.py` into smaller modules
- Changing model architecture or training behavior
- Adding packaging metadata such as `pyproject.toml`
- Publishing to PyPI
- Adding tests beyond what is needed to verify the reorganization
- Any new product functionality

## Risks And Mitigations

### Risk: Path resolution breaks after moving files

Mitigation:

- Update repository-root path calculations explicitly rather than relying on old root-relative assumptions
- Verify `data/` and `output/` access after the move

### Risk: Startup commands in docs become stale

Mitigation:

- Update `README.md`, `README.zh.md`, and `ENVIRONMENT.md` in the same change
- Verify the final startup command against the new layout

### Risk: Generated artifacts are accidentally prepared for public upload

Mitigation:

- Expand `.gitignore` before final cleanup
- Explicitly ignore `output/` generated content and legacy root-level artifacts

### Risk: Scope creep into a full module refactor

Mitigation:

- Keep this pass focused on moving files, updating imports and paths, and cleaning the repo for publication
- Defer further decomposition of `speech_emotion_classifier.py` to a future refactor

## Success Criteria

- Business code lives under `src/transformer_mood/`
- FastAPI entry point is `transformer_mood.main:app`
- Templates live under `src/transformer_mood/templates/`
- A standard `static/` directory exists under the package
- Root-level script-style business files are removed
- Public-unfriendly generated artifacts are covered by `.gitignore`
- README and environment docs match the new structure and commands
- The application still starts successfully after the move
