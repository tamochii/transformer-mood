# TESS-Only GPU Training Design

**Date:** 2026-04-16
**Status:** Draft approved in chat
**Goal:** Switch the project to a TESS-only 7-class training path, move repeated preprocessing out of the epoch loop, and improve training quality by keeping GPU utilization high and label semantics clean.

## Problem

The current training flow mixes multiple datasets with different speaker timbres, recording conditions, and label coverage. This creates domain mismatch and unstable evaluation. It also performs feature extraction online during training, which pushes heavy audio decoding and spectral preprocessing onto CPU and can make the machine feel frozen even when CUDA is available.

For the new target, the user wants:
- TESS only
- better accuracy
- more real GPU usage during training
- no ambiguous 8-class setup when TESS only contains 7 actual emotions

## Dataset Scope

Training uses only `data/tess/`.

TESS label space becomes 7 classes:
- angry
- disgust
- fearful
- happy
- neutral
- sad
- surprised

TESS directory labels `pleasant_surprise` and `pleasant_surprised` both map to `surprised`.

The existing 8-class global mapping is not reused for TESS-only mode. This avoids carrying a fake `calm` class that TESS does not contain.

## Data Splitting

TESS samples are grouped by speaker family (`OAF`, `YAF`) and split deterministically.

Because TESS has limited speaker variety, the split should avoid two failure modes:
- leakage from the same speaker family across train and test
- a degenerate validation set with zero samples for some classes

The TESS-only path will:
- scan all TESS samples
- build speaker-aware sample groups
- create stable train, validation, and test subsets for TESS-only evaluation
- keep the split deterministic via the existing seed

## Feature Pipeline

The feature pipeline remains aligned with the current improved path:
- load waveform
- resample to 16 kHz
- build Mel spectrogram
- apply per-feature-bin CMVN across time
- append delta and delta-delta features
- truncate or pad to fixed sequence length

Training-only augmentations remain available for train split only:
- waveform augmentation
- SpecAugment
- Mixup at batch level

## Caching Strategy

The main change is to separate preprocessing from model training.

Add a cacheable TESS preprocessing phase:
- first run scans all TESS audio files
- computes base features for each sample
- saves features and metadata to `output/cache/tess_7class/`

Cache contents should include enough information to avoid recomputing CPU-heavy steps inside each epoch:
- feature tensor before batch assembly
- label index
- speaker identifier
- sequence length or mask information needed for padding

The cache is keyed by dataset mode and feature settings so future changes do not silently reuse stale artifacts.

During training, the dataset reads cached tensors instead of raw wav files. This removes repeated audio decode, torchaudio transform, and librosa delta work from each epoch.

## Training Path

Add a TESS-only training mode controlled from CLI.

Recommended CLI shape:
- `--dataset tess`
- `--cache-features`

The TESS-only path will:
- build 7-class label mapping
- load TESS samples only
- optionally create or reuse cached features
- train the Transformer on cached tensors

GPU-focused training changes:
- enable AMP mixed precision for forward/backward pass
- keep `pin_memory=True` when using CUDA
- keep `num_workers` conservative by default to avoid freezing the machine
- preserve weighted sampling, label smoothing, learning-rate warmup, cosine decay, and Mixup
- allow batch size tuning for GPU throughput

## Accuracy Strategy

Accuracy improvement comes from three concrete changes:
- remove multi-dataset domain noise by using TESS only
- remove fake 8-class evaluation mismatch by training true 7-class TESS labels
- remove epoch-time CPU preprocessing bottlenecks so GPU training is steadier and easier to scale

The expectation is not just faster epochs. The more important effect is cleaner supervision and more stable optimization.

## Files To Change

Primary changes are expected in:
- `src/transformer_mood/speech_emotion_classifier.py`
- `run.py`
- `tests/test_speech_emotion_classifier.py`
- `tests/test_run.py`

No broad refactor is planned outside the TESS-only path.

## Verification

Verification must include:
- unit tests for TESS-only label mapping and split behavior
- unit tests for cache creation and cache-backed dataset loading
- unit tests for CLI argument handling
- full existing test suite passing
- one smoke training run in TESS-only mode
- evidence that CUDA is used during training

## Non-Goals

This design does not include:
- switching to a pretrained audio backbone
- redesigning the Transformer architecture
- keeping multi-dataset training as the default optimized path for this round

## Success Criteria

The work is successful when:
- TESS-only training runs end-to-end through the launcher
- the project trains on 7 real TESS classes only
- preprocessing is cached and not recomputed every epoch
- training uses CUDA with mixed precision when available
- the machine remains responsive compared with the old CPU-heavy online preprocessing path
- test accuracy on TESS-only evaluation improves over the prior mixed-dataset setup
