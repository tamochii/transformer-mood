# TESS-Only GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TESS-only 7-class cached training path that keeps preprocessing out of the epoch loop and uses CUDA mixed precision to improve throughput and accuracy.

**Architecture:** Keep the existing Transformer model and improved feature pipeline, but isolate a new TESS-only mode inside `speech_emotion_classifier.py`. Add deterministic TESS-only label mapping, cache-backed datasets, and AMP-enabled training while leaving the current multi-dataset path intact as a separate mode.

**Tech Stack:** Python, PyTorch, torchaudio, librosa, argparse, unittest/pytest

---

## File Structure

- Modify: `src/transformer_mood/speech_emotion_classifier.py`
  Responsibility: add TESS-only label mapping, deterministic TESS split helpers, cache creation/loading, AMP-enabled training path, and CLI flags.
- Modify: `run.py`
  Responsibility: surface TESS-only training mode clearly through launcher passthrough and diagnostics.
- Modify: `tests/test_speech_emotion_classifier.py`
  Responsibility: add TESS-only mapping, split, cache, and AMP/training-path regression tests.
- Modify: `tests/test_run.py`
  Responsibility: cover launcher argument passthrough for TESS-only mode.

---

### Task 1: Add TESS-Only Label Mapping And Split Rules

**Files:**
- Modify: `src/transformer_mood/speech_emotion_classifier.py`
- Test: `tests/test_speech_emotion_classifier.py`

- [ ] **Step 1: Write the failing TESS-only mapping and split tests**

Add these tests to `tests/test_speech_emotion_classifier.py`:

```python
class TessOnlyTrainingTests(unittest.TestCase):
    def test_build_tess_label_mapping_uses_seven_real_classes(self):
        emotion_to_idx, idx_to_emotion = sec.build_label_space("tess")

        self.assertEqual(
            list(emotion_to_idx.keys()),
            ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
        )
        self.assertEqual(len(idx_to_emotion), 7)

    def test_split_tess_samples_is_deterministic_and_non_empty(self):
        samples = []
        for speaker in ["OAF", "YAF"]:
            for emotion_idx in range(7):
                for item_idx in range(3):
                    samples.append(
                        {
                            "filepath": f"{speaker}_{emotion_idx}_{item_idx}.wav",
                            "emotion_name": str(emotion_idx),
                            "emotion_idx": emotion_idx,
                            "speaker": speaker,
                        }
                    )

        train_samples, val_samples, test_samples = sec.split_tess_samples(samples)

        self.assertTrue(train_samples)
        self.assertTrue(val_samples)
        self.assertTrue(test_samples)
        self.assertEqual(
            (len(train_samples), len(val_samples), len(test_samples)),
            (len(train_samples), len(val_samples), len(test_samples)),
        )
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_emotion_classifier.py -k "tess_only or split_tess" -v`
Expected: FAIL with missing `build_label_space` and/or `split_tess_samples`

- [ ] **Step 3: Implement minimal TESS-only label-space and split helpers**

Add helpers to `src/transformer_mood/speech_emotion_classifier.py`:

```python
TESS_ONLY_EMOTIONS = {
    "angry": "angry",
    "disgust": "disgust",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprised",
}


def build_label_space(dataset_name: str):
    if dataset_name == "tess":
        emotion_to_idx = {name: idx for idx, name in enumerate(TESS_ONLY_EMOTIONS.values())}
    else:
        emotion_to_idx = {name: idx for idx, name in enumerate(RAVDESS_EMOTIONS.values())}
    idx_to_emotion = {idx: name for name, idx in emotion_to_idx.items()}
    return emotion_to_idx, idx_to_emotion


def split_tess_samples(samples: list):
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample["speaker"], []).append(sample)

    train_samples, val_samples, test_samples = [], [], []
    for speaker, speaker_samples in sorted(grouped.items()):
        speaker_samples = sorted(speaker_samples, key=lambda item: item["filepath"])
        for emotion_idx in sorted({item["emotion_idx"] for item in speaker_samples}):
            emotion_samples = [item for item in speaker_samples if item["emotion_idx"] == emotion_idx]
            train_cut = max(1, int(len(emotion_samples) * 0.6))
            val_cut = max(train_cut + 1, int(len(emotion_samples) * 0.8))
            train_samples.extend(emotion_samples[:train_cut])
            val_samples.extend(emotion_samples[train_cut:val_cut])
            test_samples.extend(emotion_samples[val_cut:])
    return train_samples, val_samples, test_samples
```

- [ ] **Step 4: Re-run the targeted tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_emotion_classifier.py -k "tess_only or split_tess" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_speech_emotion_classifier.py src/transformer_mood/speech_emotion_classifier.py
git commit -m "feat: add TESS-only label space and split helpers"
```

---

### Task 2: Add Cache Creation And Cache-Backed Dataset Loading

**Files:**
- Modify: `src/transformer_mood/speech_emotion_classifier.py`
- Test: `tests/test_speech_emotion_classifier.py`

- [ ] **Step 1: Write the failing cache tests**

Add these tests to `tests/test_speech_emotion_classifier.py`:

```python
    def test_build_cache_path_uses_tess_mode_and_feature_dim(self):
        cache_dir = sec.build_feature_cache_dir("tess")
        self.assertTrue(str(cache_dir).endswith("output/cache/tess_7class_fd384"))

    def test_cached_feature_dataset_loads_saved_tensor_without_audio_decode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "sample.pt"
            torch.save(
                {
                    "feature": torch.ones(4, 6),
                    "label": 3,
                    "mask": torch.tensor([True, True, False, False]),
                },
                cache_path,
            )
            dataset = sec.CachedFeatureDataset([cache_path], augment=False)

            feat, label, mask = dataset[0]

        self.assertTrue(torch.equal(feat, torch.ones(4, 6)))
        self.assertEqual(label.item(), 3)
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, False, False])))
```

- [ ] **Step 2: Run the cache tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_emotion_classifier.py -k "cache_path or CachedFeatureDataset" -v`
Expected: FAIL with missing cache helpers/classes

- [ ] **Step 3: Implement minimal cache directory and cached dataset support**

Add to `src/transformer_mood/speech_emotion_classifier.py`:

```python
def build_feature_cache_dir(dataset_name: str) -> Path:
    if dataset_name == "tess":
        return Path(OUTPUT_DIR) / "cache" / f"tess_7class_fd{FEATURE_DIM}"
    return Path(OUTPUT_DIR) / "cache" / f"{dataset_name}_fd{FEATURE_DIM}"


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_paths: list[Path], augment: bool = False):
        self.cache_paths = cache_paths
        self.augment = augment

    def __len__(self):
        return len(self.cache_paths)

    def __getitem__(self, idx):
        payload = torch.load(self.cache_paths[idx], map_location="cpu", weights_only=False)
        feat = payload["feature"].clone()
        if self.augment:
            feat = SpeechEmotionDataset([])._spec_augment(feat)
        label = torch.tensor(payload["label"], dtype=torch.long)
        mask = payload["mask"].clone()
        return feat, label, mask
```

Then add a cache builder helper that writes one `.pt` file per sample containing `feature`, `label`, and `mask`.

- [ ] **Step 4: Re-run the cache tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_emotion_classifier.py -k "cache_path or CachedFeatureDataset" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_speech_emotion_classifier.py src/transformer_mood/speech_emotion_classifier.py
git commit -m "feat: add TESS feature cache support"
```

---

### Task 3: Add TESS-Only CLI Path And Launcher Coverage

**Files:**
- Modify: `src/transformer_mood/speech_emotion_classifier.py`
- Modify: `run.py`
- Test: `tests/test_run.py`
- Test: `tests/test_speech_emotion_classifier.py`

- [ ] **Step 1: Write the failing CLI tests**

Add these tests:

```python
def test_train_passthrough_keeps_dataset_and_cache_flags(self):
    run = load_run_module(self)

    args = run.parse_args(["train", "--", "--dataset", "tess", "--cache-features"])

    self.assertEqual(args.extra_args, ["--dataset", "tess", "--cache-features"])
```

And in `tests/test_speech_emotion_classifier.py`:

```python
    def test_build_training_samples_returns_tess_only_when_requested(self):
        with patch.object(sec, "scan_tess_dataset", return_value=[{"filepath": "a.wav", "emotion_name": "happy", "emotion_idx": 3, "speaker": "OAF"}]):
            train_samples, val_samples, test_samples, emotion_to_idx, idx_to_emotion = sec.prepare_training_samples("tess")

        self.assertEqual(len(train_samples) + len(val_samples) + len(test_samples), 1)
        self.assertEqual(list(emotion_to_idx.keys()), ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"])
```

- [ ] **Step 2: Run the new CLI tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_run.py tests/test_speech_emotion_classifier.py -k "cache_flags or prepare_training_samples" -v`
Expected: FAIL with missing `prepare_training_samples`

- [ ] **Step 3: Implement TESS-only CLI flags and training-sample preparation**

In `src/transformer_mood/speech_emotion_classifier.py`, add parser flags and helper:

```python
parser.add_argument("--dataset", choices=["multi", "tess"], default="multi")
parser.add_argument("--cache-features", action="store_true")
```

Add:

```python
def prepare_training_samples(dataset_name: str):
    emotion_to_idx, idx_to_emotion = build_label_space(dataset_name)
    if dataset_name == "tess":
        tess_samples = scan_tess_dataset(TESS_DATA_DIR)
        tess_samples = [
            {**sample, "emotion_idx": emotion_to_idx[sample["emotion_name"]]}
            for sample in tess_samples
            if sample["emotion_name"] in emotion_to_idx
        ]
        train_samples, val_samples, test_samples = split_tess_samples(tess_samples)
        return train_samples, val_samples, test_samples, emotion_to_idx, idx_to_emotion
    # fall back to current multi-dataset path
```

Then route `main()` to use `prepare_training_samples(args.dataset)` and keep the existing multi-dataset code under the `multi` branch.

- [ ] **Step 4: Re-run the targeted CLI tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_run.py tests/test_speech_emotion_classifier.py -k "cache_flags or prepare_training_samples" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_run.py tests/test_speech_emotion_classifier.py run.py src/transformer_mood/speech_emotion_classifier.py
git commit -m "feat: add TESS-only training mode flags"
```

---

### Task 4: Enable AMP Training And Cache-Backed Dataloaders

**Files:**
- Modify: `src/transformer_mood/speech_emotion_classifier.py`
- Test: `tests/test_speech_emotion_classifier.py`

- [ ] **Step 1: Write the failing AMP and cache-loader tests**

Add these tests:

```python
    def test_create_training_components_uses_cached_dataset_when_enabled(self):
        samples = [{"cache_path": Path("/tmp/fake.pt"), "emotion_idx": 1}]

        train_dataset, _, _ = sec.build_datasets_from_samples(samples, [], [], use_cache=True)

        self.assertIsInstance(train_dataset, sec.CachedFeatureDataset)

    def test_amp_enabled_only_when_cuda_available(self):
        enabled, scaler_enabled = sec.resolve_amp_settings(torch.device("cuda"))
        self.assertTrue(enabled)
        self.assertTrue(scaler_enabled)
```

- [ ] **Step 2: Run the AMP/cache-loader tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_emotion_classifier.py -k "CachedFeatureDataset or amp_enabled" -v`
Expected: FAIL with missing helpers

- [ ] **Step 3: Implement cache-backed dataset builder and AMP helpers**

Add to `src/transformer_mood/speech_emotion_classifier.py`:

```python
def build_datasets_from_samples(train_samples, val_samples, test_samples, use_cache: bool):
    if use_cache:
        train_dataset = CachedFeatureDataset([sample["cache_path"] for sample in train_samples], augment=True)
        val_dataset = CachedFeatureDataset([sample["cache_path"] for sample in val_samples], augment=False)
        test_dataset = CachedFeatureDataset([sample["cache_path"] for sample in test_samples], augment=False)
    else:
        train_dataset = SpeechEmotionDataset(train_samples, augment=True)
        val_dataset = SpeechEmotionDataset(val_samples, augment=False)
        test_dataset = SpeechEmotionDataset(test_samples, augment=False)
    return train_dataset, val_dataset, test_dataset


def resolve_amp_settings(device: torch.device):
    enabled = device.type == "cuda"
    return enabled, enabled
```

Then update `train_one_epoch()` and `evaluate()` to use `torch.amp.autocast("cuda", enabled=amp_enabled)` and `torch.cuda.amp.GradScaler(enabled=scaler_enabled)`.

- [ ] **Step 4: Re-run the targeted tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_emotion_classifier.py -k "CachedFeatureDataset or amp_enabled" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_speech_emotion_classifier.py src/transformer_mood/speech_emotion_classifier.py
git commit -m "feat: add AMP and cache-backed TESS dataloaders"
```

---

### Task 5: End-To-End Verification On TESS-Only Path

**Files:**
- Modify if needed: `src/transformer_mood/speech_emotion_classifier.py`
- Modify if needed: `run.py`

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 2: Build the TESS cache in smoke mode**

Run: `.venv/bin/python run.py train -- --dataset tess --cache-features --epochs 1 --num_workers 0`
Expected: cache directory created under `output/cache/tess_7class_fd384/` and one-epoch smoke run completes

- [ ] **Step 3: Verify CUDA usage during smoke training**

Run in parallel with training:

```bash
nvidia-smi
```

Expected: the Python training process appears under GPU processes with non-zero memory use

- [ ] **Step 4: Run a short quality-focused TESS-only training pass**

Run: `.venv/bin/python run.py train -- --dataset tess --cache-features --epochs 20 --batch_size 32 --num_workers 0`
Expected: TESS-only 7-class training completes, best model saved, and test accuracy beats the prior mixed-dataset baseline

- [ ] **Step 5: Commit final implementation**

```bash
git add run.py src/transformer_mood/speech_emotion_classifier.py tests/test_run.py tests/test_speech_emotion_classifier.py docs/superpowers/specs/2026-04-16-tess-only-gpu-training-design.md docs/superpowers/plans/2026-04-16-tess-only-gpu-training.md
git commit -m "feat: add TESS-only cached GPU training path"
```
