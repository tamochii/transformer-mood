# Tess Vec Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the `--dataset tess` CLI contract unchanged while internally training from `data/vec/` with a 6-class label space and leakage-safe splitting for augmented samples.

**Architecture:** Reuse the existing `tess` mode entrypoint and swap only its internal dataset source and split logic. Add a small vec-specific scanning/parsing layer in `speech_emotion_classifier.py`, keep `multi` mode untouched, and update launcher validation plus docs so the external workflow matches the new internal behavior.

**Tech Stack:** Python, unittest, pathlib/os, existing training pipeline in `src/transformer_mood/speech_emotion_classifier.py`

---

### Task 1: Add failing vec-mode tests

**Files:**
- Modify: `tests/test_speech_emotion_classifier.py`
- Modify: `tests/test_run.py`

- [ ] **Step 1: Write the failing tests**

```python
class VecBackedTessTests(unittest.TestCase):
    def test_build_tess_label_mapping_uses_six_vec_classes(self):
        emotion_to_idx, idx_to_emotion = sec.build_label_space("tess")
        self.assertEqual(
            list(emotion_to_idx.keys()),
            ["angry", "disgust", "fearful", "happy", "neutral", "sad"],
        )
        self.assertNotIn("surprised", emotion_to_idx)
        self.assertEqual(len(idx_to_emotion), 6)

    def test_scan_vec_dataset_maps_directory_names_and_speakers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative in [
                "anger/OAF_back_angry.wav",
                "fear/1001_IEO_FEA_XX.wav",
                "happy/03-01-03-01-01-01-19.wav",
            ]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            samples = sec.scan_vec_dataset(str(root))

        by_name = {Path(sample["filepath"]).name: sample for sample in samples}
        self.assertEqual(by_name["OAF_back_angry.wav"]["emotion_name"], "angry")
        self.assertEqual(by_name["OAF_back_angry.wav"]["speaker"], "OAF")
        self.assertEqual(by_name["1001_IEO_FEA_XX.wav"]["emotion_name"], "fearful")
        self.assertEqual(by_name["1001_IEO_FEA_XX.wav"]["speaker"], "1001")
        self.assertEqual(by_name["03-01-03-01-01-01-19.wav"]["speaker"], "19")

    def test_vec_group_id_strips_augmentation_suffix(self):
        base = sec.build_vec_group_id("/tmp/anger/YAF_white_angry.wav")
        aug = sec.build_vec_group_id("/tmp/anger/YAF_white_angry_aug3.wav")
        self.assertEqual(base, aug)

    def test_split_tess_samples_keeps_augmented_group_together(self):
        samples = [
            {"filepath": "s1_g1.wav", "emotion_name": "happy", "emotion_idx": 3, "speaker": "OAF", "group_id": "g1"},
            {"filepath": "s1_g1_aug2.wav", "emotion_name": "happy", "emotion_idx": 3, "speaker": "OAF", "group_id": "g1"},
            {"filepath": "s1_g2.wav", "emotion_name": "happy", "emotion_idx": 3, "speaker": "OAF", "group_id": "g2"},
            {"filepath": "s1_g3.wav", "emotion_name": "happy", "emotion_idx": 3, "speaker": "OAF", "group_id": "g3"},
            {"filepath": "s1_g4.wav", "emotion_name": "happy", "emotion_idx": 3, "speaker": "OAF", "group_id": "g4"},
        ]

        train_samples, val_samples, test_samples = sec.split_tess_samples(samples)

        buckets = {
            "train": {sample["filepath"] for sample in train_samples},
            "val": {sample["filepath"] for sample in val_samples},
            "test": {sample["filepath"] for sample in test_samples},
        }
        placements = [name for name, paths in buckets.items() if {"s1_g1.wav", "s1_g1_aug2.wav"} & paths]
        self.assertEqual(len(placements), 1)
        self.assertTrue({"s1_g1.wav", "s1_g1_aug2.wav"}.issubset(buckets[placements[0]]))

    def test_split_tess_samples_keeps_three_splits_non_empty(self):
        samples = []
        for speaker in ["OAF", "YAF"]:
            for index in range(5):
                samples.append(
                    {
                        "filepath": f"{speaker}_group_{index}.wav",
                        "emotion_name": "happy",
                        "emotion_idx": 3,
                        "speaker": speaker,
                        "group_id": f"group_{index}",
                    }
                )

        train_samples, val_samples, test_samples = sec.split_tess_samples(samples)

        self.assertTrue(train_samples)
        self.assertTrue(val_samples)
        self.assertTrue(test_samples)

    def test_prepare_training_samples_uses_vec_scan_for_tess(self):
        fake_samples = [
            {"filepath": "anger/OAF_a.wav", "emotion_name": "angry", "speaker": "OAF", "group_id": "g1"},
            {"filepath": "happy/OAF_b.wav", "emotion_name": "happy", "speaker": "OAF", "group_id": "g2"},
            {"filepath": "sad/YAF_c.wav", "emotion_name": "sad", "speaker": "YAF", "group_id": "g3"},
            {"filepath": "neutral/YAF_d.wav", "emotion_name": "neutral", "speaker": "YAF", "group_id": "g4"},
            {"filepath": "fear/YAF_e.wav", "emotion_name": "fearful", "speaker": "YAF", "group_id": "g5"},
        ]
        with patch.object(sec, "scan_vec_dataset", return_value=fake_samples) as scan_mock, patch.object(
            sec, "scan_tess_dataset", side_effect=AssertionError("legacy tess scanner should not be used")
        ):
            train_samples, val_samples, test_samples, emotion_to_idx, idx_to_emotion = sec.prepare_training_samples("tess")

        scan_mock.assert_called_once_with(sec.VEC_DATA_DIR)
        self.assertEqual(len(train_samples) + len(val_samples) + len(test_samples), 5)
        self.assertEqual(list(emotion_to_idx.keys()), ["angry", "disgust", "fearful", "happy", "neutral", "sad"])
        self.assertEqual(idx_to_emotion[2], "fearful")

class RunValidationTests(unittest.TestCase):
    def test_train_requires_vec_directory_when_tess_dataset_requested(self):
        run = load_run_module(self)
        result = run.validate_command_requirements(
            command="train",
            repo_root=self.repo_root,
            explicit_model=None,
            ffmpeg_path="/usr/bin/ffmpeg",
            train_args=["--dataset", "tess"],
        )
        self.assertTrue(result.errors)
        self.assertIn("data/vec", result.errors[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m unittest tests.test_speech_emotion_classifier tests.test_run`
Expected: FAIL because `scan_vec_dataset`, vec grouping helpers, new `validate_command_requirements(..., train_args=...)`, and the 6-class tess behavior do not exist yet.

### Task 2: Implement vec-backed tess scanning and splitting

**Files:**
- Modify: `src/transformer_mood/speech_emotion_classifier.py`
- Test: `tests/test_speech_emotion_classifier.py`

- [ ] **Step 1: Add minimal vec constants and parsing helpers**

```python
import re

VEC_DATA_DIR = str(PROJECT_ROOT / "data" / "vec")
VEC_EMOTIONS = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}
VEC_ONLY_EMOTIONS = {
    "angry": "angry",
    "disgust": "disgust",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}
AUGMENTED_SUFFIX_RE = re.compile(r"_aug\d+$")


def strip_augmented_suffix(filepath: str) -> str:
    stem = Path(filepath).stem
    return AUGMENTED_SUFFIX_RE.sub("", stem)


def build_vec_group_id(filepath: str) -> str:
    return strip_augmented_suffix(filepath)


def parse_vec_speaker(filepath: str) -> str:
    stem = strip_augmented_suffix(filepath)
    if stem.startswith(("OAF_", "YAF_")):
        return stem.split("_", 1)[0]
    if "_" in stem:
        return stem.split("_", 1)[0]
    if "-" in stem:
        return stem.split("-")[-1]
    return stem
```

- [ ] **Step 2: Implement `scan_vec_dataset()` and switch tess label space to six classes**

```python
def build_label_space(dataset_name: str):
    if dataset_name == "tess":
        emotion_to_idx = {name: idx for idx, name in enumerate(VEC_ONLY_EMOTIONS.values())}
    else:
        emotion_to_idx = {name: idx for idx, name in enumerate(RAVDESS_EMOTIONS.values())}
    idx_to_emotion = {idx: name for name, idx in emotion_to_idx.items()}
    return emotion_to_idx, idx_to_emotion


def scan_vec_dataset(data_dir: str) -> list:
    samples = []
    if not os.path.isdir(data_dir):
        return samples
    for emotion_dir in sorted(os.listdir(data_dir)):
        emotion_path = os.path.join(data_dir, emotion_dir)
        if not os.path.isdir(emotion_path):
            continue
        emotion_name = VEC_EMOTIONS.get(emotion_dir.lower())
        if emotion_name is None:
            continue
        for wav_file in sorted(os.listdir(emotion_path)):
            if not wav_file.endswith(".wav"):
                continue
            filepath = os.path.join(emotion_path, wav_file)
            samples.append(
                {
                    "filepath": filepath,
                    "emotion_name": emotion_name,
                    "speaker": parse_vec_speaker(filepath),
                    "group_id": build_vec_group_id(filepath),
                }
            )
    return samples
```

- [ ] **Step 3: Replace tess split with group-safe splitting and wire `prepare_training_samples()` to vec**

```python
def _split_sorted_groups(groups: list[list[dict]]):
    if not groups:
        return [], [], []
    if len(groups) == 1:
        return groups[0], [], []
    if len(groups) == 2:
        return groups[0], groups[1], []

    train_end = max(1, int(len(groups) * 0.6))
    val_end = max(train_end + 1, int(len(groups) * 0.8))
    val_end = min(val_end, len(groups) - 1)
    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]
    return [item for group in train_groups for item in group], [item for group in val_groups for item in group], [item for group in test_groups for item in group]


def split_tess_samples(samples: list):
    grouped_by_speaker = {}
    for sample in samples:
        speaker_groups = grouped_by_speaker.setdefault(sample["speaker"], {})
        speaker_groups.setdefault(sample["group_id"], []).append(sample)

    train_samples, val_samples, test_samples = [], [], []
    for speaker in sorted(grouped_by_speaker):
        groups = [
            sorted(group, key=lambda item: item["filepath"])
            for _, group in sorted(grouped_by_speaker[speaker].items())
        ]
        speaker_train, speaker_val, speaker_test = _split_sorted_groups(groups)
        train_samples.extend(speaker_train)
        val_samples.extend(speaker_val)
        test_samples.extend(speaker_test)
    return train_samples, val_samples, test_samples


def prepare_training_samples(dataset_name: str):
    emotion_to_idx, idx_to_emotion = build_label_space(dataset_name)
    if dataset_name == "tess":
        vec_samples = []
        for sample in scan_vec_dataset(VEC_DATA_DIR):
            if sample["emotion_name"] not in emotion_to_idx:
                continue
            vec_samples.append({**sample, "emotion_idx": emotion_to_idx[sample["emotion_name"]]})
        train_samples, val_samples, test_samples = split_tess_samples(vec_samples)
        return train_samples, val_samples, test_samples, emotion_to_idx, idx_to_emotion
```

- [ ] **Step 4: Update cache naming to avoid stale 7-class reuse**

```python
def build_feature_cache_dir(dataset_name: str) -> Path:
    if dataset_name == "tess":
        return Path(OUTPUT_DIR) / "cache" / f"tess_vec_6class_fd{FEATURE_DIM}"
    return Path(OUTPUT_DIR) / "cache" / f"{dataset_name}_fd{FEATURE_DIM}"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/bin/python -m unittest tests.test_speech_emotion_classifier`
Expected: PASS

### Task 3: Align launcher validation and docs

**Files:**
- Modify: `run.py`
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `data/README.md`
- Modify: `data/README.zh.md`
- Test: `tests/test_run.py`

- [ ] **Step 1: Make `run.py` validate by requested dataset**

```python
def _resolve_train_dataset_dir(repo_root: Path, train_args: list[str] | None) -> tuple[Path, str]:
    requested_dataset = "multi"
    args = list(train_args or [])
    for index, value in enumerate(args[:-1]):
        if value == "--dataset":
            requested_dataset = args[index + 1]
            break
    if requested_dataset == "tess":
        return repo_root / "data" / "vec", "Place vec data under data/vec/ for --dataset tess."
    return repo_root / "data" / "ravdess", "Place RAVDESS under data/ravdess/."


def validate_command_requirements(command, repo_root, explicit_model, ffmpeg_path, train_args=None):
    ...
    if command == "train":
        result.dataset_path, hint = _resolve_train_dataset_dir(repo_root, train_args)
        if not result.dataset_path.is_dir():
            result.errors.append(f"Dataset not found: {result.dataset_path}. {hint}")
```

- [ ] **Step 2: Pass train args through dispatch/doctor output and update docs**

```python
def dispatch_command(args, repo_root, venv_python):
    validation = validate_command_requirements(
        command,
        repo_root,
        explicit_model,
        ffmpeg_path,
        train_args=getattr(args, "extra_args", None),
    )

# README wording update example
# - `python run.py train -- --dataset tess` now reads `data/vec/`
# - `tess` mode now trains 6 classes: angry, disgust, fearful, happy, neutral, sad
# - `multi` mode remains unchanged
```

- [ ] **Step 3: Run related tests**

Run: `./.venv/bin/python -m unittest tests.test_main tests.test_run tests.test_speech_emotion_classifier`
Expected: PASS

### Task 4: Final verification

**Files:**
- No code changes expected

- [ ] **Step 1: Run focused verification commands**

Run: `./.venv/bin/python -m unittest tests.test_speech_emotion_classifier`
Expected: PASS

Run: `./.venv/bin/python -m unittest tests.test_main tests.test_run tests.test_speech_emotion_classifier`
Expected: PASS

- [ ] **Step 2: Inspect worktree**

Run: `git status --short`
Expected: Modified source/tests/docs only; existing untracked `data/vec/` remains untouched.
