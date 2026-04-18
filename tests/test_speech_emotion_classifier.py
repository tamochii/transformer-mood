from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import transformer_mood.speech_emotion_classifier as sec


class _RecordingModel:
    def __init__(self):
        self.last_feat = None
        self.last_mask = None

    def eval(self):
        return self

    def __call__(self, feat, mask=None):
        self.last_feat = feat.detach().cpu()
        self.last_mask = mask.detach().cpu() if mask is not None else None
        logits = torch.zeros((feat.shape[0], sec.NUM_CLASSES), dtype=feat.dtype, device=feat.device)
        logits[:, 0] = 1.0
        return logits


class _RecordingModelSeven:
    def __init__(self):
        self.last_feat = None
        self.last_mask = None

    def eval(self):
        return self

    def __call__(self, feat, mask=None):
        self.last_feat = feat.detach().cpu()
        self.last_mask = mask.detach().cpu() if mask is not None else None
        logits = torch.zeros((feat.shape[0], 7), dtype=feat.dtype, device=feat.device)
        logits[:, 6] = 1.0
        return logits


class SpeechEmotionFeatureTests(unittest.TestCase):
    def test_dataset_handles_short_clips_without_delta_crash(self):
        dataset = sec.SpeechEmotionDataset(
            [{"filepath": "fake.wav", "emotion_idx": 2}],
            max_seq_len=4,
            augment=False,
        )
        mel_spec = torch.arange(1, sec.N_MELS * 2 + 1, dtype=torch.float32).reshape(sec.N_MELS, 2)
        cmvn = (mel_spec - mel_spec.mean(dim=1, keepdim=True)) / (mel_spec.std(dim=1, keepdim=True) + 1e-6)

        with patch.object(sec, "load_audio", return_value=torch.zeros(1, 1600)), patch.object(
            sec, "extract_mel_spectrogram", return_value=mel_spec
        ):
            feat, label, mask = dataset[0]

        self.assertEqual(feat.shape, (4, sec.FEATURE_DIM))
        self.assertTrue(torch.allclose(feat[:2, :sec.N_MELS], cmvn.T, atol=1e-5))
        self.assertTrue(torch.equal(feat[:2, sec.N_MELS:], torch.zeros(2, sec.N_MELS * 2)))
        self.assertEqual(label.item(), 2)
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, False, False])))

    def test_dataset_applies_cmvn_delta_and_padding(self):
        dataset = sec.SpeechEmotionDataset(
            [{"filepath": "fake.wav", "emotion_idx": 3}],
            max_seq_len=5,
            augment=False,
        )
        mel_spec = torch.arange(1, sec.N_MELS * 3 + 1, dtype=torch.float32).reshape(sec.N_MELS, 3)
        cmvn = (mel_spec - mel_spec.mean(dim=1, keepdim=True)) / (mel_spec.std(dim=1, keepdim=True) + 1e-6)
        delta1 = torch.full_like(cmvn, 10.0)
        delta2 = torch.full_like(cmvn, 20.0)
        expected_feat = torch.cat([cmvn, delta1, delta2], dim=0).T

        with patch.object(sec, "load_audio", return_value=torch.zeros(1, 1600)), patch.object(
            sec, "extract_mel_spectrogram", return_value=mel_spec
        ), patch.object(sec.librosa.feature, "delta", side_effect=[delta1.numpy(), delta2.numpy()]):
            feat, label, mask = dataset[0]

        self.assertEqual(feat.shape, (5, sec.FEATURE_DIM))
        self.assertTrue(torch.allclose(feat[:3], expected_feat, atol=1e-5))
        self.assertTrue(torch.equal(feat[3:], torch.zeros(2, sec.FEATURE_DIM)))
        self.assertEqual(label.item(), 3)
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, True, False, False])))

    def test_training_dataset_applies_spec_augment_after_feature_creation(self):
        dataset = sec.SpeechEmotionDataset(
            [{"filepath": "fake.wav", "emotion_idx": 1}],
            max_seq_len=4,
            augment=True,
        )
        mel_spec = torch.arange(1, sec.N_MELS * 4 + 1, dtype=torch.float32).reshape(sec.N_MELS, 4)
        delta = torch.zeros_like(mel_spec)
        augmented_feat = torch.full((4, sec.FEATURE_DIM), 7.0)

        with patch.object(sec, "load_audio", return_value=torch.zeros(1, 1600)), patch.object(
            sec, "extract_mel_spectrogram", return_value=mel_spec
        ), patch.object(sec.librosa.feature, "delta", side_effect=[delta.numpy(), delta.numpy()]), patch.object(
            dataset, "_augment", return_value=torch.ones(1, 1600)
        ) as augment_mock, patch.object(dataset, "_spec_augment", return_value=augmented_feat) as spec_mock:
            feat, _, _ = dataset[0]

        augment_mock.assert_called_once()
        spec_mock.assert_called_once()
        self.assertTrue(torch.equal(feat, augmented_feat))

    def test_spec_augment_masks_same_mel_bins_across_feature_groups(self):
        dataset = sec.SpeechEmotionDataset([], augment=True)
        feat = torch.arange(12 * sec.FEATURE_DIM, dtype=torch.float32).reshape(12, sec.FEATURE_DIM)

        with patch.object(sec.random, "randint", side_effect=[2, 1, 0]):
            augmented = dataset._spec_augment(feat)

        self.assertTrue(torch.equal(augmented[:, 1:3], torch.zeros(12, 2)))
        self.assertTrue(torch.equal(augmented[:, sec.N_MELS + 1:sec.N_MELS + 3], torch.zeros(12, 2)))
        self.assertTrue(
            torch.equal(augmented[:, 2 * sec.N_MELS + 1:2 * sec.N_MELS + 3], torch.zeros(12, 2))
        )
        self.assertTrue(torch.equal(augmented[:, 0], feat[:, 0]))
        self.assertTrue(torch.equal(augmented[:, sec.N_MELS], feat[:, sec.N_MELS]))

    def test_predict_single_matches_training_feature_pipeline(self):
        mel_spec = torch.arange(1, sec.N_MELS * 3 + 1, dtype=torch.float32).reshape(sec.N_MELS, 3)
        cmvn = (mel_spec - mel_spec.mean(dim=1, keepdim=True)) / (mel_spec.std(dim=1, keepdim=True) + 1e-6)
        delta1 = torch.full_like(cmvn, 5.0)
        delta2 = torch.full_like(cmvn, 9.0)
        expected_feat = torch.cat([cmvn, delta1, delta2], dim=0).T
        model = _RecordingModel()

        with patch.object(sec, "load_audio", return_value=torch.zeros(1, 1600)), patch.object(
            sec, "extract_mel_spectrogram", return_value=mel_spec
        ), patch.object(sec.librosa.feature, "delta", side_effect=[delta1.numpy(), delta2.numpy()]):
            result = sec.predict_single(model, "fake.wav", torch.device("cpu"))

        self.assertEqual(model.last_feat.shape, (1, sec.MAX_SEQ_LEN, sec.FEATURE_DIM))
        self.assertTrue(torch.allclose(model.last_feat[0, :3], expected_feat, atol=1e-5))
        self.assertTrue(torch.equal(model.last_mask[0, :5], torch.tensor([True, True, True, False, False])))
        self.assertEqual(result["predicted_emotion"], sec.IDX_TO_EMOTION[0])

    def test_predict_single_handles_short_clips_without_delta_crash(self):
        mel_spec = torch.arange(1, sec.N_MELS * 2 + 1, dtype=torch.float32).reshape(sec.N_MELS, 2)
        cmvn = (mel_spec - mel_spec.mean(dim=1, keepdim=True)) / (mel_spec.std(dim=1, keepdim=True) + 1e-6)
        model = _RecordingModel()

        with patch.object(sec, "load_audio", return_value=torch.zeros(1, 1600)), patch.object(
            sec, "extract_mel_spectrogram", return_value=mel_spec
        ):
            result = sec.predict_single(model, "fake.wav", torch.device("cpu"))

        self.assertEqual(model.last_feat.shape, (1, sec.MAX_SEQ_LEN, sec.FEATURE_DIM))
        self.assertTrue(torch.allclose(model.last_feat[0, :2, :sec.N_MELS], cmvn.T, atol=1e-5))
        self.assertTrue(torch.equal(model.last_feat[0, :2, sec.N_MELS:], torch.zeros(2, sec.N_MELS * 2)))
        self.assertTrue(torch.equal(model.last_mask[0, :4], torch.tensor([True, True, False, False])))
        self.assertEqual(result["predicted_emotion"], sec.IDX_TO_EMOTION[0])

    def test_predict_single_accepts_explicit_emotion_map_for_seven_class_model(self):
        mel_spec = torch.arange(1, sec.N_MELS * 2 + 1, dtype=torch.float32).reshape(sec.N_MELS, 2)
        model = _RecordingModelSeven()
        emotion_map = {
            0: "angry",
            1: "disgust",
            2: "fearful",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprised",
        }

        with patch.object(sec, "load_audio", return_value=torch.zeros(1, 1600)), patch.object(
            sec, "extract_mel_spectrogram", return_value=mel_spec
        ):
            result = sec.predict_single(model, "fake.wav", torch.device("cpu"), emotion_map=emotion_map)

        self.assertEqual(result["predicted_emotion"], "surprised")
        self.assertEqual(set(result["all_probabilities"].keys()), set(emotion_map.values()))

    def test_predict_mode_uses_env_model_complete_checkpoint(self):
        state_dict = sec.SpeechEmotionClassifier(input_dim=sec.FEATURE_DIM).state_dict()
        checkpoint = {
            "model_state_dict": state_dict,
            "config": {
                "input_dim": sec.FEATURE_DIM,
                "num_classes": len(sec.IDX_TO_EMOTION),
                "d_model": sec.D_MODEL,
                "nhead": sec.NHEAD,
                "num_layers": sec.NUM_ENCODER_LAYERS,
                "dim_feedforward": sec.DIM_FEEDFORWARD,
                "dropout": sec.DROPOUT,
            },
            "emotion_map": sec.IDX_TO_EMOTION,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model_complete.pth"
            audio_path = Path(tmpdir) / "sample.wav"
            torch.save(checkpoint, model_path)
            audio_path.write_bytes(b"fake")

            with patch.object(sys, "argv", [
                "speech_emotion_classifier",
                "--mode",
                "predict",
                "--audio",
                str(audio_path),
                "--model_path",
                str(model_path),
            ]), patch.object(sec, "predict_single", return_value={
                "predicted_emotion": "neutral",
                "confidence": 0.9,
                "all_probabilities": {emotion: 0.0 for emotion in sec.IDX_TO_EMOTION.values()},
            }) as predict_mock:
                sec.main()

        self.assertIsInstance(predict_mock.call_args.args[0], sec.SpeechEmotionClassifier)
        self.assertEqual(predict_mock.call_args.args[1], str(audio_path))
        self.assertEqual(predict_mock.call_args.args[2], sec.DEVICE)
        self.assertEqual(predict_mock.call_args.kwargs["emotion_map"], sec.IDX_TO_EMOTION)


class AdditionalDatasetScanTests(unittest.TestCase):
    def test_resolve_cremad_data_dir_accepts_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "1001_IEO_ANG_XX.wav").touch()

            resolved = sec.resolve_cremad_data_dir(str(root))

        self.assertEqual(resolved, str(root))

    def test_scan_savee_dataset_maps_emotions_and_speakers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for filename in ["DC_a01.wav", "JE_sa03.wav", "JK_su02.wav", "KL_x99.wav"]:
                (root / filename).touch()

            samples = sec.scan_savee_dataset(str(root))

        self.assertEqual(len(samples), 3)
        self.assertEqual({sample["emotion_name"] for sample in samples}, {"angry", "sad", "surprised"})
        self.assertEqual({sample["speaker"] for sample in samples}, {"DC", "JE", "JK"})

    def test_scan_tess_dataset_maps_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "TESS Toronto emotional speech set data"
            paths = [
                root / "OAF_angry" / "OAF_back_angry.wav",
                root / "YAF_pleasant_surprised" / "YAF_home_pleasant_surprised.wav",
                root / "OAF_Fear" / "OAF_dog_fear.wav",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            samples = sec.scan_tess_dataset(str(root.parent))

        self.assertEqual(len(samples), 3)
        self.assertEqual({sample["emotion_name"] for sample in samples}, {"angry", "surprised", "fearful"})
        self.assertEqual({sample["speaker"] for sample in samples}, {"OAF", "YAF"})

    def test_scan_tess_dataset_maps_ps_suffix_to_surprised(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "TESS Toronto emotional speech set data" / "OAF_ps"
            root.mkdir(parents=True, exist_ok=True)
            (root / "OAF_book_ps.wav").touch()

            samples = sec.scan_tess_dataset(str(root.parents[1]))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["emotion_name"], "surprised")

    def test_parse_tess_filename_preserves_three_letter_speaker_for_neutral(self):
        meta = sec.parse_tess_filename("/tmp/OAF_back_neutral.wav")

        self.assertEqual(meta["speaker"], "OAF")
        self.assertEqual(meta["emotion_name"], "neutral")


class TessOnlyTrainingTests(unittest.TestCase):
    def test_build_tess_label_mapping_uses_six_vec_classes(self):
        emotion_to_idx, idx_to_emotion = sec.build_label_space("tess")

        self.assertEqual(
            list(emotion_to_idx.keys()),
            ["angry", "disgust", "fearful", "happy", "neutral", "sad"],
        )
        self.assertNotIn("surprised", emotion_to_idx)
        self.assertEqual(len(idx_to_emotion), 6)

    def test_split_tess_samples_is_deterministic_and_non_empty(self):
        samples = []
        emotions = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        for speaker in ["OAF", "YAF"]:
            for emotion_idx, emotion_name in enumerate(emotions):
                for item_idx in range(5):
                    samples.append(
                        {
                            "filepath": f"{speaker}_{emotion_name}_{item_idx}.wav",
                            "emotion_name": emotion_name,
                            "emotion_idx": emotion_idx,
                            "speaker": speaker,
                        }
                    )

        first_split = sec.split_tess_samples(samples)
        second_split = sec.split_tess_samples(samples)

        self.assertEqual(first_split, second_split)
        self.assertTrue(first_split[0])
        self.assertTrue(first_split[1])
        self.assertTrue(first_split[2])

    def test_build_cache_path_uses_tess_mode_and_feature_dim(self):
        cache_dir = sec.build_feature_cache_dir("tess")

        self.assertTrue(str(cache_dir).endswith(f"output/cache/tess_vec_6class_fd{sec.FEATURE_DIM}"))

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

    def test_prepare_training_samples_returns_tess_only_when_requested(self):
        fake_samples = [
            {"filepath": "anger/a.wav", "emotion_name": "angry", "speaker": "OAF", "group_id": "g1"},
            {"filepath": "happy/b.wav", "emotion_name": "happy", "speaker": "OAF", "group_id": "g2"},
            {"filepath": "neutral/c.wav", "emotion_name": "neutral", "speaker": "YAF", "group_id": "g3"},
            {"filepath": "sad/d.wav", "emotion_name": "sad", "speaker": "YAF", "group_id": "g4"},
            {"filepath": "fear/e.wav", "emotion_name": "fearful", "speaker": "YAF", "group_id": "g5"},
        ]
        with patch.object(sec, "scan_vec_dataset", return_value=fake_samples) as scan_mock, patch.object(
            sec, "scan_tess_dataset", side_effect=AssertionError("legacy tess scanner should not be used")
        ):
            train_samples, val_samples, test_samples, emotion_to_idx, idx_to_emotion = sec.prepare_training_samples(
                "tess"
            )

        scan_mock.assert_called_once_with(sec.VEC_DATA_DIR)
        self.assertEqual(len(train_samples) + len(val_samples) + len(test_samples), 5)
        self.assertEqual(
            list(emotion_to_idx.keys()),
            ["angry", "disgust", "fearful", "happy", "neutral", "sad"],
        )
        self.assertEqual(idx_to_emotion[2], "fearful")


class VecBackedTessTests(unittest.TestCase):
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

    def test_create_training_components_uses_cached_dataset_when_enabled(self):
        samples = [{"cache_path": Path("/tmp/fake.pt"), "emotion_idx": 1}]

        train_dataset, _, _ = sec.build_datasets_from_samples(samples, [], [], use_cache=True)

        self.assertIsInstance(train_dataset, sec.CachedFeatureDataset)

    def test_amp_enabled_only_when_cuda_available(self):
        enabled, scaler_enabled = sec.resolve_amp_settings(torch.device("cuda"))

        self.assertTrue(enabled)
        self.assertTrue(scaler_enabled)

    def test_plot_confusion_matrix_uses_supplied_class_names(self):
        fake_axis = MagicMock()
        fake_axis.figure = MagicMock()
        fake_axis.imshow.return_value = MagicMock()
        fake_figure = MagicMock()

        with patch.object(sec.plt, "subplots", return_value=(fake_figure, fake_axis)), patch.object(
            sec.plt, "tight_layout"
        ), patch.object(sec.plt, "savefig"), patch.object(sec.plt, "close"):
            sec.plot_confusion_matrix(
                [0, 1, 2],
                [0, 1, 2],
                "/tmp/cm.png",
                class_names=["angry", "happy", "sad"],
            )

        kwargs = fake_axis.set.call_args.kwargs
        self.assertEqual(kwargs["xticklabels"], ["angry", "happy", "sad"])
        self.assertEqual(kwargs["yticklabels"], ["angry", "happy", "sad"])

    def test_plot_emotion_distribution_uses_supplied_class_names(self):
        fake_axis = MagicMock()
        fake_bars = [MagicMock() for _ in range(3)]
        for idx, bar in enumerate(fake_bars):
            bar.get_x.return_value = float(idx)
            bar.get_width.return_value = 1.0
            bar.get_height.return_value = 5.0
        fake_axis.bar.return_value = fake_bars
        fake_figure = MagicMock()

        with patch.object(sec.plt, "subplots", return_value=(fake_figure, fake_axis)), patch.object(
            sec.plt, "tight_layout"
        ), patch.object(sec.plt, "savefig"), patch.object(sec.plt, "close"):
            sec.plot_emotion_distribution(
                [
                    {"emotion_name": "angry"},
                    {"emotion_name": "happy"},
                    {"emotion_name": "sad"},
                ],
                "/tmp/dist.png",
                class_names=["angry", "happy", "sad"],
            )

        args, _ = fake_axis.bar.call_args
        self.assertEqual(args[0], ["angry", "happy", "sad"])


class TrainingOptimizationTests(unittest.TestCase):
    def test_apply_mixup_blends_features_labels_and_masks(self):
        features = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        labels = torch.tensor([0, 1])
        masks = torch.tensor([[True, False], [True, True]])

        with patch.object(sec.np.random, "beta", return_value=0.25), patch.object(
            sec.torch, "randperm", return_value=torch.tensor([1, 0])
        ):
            mixed_features, labels_a, labels_b, mixed_masks, lam = sec.apply_mixup(features, labels, masks)

        expected = 0.25 * features + 0.75 * features[[1, 0]]
        self.assertAlmostEqual(lam, 0.25)
        self.assertTrue(torch.equal(labels_a, labels))
        self.assertTrue(torch.equal(labels_b, torch.tensor([1, 0])))
        self.assertTrue(torch.allclose(mixed_features, expected))
        self.assertTrue(torch.equal(mixed_masks, torch.tensor([[True, True], [True, True]])))

    def test_build_sample_weights_prioritizes_minority_classes(self):
        samples = [
            {"emotion_idx": 0},
            {"emotion_idx": 0},
            {"emotion_idx": 1},
        ]

        weights = sec.build_sample_weights(samples)

        self.assertEqual(weights.shape[0], 3)
        self.assertGreater(weights[2].item(), weights[0].item())
        self.assertGreater(weights[2].item(), weights[1].item())

    def test_build_lr_scheduler_warms_up_then_decays(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = sec.build_lr_scheduler(optimizer, total_epochs=8, warmup_epochs=2)

        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(4):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        self.assertLess(lrs[0], lrs[1])
        self.assertLess(lrs[1], lrs[2])
        self.assertGreater(lrs[2], lrs[3])


if __name__ == "__main__":
    unittest.main()
