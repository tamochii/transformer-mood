"""
基于 Transformer 的语音情感特征分析（RAVDESS 真实数据集版）
==========================================================
任务1: 数据准备与预处理 - 加载 RAVDESS 真实音频，提取 Mel-spectrogram 特征
任务2: Dataset 类 - 截断/填充至固定长度 L
任务3: 正弦位置编码 (Positional Encoding)
任务4: Transformer Encoder 层搭建
任务5: 分类头设计 - GAP + FC
任务6: 训练循环与评估 + 推理接口

数据集: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
  - 24 位专业演员 (12 男 12 女)
  - 8 种情感: neutral, calm, happy, sad, angry, fearful, disgust, surprised
  - 1440 条语音文件
  - 文件名编码: {modality}-{channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav

用法:
  训练:   PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode train
  推理:   PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode predict --audio <path_to_wav>
"""

import os
import sys
import math
import argparse
import random
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib

matplotlib.use("Agg")  # WSL2 无 GUI，使用非交互后端
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 全局配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SAMPLE_RATE = 16000       # 统一重采样到 16kHz
N_MELS = 128              # Mel 频带数（真实数据用更高分辨率）
N_MFCC = 40               # MFCC 系数数
MAX_SEQ_LEN = 200         # 固定序列长度 L（时间帧数）
USE_DELTA = True
FEATURE_DIM = N_MELS * 3 if USE_DELTA else N_MELS
D_MODEL = 128             # Transformer 隐藏维度
NHEAD = 8                 # 多头注意力头数
NUM_ENCODER_LAYERS = 4    # Encoder 层数
DIM_FEEDFORWARD = 512     # FFN 中间维度
DROPOUT = 0.2
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# RAVDESS 数据集路径
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = str(PROJECT_ROOT / "data" / "ravdess")
CREMA_DATA_ROOT = str(PROJECT_ROOT / "data" / "cremad")
SAVEE_DATA_DIR = str(PROJECT_ROOT / "data" / "savee")
TESS_DATA_DIR = str(PROJECT_ROOT / "data" / "tess")
VEC_DATA_DIR = str(PROJECT_ROOT / "data" / "vec")
OUTPUT_DIR = str(PROJECT_ROOT / "output")

# RAVDESS 情感标签映射（文件名第3段）
# 原始8类: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# CREMA-D 情感标签映射（文件名第3段）
# 情感代码: ANG=angry, DIS=disgust, FEA=fearful, HAP=happy, NEU=neutral, SAD=sad
CREMA_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

SAVEE_EMOTIONS = {
    "a": "angry",
    "d": "disgust",
    "f": "fearful",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprised",
}

TESS_EMOTIONS = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "ps": "surprised",
    "pleasant_surprise": "surprised",
    "pleasant_surprised": "surprised",
    "sad": "sad",
}

TESS_ONLY_EMOTIONS = {
    "angry": "angry",
    "disgust": "disgust",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}

VEC_EMOTIONS = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}

AUGMENTED_SUFFIX_RE = re.compile(r"_aug\d+$")

# 使用全部 8 类情感
EMOTION_TO_IDX = {name: idx for idx, name in enumerate(RAVDESS_EMOTIONS.values())}
IDX_TO_EMOTION = {idx: name for name, idx in EMOTION_TO_IDX.items()}
NUM_CLASSES = len(EMOTION_TO_IDX)


def build_label_space(dataset_name: str):
    """根据数据集返回标签空间。"""
    if dataset_name == "tess":
        emotion_to_idx = {name: idx for idx, name in enumerate(TESS_ONLY_EMOTIONS.values())}
    else:
        emotion_to_idx = {name: idx for idx, name in enumerate(RAVDESS_EMOTIONS.values())}
    idx_to_emotion = {idx: name for name, idx in emotion_to_idx.items()}
    return emotion_to_idx, idx_to_emotion


def set_seed(seed=SEED):
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ############################################################
# 任务 1: 数据准备与预处理
# ############################################################
# 说明: 加载 RAVDESS 真实语音文件，从文件名解析情感标签，
# 将原始音频波形转换为 Mel-spectrogram 时序频谱特征。
# ############################################################


def parse_ravdess_filename(filepath: str) -> dict:
    """
    解析 RAVDESS 文件名，提取元数据。
    文件名格式: {modality}-{channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav

    Args:
        filepath: wav 文件路径
    Returns:
        dict: 包含 emotion, intensity, actor 等信息
    """
    filename = os.path.basename(filepath)
    parts = filename.replace(".wav", "").split("-")
    return {
        "modality": parts[0],      # 03=audio-only
        "channel": parts[1],       # 01=speech
        "emotion": parts[2],       # 01-08
        "intensity": parts[3],     # 01=normal, 02=strong
        "statement": parts[4],     # 01 or 02
        "repetition": parts[5],    # 01 or 02
        "actor": parts[6],         # 01-24
        "emotion_name": RAVDESS_EMOTIONS.get(parts[2], "unknown"),
    }


def parse_cremad_filename(filepath: str) -> dict:
    """
    解析 CREMA-D 文件名，提取元数据。
    文件名格式: {speaker}_{sentence}_{emotion}_{intensity}.wav

    Args:
        filepath: wav 文件路径
    Returns:
        dict: 包含 speaker, emotion, intensity, emotion_name 等信息
    """
    filename = os.path.basename(filepath)
    parts = filename.replace(".wav", "").split("_")
    emotion_code = parts[2]
    return {
        "speaker": parts[0],          # 1001-1091
        "sentence": parts[1],         # DFA, IEO, IOM, etc.
        "emotion": emotion_code,      # ANG, DIS, FEA, HAP, NEU, SAD
        "intensity": parts[3],        # XX (no intensity), HI, LO, MD
        "emotion_name": CREMA_EMOTIONS.get(emotion_code, "unknown"),
    }


def resolve_cremad_data_dir(data_dir: str) -> str:
    """兼容 CREMA-D 的两种常见目录布局。"""
    audio_wav_dir = os.path.join(data_dir, "AudioWAV")
    if os.path.isdir(audio_wav_dir):
        return audio_wav_dir
    return data_dir


def parse_savee_filename(filepath: str) -> dict:
    """解析 SAVEE 文件名。格式如 DC_a01.wav / JE_sa03.wav。"""
    filename = os.path.basename(filepath)
    stem = filename.replace(".wav", "")
    speaker, emotion_part = stem.split("_", 1)
    emotion_code = "".join(ch for ch in emotion_part.lower() if ch.isalpha())
    return {
        "speaker": speaker,
        "emotion": emotion_code,
        "emotion_name": SAVEE_EMOTIONS.get(emotion_code, "unknown"),
    }


def parse_tess_filename(filepath: str) -> dict:
    """解析 TESS 文件名。格式如 OAF_back_angry.wav。"""
    filename = os.path.basename(filepath)
    parts = filename.replace(".wav", "").split("_")
    emotion_code = "_".join(parts[2:]).lower()
    return {
        "speaker": parts[0],
        "emotion": emotion_code,
        "emotion_name": TESS_EMOTIONS.get(emotion_code, "unknown"),
    }


def strip_augmented_suffix(filepath: str) -> str:
    """移除 vec 文件名中的增强后缀，便于做 speaker/group 解析。"""
    return AUGMENTED_SUFFIX_RE.sub("", Path(filepath).stem)


def parse_vec_speaker(filepath: str) -> str:
    """兼容 TESS / CREMA-D / RAVDESS 风格的 speaker 解析。"""
    stem = strip_augmented_suffix(filepath)
    if stem.startswith(("OAF_", "YAF_")):
        return stem.split("_", 1)[0]
    if re.match(r"^\d+_", stem):
        return stem.split("_", 1)[0]
    if re.match(r"^(\d{2}-){6}\d{2}$", stem):
        return stem.split("-")[-1]
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def build_vec_group_id(filepath: str) -> str:
    """对原始样本和 *_augN.wav 生成相同分组键。"""
    return strip_augmented_suffix(filepath)


def load_audio(filepath: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    加载音频文件并重采样到目标采样率。

    Args:
        filepath: wav 文件路径
        target_sr: 目标采样率
    Returns:
        torch.Tensor: 音频波形 shape=(1, num_samples)
    """
    import soundfile as sf
    data, sr = sf.read(filepath, dtype="float32")
    # soundfile 返回 (num_samples,) 或 (num_samples, channels)
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, num_samples)
    else:
        waveform = torch.from_numpy(data.T)  # (channels, num_samples)
    # 如果是立体声，转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # 重采样
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform


def extract_mel_spectrogram(waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    将音频波形转换为 Mel-spectrogram 时序频谱特征。
    这是语音信号预处理的核心步骤：将时域信号转为时频域表示。

    Args:
        waveform: shape=(1, num_samples)
        sr: 采样率
    Returns:
        torch.Tensor: Mel-spectrogram (dB), shape=(n_mels, time_frames)
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_spec = mel_transform(waveform)            # (1, n_mels, time)
    mel_spec_db = T.AmplitudeToDB(top_db=80)(mel_spec)  # 转为对数刻度
    return mel_spec_db.squeeze(0)                  # (n_mels, time)


def build_feature_sequence(mel_spec: torch.Tensor) -> torch.Tensor:
    """对 mel 频谱做归一化并按需拼接 delta 特征。"""
    mel_spec = (mel_spec - mel_spec.mean(dim=1, keepdim=True)) / (
        mel_spec.std(dim=1, keepdim=True) + 1e-6
    )
    if not USE_DELTA:
        return mel_spec.T

    num_frames = mel_spec.shape[1]
    if num_frames < 3:
        delta1 = torch.zeros_like(mel_spec)
        delta2 = torch.zeros_like(mel_spec)
    else:
        delta_width = min(9, num_frames if num_frames % 2 == 1 else num_frames - 1)
        mel_spec_np = mel_spec.cpu().numpy()
        delta1 = torch.from_numpy(librosa.feature.delta(mel_spec_np, width=delta_width)).to(mel_spec.dtype)
        delta2 = torch.from_numpy(librosa.feature.delta(mel_spec_np, order=2, width=delta_width)).to(mel_spec.dtype)

    return torch.cat([mel_spec, delta1, delta2], dim=0).T


def build_feature_cache_dir(dataset_name: str) -> Path:
    """返回特征缓存目录。"""
    if dataset_name == "tess":
        return Path(OUTPUT_DIR) / "cache" / f"tess_vec_6class_fd{FEATURE_DIM}"
    return Path(OUTPUT_DIR) / "cache" / f"{dataset_name}_fd{FEATURE_DIM}"


def build_feature_payload(sample: dict) -> dict:
    """将单条样本转换为可缓存的特征载荷。"""
    mel_spec = extract_mel_spectrogram(load_audio(sample["filepath"]))
    feat = build_feature_sequence(mel_spec)
    seq_len = feat.shape[0]
    feat_dim = feat.shape[1]
    mask = torch.ones(MAX_SEQ_LEN, dtype=torch.bool)
    if seq_len > MAX_SEQ_LEN:
        feat = feat[:MAX_SEQ_LEN, :]
    elif seq_len < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - seq_len
        feat = torch.cat([feat, torch.zeros(pad_len, feat_dim, dtype=feat.dtype)], dim=0)
        mask[seq_len:] = False
    return {
        "feature": feat,
        "label": sample["emotion_idx"],
        "mask": mask,
        "speaker": sample.get("speaker"),
        "filepath": sample["filepath"],
    }


class CachedFeatureDataset(Dataset):
    """从缓存特征文件读取样本。"""

    def __init__(self, cache_paths: list[Path], augment: bool = False):
        self.cache_paths = cache_paths
        self.augment = augment

    def __len__(self):
        return len(self.cache_paths)

    def __getitem__(self, idx):
        payload = torch.load(self.cache_paths[idx], map_location="cpu", weights_only=False)
        feat = payload["feature"].clone()
        if self.augment:
            feat = SpeechEmotionDataset([], augment=True)._spec_augment(feat)
        label = torch.tensor(payload["label"], dtype=torch.long)
        mask = payload["mask"].clone()
        return feat, label, mask


def ensure_feature_cache(samples: list, dataset_name: str):
    """为样本生成缓存文件，并回填 cache_path。"""
    cache_dir = build_feature_cache_dir(dataset_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_samples = []
    for index, sample in enumerate(samples):
        cache_path = cache_dir / f"sample_{index:05d}.pt"
        if not cache_path.exists():
            torch.save(build_feature_payload(sample), cache_path)
        cached_samples.append({**sample, "cache_path": cache_path})
    return cached_samples


def build_datasets_from_samples(train_samples, val_samples, test_samples, use_cache: bool):
    """根据是否使用缓存构建 Dataset。"""
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
    """根据设备决定是否启用 AMP。"""
    enabled = device.type == "cuda"
    return enabled, enabled


def scan_ravdess_dataset(data_dir: str) -> list:
    """
    扫描 RAVDESS 数据集，收集所有音频文件路径和标签。

    Args:
        data_dir: RAVDESS 数据集根目录（包含 Actor_01 到 Actor_24）
    Returns:
        list of dict: 每条包含 filepath, emotion_name, emotion_idx, actor
    """
    samples = []
    for actor_dir in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor_dir)
        if not os.path.isdir(actor_path) or not actor_dir.startswith("Actor_"):
            continue
        for wav_file in sorted(os.listdir(actor_path)):
            if not wav_file.endswith(".wav"):
                continue
            filepath = os.path.join(actor_path, wav_file)
            meta = parse_ravdess_filename(filepath)
            emotion_name = meta["emotion_name"]
            if emotion_name in EMOTION_TO_IDX:
                samples.append({
                    "filepath": filepath,
                    "emotion_name": emotion_name,
                    "emotion_idx": EMOTION_TO_IDX[emotion_name],
                    "speaker": int(meta["actor"]),
                })
    return samples


def scan_cremad_dataset(data_dir: str) -> list:
    """
    扫描 CREMA-D 数据集，收集所有音频文件路径和标签。

    Args:
        data_dir: CREMA-D AudioWAV 目录
    Returns:
        list of dict: 每条包含 filepath, emotion_name, emotion_idx, speaker
    """
    samples = []
    if not os.path.isdir(data_dir):
        return samples
    for wav_file in sorted(os.listdir(data_dir)):
        if not wav_file.endswith(".wav"):
            continue
        filepath = os.path.join(data_dir, wav_file)
        meta = parse_cremad_filename(filepath)
        emotion_name = meta["emotion_name"]
        if emotion_name in EMOTION_TO_IDX:
            samples.append({
                "filepath": filepath,
                "emotion_name": emotion_name,
                "emotion_idx": EMOTION_TO_IDX[emotion_name],
                "speaker": int(meta["speaker"]),
            })
    return samples


def scan_savee_dataset(data_dir: str) -> list:
    """扫描 SAVEE 数据集目录。"""
    samples = []
    if not os.path.isdir(data_dir):
        return samples
    for wav_file in sorted(os.listdir(data_dir)):
        if not wav_file.endswith(".wav"):
            continue
        filepath = os.path.join(data_dir, wav_file)
        meta = parse_savee_filename(filepath)
        emotion_name = meta["emotion_name"]
        if emotion_name in EMOTION_TO_IDX:
            samples.append({
                "filepath": filepath,
                "emotion_name": emotion_name,
                "emotion_idx": EMOTION_TO_IDX[emotion_name],
                "speaker": meta["speaker"],
            })
    return samples


def scan_tess_dataset(data_dir: str) -> list:
    """递归扫描 TESS 数据集目录。"""
    samples = []
    if not os.path.isdir(data_dir):
        return samples
    for root, _, files in os.walk(data_dir):
        for wav_file in sorted(files):
            if not wav_file.endswith(".wav"):
                continue
            filepath = os.path.join(root, wav_file)
            meta = parse_tess_filename(filepath)
            emotion_name = meta["emotion_name"]
            if emotion_name in EMOTION_TO_IDX:
                samples.append({
                    "filepath": filepath,
                    "emotion_name": emotion_name,
                    "emotion_idx": EMOTION_TO_IDX[emotion_name],
                    "speaker": meta["speaker"],
                })
    return samples


def scan_vec_dataset(data_dir: str) -> list:
    """扫描 vec 目录并统一为 tess 模式所需的样本结构。"""
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
            samples.append({
                "filepath": filepath,
                "emotion_name": emotion_name,
                "speaker": parse_vec_speaker(filepath),
                "group_id": build_vec_group_id(filepath),
            })
    return samples


# ############################################################
# 任务 2: Dataset 类 —— 截断/填充至固定长度 L
# ############################################################
# 说明: 不同语音的时长不同，产生的 Mel-spectrogram 时间帧数也不同。
# Transformer 需要固定长度输入，因此：
#   - 过长的序列 → 截断 (truncate)
#   - 过短的序列 → 零填充 (zero-padding)
# 同时生成 padding mask，让 Transformer 忽略填充位置。
# ############################################################


class SpeechEmotionDataset(Dataset):
    """
    RAVDESS 语音情感数据集类。

    将 Mel-spectrogram 特征截断或填充至固定长度 L。
    输出:
        features: (max_seq_len, n_mels) — 时间步 x 特征维度
        label: int — 情感类别索引
        mask: (max_seq_len,) — padding mask (True=有效, False=填充)
    """

    def __init__(self, samples: list, max_seq_len: int = MAX_SEQ_LEN,
                 augment: bool = False):
        """
        Args:
            samples: list of dict, 包含 filepath 和 emotion_idx
            max_seq_len: 固定时间步长度 L
            augment: 是否开启数据增强
        """
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filepath = sample["filepath"]
        label = sample["emotion_idx"]

        # 1. 加载音频
        waveform = load_audio(filepath)

        # 2. 数据增强（仅训练时）
        if self.augment:
            waveform = self._augment(waveform)

        # 3. 提取 Mel-spectrogram
        mel_spec = extract_mel_spectrogram(waveform)  # (n_mels, time_frames)
        feat = build_feature_sequence(mel_spec)

        if self.augment:
            feat = self._spec_augment(feat)

        seq_len = feat.shape[0]
        feat_dim = feat.shape[1]

        # 5. 创建 padding mask
        mask = torch.ones(self.max_seq_len, dtype=torch.bool)

        # 6. 截断或填充至固定长度
        if seq_len > self.max_seq_len:
            feat = feat[:self.max_seq_len, :]
        elif seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            padding = torch.zeros(pad_len, feat_dim, dtype=feat.dtype)
            feat = torch.cat([feat, padding], dim=0)
            mask[seq_len:] = False  # 填充位置标记为 False

        label_tensor = torch.tensor(label, dtype=torch.long)
        return feat, label_tensor, mask

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """简单的音频数据增强。"""
        # 随机添加高斯噪声
        if random.random() < 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        # 随机时间偏移
        if random.random() < 0.3:
            shift = random.randint(-1600, 1600)  # ±0.1s
            waveform = torch.roll(waveform, shifts=shift, dims=-1)
        # 随机音量缩放
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            waveform = waveform * scale
        return waveform

    def _spec_augment(self, feat: torch.Tensor) -> torch.Tensor:
        """对时频特征做简单 SpecAugment。"""
        feat = feat.clone()
        time_len, feat_dim = feat.shape

        freq_bins = N_MELS if USE_DELTA and feat_dim >= N_MELS else feat_dim
        freq_mask_width = random.randint(0, min(27, freq_bins // 4))
        if freq_mask_width > 0:
            freq_start = random.randint(0, freq_bins - freq_mask_width)
            if USE_DELTA and feat_dim >= FEATURE_DIM:
                for offset in range(0, feat_dim, N_MELS):
                    feat[:, offset + freq_start:offset + freq_start + freq_mask_width] = 0
            else:
                feat[:, freq_start:freq_start + freq_mask_width] = 0

        time_mask_width = random.randint(0, min(40, time_len // 4))
        if time_mask_width > 0:
            time_start = random.randint(0, time_len - time_mask_width)
            feat[time_start:time_start + time_mask_width, :] = 0

        return feat


# ############################################################
# 任务 3: 位置编码 (Positional Encoding)
# ############################################################
# 由于 Transformer 缺乏对时序位置的感知能力，
# 使用正弦位置编码注入位置信息：
#   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
# ############################################################


class PositionalEncoding(nn.Module):
    """
    正弦位置编码 (Sinusoidal Positional Encoding)。

    为输入序列的每个时间步添加位置信息，使 Transformer 能够
    感知序列中元素的顺序。

    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i)
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, seq_len, d_model)
        Returns:
            加上位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ############################################################
# 任务 4: Transformer 编码器层搭建
# ############################################################
# 核心: Multi-head Self-Attention + FeedForward
#   - W_Q, W_K, W_V: 将输入映射到 Query, Key, Value
#   - Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
#   - 多头: 将 d_model 拆分为 h 个头, 各自计算注意力后拼接
#   - FFN: Linear → ReLU → Linear
# ############################################################


class TransformerEmotionEncoder(nn.Module):
    """
    Transformer 编码器: 输入投影 + 位置编码 + N 层 EncoderLayer。

    结构:
        (L, n_mels) → Linear → (L, d_model)
        → Positional Encoding
        → N x TransformerEncoderLayer
            [Multi-Head Self-Attention + FFN + LayerNorm + Residual]
        → (L, d_model)
    """

    def __init__(self, input_dim: int, d_model: int = D_MODEL,
                 nhead: int = NHEAD, num_layers: int = NUM_ENCODER_LAYERS,
                 dim_feedforward: int = DIM_FEEDFORWARD, dropout: float = DROPOUT):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",        # GELU 激活比 ReLU 效果更好
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            src_key_padding_mask: (batch, seq_len) — True=忽略该位置
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        # TransformerEncoder 的 mask 约定: True=忽略, 所以我们要取反
        # 我们的 mask 里 True=有效, False=填充, 需要反转
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask  # 反转: True→忽略填充
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x


# ############################################################
# 任务 5: 分类头设计
# ############################################################
# Transformer 编码器输出后:
#   1. 全局平均池化 (GAP) — 将序列压缩为单个向量
#      只对有效位置 (非 padding) 取平均
#   2. 全连接层 → 类别数
# ############################################################


class SpeechEmotionClassifier(nn.Module):
    """
    完整的语音情感分类器。

    结构:
        Mel-spectrogram (L, n_mels)
        → Transformer Encoder → (L, d_model)
        → Masked Global Average Pooling → (d_model,)
        → LayerNorm → Dropout → FC → (num_classes,)
    """

    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES,
                 d_model: int = D_MODEL, nhead: int = NHEAD,
                 num_layers: int = NUM_ENCODER_LAYERS,
                 dim_feedforward: int = DIM_FEEDFORWARD, dropout: float = DROPOUT):
        super().__init__()

        self.encoder = TransformerEmotionEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) — True=有效, False=填充
        Returns:
            logits: (batch, num_classes)
        """
        encoder_output = self.encoder(x, src_key_padding_mask=mask)  # (B, L, d_model)

        # Masked Global Average Pooling: 只对有效帧取平均
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (encoder_output * mask_expanded).sum(dim=1)  # (B, d_model)
            pooled = pooled / mask_expanded.sum(dim=1).clamp(min=1)  # 避免除零
        else:
            pooled = encoder_output.mean(dim=1)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ############################################################
# 任务 6: 训练循环与评估
# ############################################################


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch,
                    amp_enabled: bool = False, scaler=None):
    """单个 epoch 的训练。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels, masks in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            use_mixup = random.random() < 0.5
            if use_mixup:
                mixed_features, labels_a, labels_b, mixed_masks, lam = apply_mixup(features, labels, masks)
                logits = model(mixed_features, mask=mixed_masks)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(features, mask=masks)
                loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if scheduler is not None:
        scheduler.step()

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device, amp_enabled: bool = False):
    """评估模型。"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels, masks in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(features, mask=masks)
                loss = criterion(logits, labels)

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def predict_single(model, audio_path: str, device, emotion_map=None) -> dict:
    """
    对单个音频文件进行情感预测。

    Args:
        model: 训练好的模型
        audio_path: wav 文件路径
        device: 计算设备
    Returns:
        dict: 包含预测标签、概率分布
    """
    if emotion_map is None:
        emotion_map = IDX_TO_EMOTION

    model.eval()
    waveform = load_audio(audio_path)
    mel_spec = extract_mel_spectrogram(waveform)
    feat = build_feature_sequence(mel_spec)

    # 截断/填充
    seq_len = feat.shape[0]
    feat_dim = feat.shape[1]
    mask = torch.ones(MAX_SEQ_LEN, dtype=torch.bool)
    if seq_len > MAX_SEQ_LEN:
        feat = feat[:MAX_SEQ_LEN, :]
    elif seq_len < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - seq_len
        padding = torch.zeros(pad_len, feat_dim, dtype=feat.dtype)
        feat = torch.cat([feat, padding], dim=0)
        mask[seq_len:] = False

    feat = feat.unsqueeze(0).to(device)       # (1, L, n_mels)
    mask = mask.unsqueeze(0).to(device)        # (1, L)

    with torch.no_grad():
        logits = model(feat, mask=mask)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()

    result = {
        "predicted_emotion": emotion_map[pred_idx],
        "confidence": probs[pred_idx].item(),
        "all_probabilities": {emotion_map[i]: probs[i].item() for i in range(len(emotion_map))},
    }
    return result


def resolve_inference_model_path(explicit_model_path: str | None) -> str:
    if explicit_model_path:
        return explicit_model_path
    configured_model_path = os.environ.get("EMOTION_MODEL_PATH")
    if configured_model_path:
        return configured_model_path
    return os.path.join(OUTPUT_DIR, "best_model.pth")


def load_inference_model(model_path: str):
    payload = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        config = payload.get("config", {})
        model = SpeechEmotionClassifier(
            input_dim=config.get("input_dim", FEATURE_DIM),
            num_classes=config.get("num_classes", len(IDX_TO_EMOTION)),
            d_model=config.get("d_model", D_MODEL),
            nhead=config.get("nhead", NHEAD),
            num_layers=config.get("num_layers", NUM_ENCODER_LAYERS),
            dim_feedforward=config.get("dim_feedforward", DIM_FEEDFORWARD),
            dropout=config.get("dropout", DROPOUT),
        ).to(DEVICE)
        model.load_state_dict(payload["model_state_dict"])
        emotion_map = payload.get("emotion_map", IDX_TO_EMOTION)
    else:
        model = SpeechEmotionClassifier(input_dim=FEATURE_DIM).to(DEVICE)
        model.load_state_dict(payload)
        emotion_map = IDX_TO_EMOTION
    model.eval()
    return model, emotion_map


# ############################################################
# 可视化
# ############################################################


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """绘制训练/验证的 Loss 和 Accuracy 曲线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.set_title("训练/验证 Loss 曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-", label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, val_accs, "r-", label="Val Acc", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("训练/验证 Accuracy 曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 训练曲线已保存至: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    """绘制混淆矩阵。"""
    if class_names is None:
        class_names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 混淆矩阵已保存至: {save_path}")


def plot_emotion_distribution(samples, save_path, class_names=None):
    """绘制数据集情感分布柱状图。"""
    from collections import Counter
    emotions = [s["emotion_name"] for s in samples]
    counter = Counter(emotions)
    if class_names is None:
        class_names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]
    names = class_names
    counts = [counter.get(name, 0) for name in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, counts, color=colors, edgecolor="gray")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.set_title("Emotion Distribution")
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 情感分布图已保存至: {save_path}")


# ############################################################
# 数据划分策略: 按 Actor 划分（防止数据泄露）
# ############################################################
# 说明: 同一演员的不同语音在声纹上高度相似。
# 如果随机划分，模型可能学到"识别谁在说话"而非"识别情感"。
# 因此按 Actor 划分: 训练集用某些演员, 测试集用其他演员。
# ############################################################


def split_by_actor(samples: list, test_actors: list = None, val_actors: list = None):
    """
    按说话者划分数据集，避免数据泄露。

    默认: Speaker 21-24 作为测试集, Speaker 19-20 作为验证集, 其余训练。
    注意：样本中应包含 "speaker" 键。
    """
    if test_actors is None:
        test_actors = [21, 22, 23, 24]       # 4 speakers for test
    if val_actors is None:
        val_actors = [19, 20]                 # 2 speakers for validation

    train_samples, val_samples, test_samples = [], [], []
    for s in samples:
        if s["speaker"] in test_actors:
            test_samples.append(s)
        elif s["speaker"] in val_actors:
            val_samples.append(s)
        else:
            train_samples.append(s)

    return train_samples, val_samples, test_samples


def split_by_sorted_speakers(samples: list, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """按 speaker 排序后切分，适合可选外部数据集。"""
    if not samples:
        return [], [], []

    speakers = sorted(set(s["speaker"] for s in samples))
    if len(speakers) == 1:
        return samples, [], []

    train_end = max(1, int(len(speakers) * train_ratio))
    val_end = max(train_end, int(len(speakers) * (train_ratio + val_ratio)))
    if len(speakers) - train_end > 1 and val_end == train_end:
        val_end += 1
    val_end = min(val_end, len(speakers) - 1)

    train_speakers = set(speakers[:train_end])
    val_speakers = set(speakers[train_end:val_end])
    test_speakers = set(speakers[val_end:])

    train_samples, val_samples, test_samples = [], [], []
    for sample in samples:
        if sample["speaker"] in train_speakers:
            train_samples.append(sample)
        elif sample["speaker"] in val_speakers:
            val_samples.append(sample)
        else:
            test_samples.append(sample)
    return train_samples, val_samples, test_samples


def split_tess_samples(samples: list):
    """按 speaker + group_id 稳定切分，避免增强样本泄露。"""
    grouped = {}
    for sample in samples:
        speaker_groups = grouped.setdefault(sample["speaker"], {})
        group_id = sample.get("group_id") or build_vec_group_id(sample["filepath"])
        speaker_groups.setdefault(group_id, []).append(sample)

    train_samples, val_samples, test_samples = [], [], []
    for speaker in sorted(grouped):
        grouped_samples = [
            sorted(group, key=lambda item: item["filepath"])
            for _, group in sorted(grouped[speaker].items())
        ]
        group_count = len(grouped_samples)
        if group_count == 1:
            train_groups, val_groups, test_groups = grouped_samples, [], []
        else:
            train_end = max(1, int(group_count * 0.6))
            val_end = max(train_end + 1, int(group_count * 0.8))
            val_end = min(val_end, group_count - 1)
            train_groups = grouped_samples[:train_end]
            val_groups = grouped_samples[train_end:val_end]
            test_groups = grouped_samples[val_end:]
        train_samples.extend(sample for group in train_groups for sample in group)
        val_samples.extend(sample for group in val_groups for sample in group)
        test_samples.extend(sample for group in test_groups for sample in group)
    return train_samples, val_samples, test_samples


def prepare_training_samples(dataset_name: str):
    """根据训练模式准备样本和标签空间。"""
    emotion_to_idx, idx_to_emotion = build_label_space(dataset_name)
    if dataset_name == "tess":
        tess_samples = []
        for sample in scan_vec_dataset(VEC_DATA_DIR):
            if sample["emotion_name"] not in emotion_to_idx:
                continue
            tess_samples.append({
                **sample,
                "emotion_idx": emotion_to_idx[sample["emotion_name"]],
            })
        train_samples, val_samples, test_samples = split_tess_samples(tess_samples)
        return train_samples, val_samples, test_samples, emotion_to_idx, idx_to_emotion

    ravdess_samples = scan_ravdess_dataset(DATA_DIR)
    cremad_samples = scan_cremad_dataset(resolve_cremad_data_dir(CREMA_DATA_ROOT))
    savee_samples = scan_savee_dataset(SAVEE_DATA_DIR)
    tess_samples = scan_tess_dataset(TESS_DATA_DIR)

    train_samples, val_samples, test_samples = [], [], []
    ravdess_train, ravdess_val, ravdess_test = split_by_actor(ravdess_samples)
    train_samples.extend(ravdess_train)
    val_samples.extend(ravdess_val)
    test_samples.extend(ravdess_test)

    for dataset_samples in [cremad_samples, savee_samples, tess_samples]:
        ds_train, ds_val, ds_test = split_by_sorted_speakers(dataset_samples)
        train_samples.extend(ds_train)
        val_samples.extend(ds_val)
        test_samples.extend(ds_test)

    return train_samples, val_samples, test_samples, emotion_to_idx, idx_to_emotion


def apply_mixup(features: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor, alpha: float = 0.2):
    """对一个 batch 应用 Mixup。"""
    lam = float(np.random.beta(alpha, alpha))
    indices = torch.randperm(features.size(0), device=features.device)
    mixed_features = lam * features + (1 - lam) * features[indices]
    mixed_masks = masks | masks[indices]
    return mixed_features, labels, labels[indices], mixed_masks, lam


def build_sample_weights(samples: list) -> torch.DoubleTensor:
    """为 WeightedRandomSampler 构建逐样本权重。"""
    labels = [sample["emotion_idx"] for sample in samples]
    class_sample_counts = np.bincount(labels, minlength=NUM_CLASSES)
    sample_weights = 1.0 / (class_sample_counts[labels] + 1e-6)
    return torch.DoubleTensor(sample_weights)


def build_lr_scheduler(optimizer, total_epochs: int, warmup_epochs: int = 5):
    """构建 warmup + cosine 学习率调度器。"""
    warmup_epochs = min(warmup_epochs, max(total_epochs - 1, 1))
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-6
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


# ############################################################
# 主函数
# ############################################################


def main():
    parser = argparse.ArgumentParser(description="基于 Transformer 的语音情感分类器 (RAVDESS)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"],
                        help="运行模式: train=训练, predict=推理")
    parser.add_argument("--audio", type=str, default=None,
                        help="推理模式下的音频文件路径")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型权重路径（推理时使用）")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"训练轮数 (默认: {NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"批大小 (默认: {BATCH_SIZE})")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help=f"DataLoader worker 数 (默认: {NUM_WORKERS})")
    parser.add_argument("--dataset", type=str, default="multi", choices=["multi", "tess"],
                        help="训练数据范围: multi=多数据集, tess=vec 6类兼容模式")
    parser.add_argument("--cache-features", action="store_true",
                        help="训练前预缓存特征，减少每个 epoch 的 CPU 预处理开销")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"学习率 (默认: {LEARNING_RATE})")
    args = parser.parse_args()

    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.mode == "predict":
        # ------ 推理模式 ------
        if args.audio is None:
            print("[ERROR] 推理模式需要指定 --audio 参数")
            sys.exit(1)
        model_path = resolve_inference_model_path(args.model_path)
        if not os.path.exists(model_path):
            print(f"[ERROR] 模型文件不存在: {model_path}")
            sys.exit(1)

        print(f"[INFO] 加载模型: {model_path}")
        model, emotion_map = load_inference_model(model_path)
        result = predict_single(model, args.audio, DEVICE, emotion_map=emotion_map)
        print(f"\n{'='*50}")
        print(f"  音频文件: {args.audio}")
        print(f"  预测情感: {result['predicted_emotion']}")
        print(f"  置信度:   {result['confidence']:.4f}")
        print(f"\n  各情感概率:")
        for emotion, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 40)
            print(f"    {emotion:>10s}: {prob:.4f} {bar}")
        print(f"{'='*50}")
        return

    # ------ 训练模式 ------
    print("=" * 60)
    print("  基于 Transformer 的语音情感特征分析")
    print("=" * 60)

    # === 第一阶段: 数据准备 ===
    print(f"\n[阶段1] 数据准备与预处理")
    print("-" * 40)

    if args.dataset == "multi" and not os.path.isdir(DATA_DIR):
        print(f"[ERROR] 数据目录不存在: {DATA_DIR}")
        print("请先下载 RAVDESS 数据集并解压到 data/ravdess/ 目录")
        sys.exit(1)
    if args.dataset == "tess" and not os.path.isdir(VEC_DATA_DIR):
        print(f"[ERROR] 数据目录不存在: {VEC_DATA_DIR}")
        print("请先将 vec 数据集放到 data/vec/ 目录")
        sys.exit(1)

    train_samples, val_samples, test_samples, emotion_to_idx, idx_to_emotion = prepare_training_samples(args.dataset)
    num_classes = len(emotion_to_idx)
    samples = train_samples + val_samples + test_samples
    print(f"  模式: {'tess 参数 / vec 数据 / 6类' if args.dataset == 'tess' else 'Multi-Dataset'}")
    print(f"  总计 {len(samples)} 条语音样本")
    print(f"  情感类别: {num_classes} 类 — {list(emotion_to_idx.keys())}")
    
    print(f"\n  数据划分 (按 Speaker, 防止数据泄露):")
    print(f"    训练集: {len(train_samples)} 条")
    print(f"    验证集: {len(val_samples)} 条")
    print(f"    测试集: {len(test_samples)} 条")

    # 绘制情感分布
    plot_emotion_distribution(
        samples,
        os.path.join(OUTPUT_DIR, "emotion_distribution.png"),
        class_names=list(emotion_to_idx.keys()),
    )

    if args.cache_features:
        train_samples = ensure_feature_cache(train_samples, args.dataset)
        val_samples = ensure_feature_cache(val_samples, args.dataset)
        test_samples = ensure_feature_cache(test_samples, args.dataset)

    train_dataset, val_dataset, test_dataset = build_datasets_from_samples(
        train_samples, val_samples, test_samples, use_cache=args.cache_features
    )

    train_sample_weights = build_sample_weights(train_samples)
    train_sampler = WeightedRandomSampler(
        train_sample_weights, num_samples=len(train_sample_weights), replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=DEVICE.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=DEVICE.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=DEVICE.type == "cuda")

    # 验证数据 shape
    sample_batch = next(iter(train_loader))
    print(f"\n  Batch 特征 shape: {sample_batch[0].shape}  (batch, L={MAX_SEQ_LEN}, feat_dim={FEATURE_DIM})")
    print(f"  Batch 标签 shape: {sample_batch[1].shape}")
    print(f"  Batch Mask shape: {sample_batch[2].shape}")

    # === 第二阶段: 构建模型 ===
    print(f"\n[阶段2] 构建 Transformer 编码器 + 分类头")
    print("-" * 40)

    model = SpeechEmotionClassifier(input_dim=FEATURE_DIM, num_classes=num_classes).to(DEVICE)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  设备: {DEVICE}")

    # === 第三阶段: 训练与评估 ===
    print(f"\n[阶段3] 模型训练与评估")
    print("-" * 40)

    amp_enabled, scaler_enabled = resolve_amp_settings(DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    # 计算类别权重（处理类别不均衡）
    train_labels = [s["emotion_idx"] for s in train_samples]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = build_lr_scheduler(optimizer, total_epochs=args.epochs, warmup_epochs=5)

    print(f"  优化器: AdamW  学习率: {args.lr}  权重衰减: {WEIGHT_DECAY}")
    print(f"  损失函数: CrossEntropyLoss (加权 + label smoothing)")
    print(f"  训练轮数: {args.epochs}  批大小: {args.batch_size}  workers: {args.num_workers}")
    print(f"  类别权重: {class_weights.round(2).tolist()}")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE, epoch,
            amp_enabled=amp_enabled, scaler=scaler,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE, amp_enabled=amp_enabled)

        current_lr = scheduler.get_last_lr()[0]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping + 保存最优
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"  Epoch [{epoch:2d}/{args.epochs}]  "
                f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f}  |  "
                f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f}  |  "
                f"LR={current_lr:.6f}  Best={best_val_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"\n  [Early Stopping] 验证集 {patience} 轮未提升, 停止训练")
            break

    print(f"\n  最佳验证准确率: {best_val_acc:.4f}")

    # === 测试集最终评估 ===
    print(f"\n[最终评估] 测试集")
    print("-" * 40)

    model.load_state_dict(torch.load(
        os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE, weights_only=True
    ))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE, amp_enabled=amp_enabled)

    print(f"  测试 Loss: {test_loss:.4f}")
    print(f"  测试 Accuracy: {test_acc:.4f}")

    target_names = [idx_to_emotion[i] for i in range(num_classes)]
    print(f"\n  分类报告:")
    print(classification_report(test_labels, test_preds, labels=list(range(num_classes)), target_names=target_names))

    # === 可视化 ===
    print("[可视化] 生成图表...")
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(OUTPUT_DIR, "training_curves.png"),
    )
    plot_confusion_matrix(
        test_labels, test_preds,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        class_names=target_names,
    )

    # === 保存最终模型 ===
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": FEATURE_DIM, "d_model": D_MODEL, "nhead": NHEAD,
            "num_layers": NUM_ENCODER_LAYERS, "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT, "num_classes": num_classes,
            "max_seq_len": MAX_SEQ_LEN, "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
        },
        "emotion_map": idx_to_emotion,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }, os.path.join(OUTPUT_DIR, "model_complete.pth"))

    print(f"\n{'='*60}")
    print(f"  全部任务完成!")
    print(f"  输出文件 ({OUTPUT_DIR}/):")
    print(f"    - best_model.pth          (最优模型权重)")
    print(f"    - model_complete.pth      (完整存档: 权重+配置+情感映射)")
    print(f"    - training_curves.png     (训练曲线)")
    print(f"    - confusion_matrix.png    (混淆矩阵)")
    print(f"    - emotion_distribution.png (情感分布)")
    print(f"\n  推理用法:")
    print(
        "    PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier "
        "--mode predict --audio <your.wav>"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
