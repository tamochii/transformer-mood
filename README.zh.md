<p align="right">
  <a href="./README.md">English</a> | 简体中文
</p>

<h1 align="center">Transformer Mood</h1>

<p align="center">
  一个本地优先的语音情绪识别项目，包含训练、命令行推理和 FastAPI WebUI。
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB.svg">
  <img alt="Model" src="https://img.shields.io/badge/Model-Transformer-7A3FFF.svg">
  <img alt="WebUI" src="https://img.shields.io/badge/WebUI-FastAPI-009688.svg">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-RAVDESS-EA4C89.svg">
</p>

## 功能

- RAVDESS 数据集预处理与训练
- Transformer Encoder 语音情绪分类模型
- 命令行单文件推理
- FastAPI WebUI
- 浏览器录音、音频上传、情绪概率展示

## 仓库结构

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
README.zh.md                    # 中文项目说明
output/                        # 运行时输出目录（除 .gitkeep 外均忽略）
data/README.md                 # 数据集放置说明
data/README.zh.md              # 中文数据集放置说明
transformer-md/                # 参考资料
requirements.txt               # 非 PyTorch Python 依赖
requirements-webui.txt         # WebUI 最小额外依赖
```

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
```

先安装 PyTorch 与 torchaudio，再安装其他依赖：

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

安装 `ffmpeg`：

```bash
sudo apt update
sudo apt install -y ffmpeg
```

推荐直接通过启动器进入项目：

```bash
python run.py doctor
python run.py
```

## 数据集

本仓库不提交 RAVDESS 数据集本体。请手动下载后放到：

```text
data/ravdess/
```

详细说明见 `data/README.zh.md`。

## 训练

```bash
python run.py train
```

## CLI 推理

```bash
python run.py predict --audio path/to/example.wav
```

## WebUI

```bash
python run.py
python run.py webui --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

WebUI 支持：

- 上传本地音频文件
- 浏览器麦克风录音
- 展示预测情绪、置信度和完整概率分布

预测功能需要本地模型文件 `output/best_model.pth`，或通过环境变量 `EMOTION_MODEL_PATH` 显式指定模型路径。

## 说明

- 本项目基于 MIT License 发布，详见 `LICENSE`。
- `data/ravdess/` 已加入 `.gitignore`，避免把原始数据集推到仓库
- 根目录旧的本地模型与图片产物已忽略，当前正式输出集中在 `output/`
- `output/` 目录仅保留结构，生成的模型和训练图不会提交到公开仓库
