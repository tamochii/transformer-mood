# Transformer Mood

基于 Transformer 的语音情绪识别项目，包含训练脚本、命令行推理，以及一个可直接上传音频或使用麦克风录音的本地 WebUI。

## Features

- RAVDESS 数据集预处理与训练
- Transformer Encoder 语音情绪分类模型
- 命令行单文件推理
- FastAPI WebUI
- 浏览器录音、音频上传、情绪概率展示

## Repository Layout

```text
speech_emotion_classifier.py   # 训练与 CLI 推理入口
app_fastapi.py                 # FastAPI WebUI 服务
templates/index.html           # WebUI 页面模板
output/                        # 已生成的模型与可视化结果
data/README.md                 # 数据集放置说明
transformer-md/                # 参考资料
requirements.txt               # 非 PyTorch Python 依赖
requirements-webui.txt         # WebUI 最小额外依赖
ENVIRONMENT.md                 # 环境与运行说明
```

## Quick Start

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

## Dataset

本仓库不提交 RAVDESS 数据集本体。请手动下载后放到：

```text
data/ravdess/
```

详细说明见 `data/README.md`。

## Training

```bash
python speech_emotion_classifier.py --mode train
```

## CLI Prediction

```bash
python speech_emotion_classifier.py --mode predict --audio path/to/example.wav
```

## WebUI

```bash
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

WebUI 支持：

- 上传本地音频文件
- 浏览器麦克风录音
- 展示预测情绪、置信度和完整概率分布

## Notes

- `data/ravdess/` 已加入 `.gitignore`，避免把原始数据集推到仓库
- 根目录旧的本地模型与图片产物已忽略，当前正式输出集中在 `output/`
- `output/` 中的文件可作为当前项目结果一并上传，或者按你的需要重新生成
