# 数据集放置

中文 | [English](README.md)

请将 RAVDESS 数据集放到：

```text
data/ravdess/
```

目录结构应为：

```text
data/ravdess/Actor_01/*.wav
data/ravdess/Actor_02/*.wav
...
data/ravdess/Actor_24/*.wav
```

数据集本体已被 Git 忽略，不应提交到仓库中。

如果使用 `python run.py train -- --dataset tess`，请将替换后的 vec 数据集放到：

```text
data/vec/
```

目录结构应为：

```text
data/vec/anger/*.wav
data/vec/disgust/*.wav
data/vec/fear/*.wav
data/vec/happy/*.wav
data/vec/neutral/*.wav
data/vec/sad/*.wav
```
