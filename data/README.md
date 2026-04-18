# Dataset Placement

[中文](README.zh.md) | English

Place the RAVDESS dataset under:

```text
data/ravdess/
```

Expected structure:

```text
data/ravdess/Actor_01/*.wav
data/ravdess/Actor_02/*.wav
...
data/ravdess/Actor_24/*.wav
```

The dataset itself is ignored by Git and should not be committed to the repository.

For `python run.py train -- --dataset tess`, place the replacement vec dataset under:

```text
data/vec/
```

Expected layout:

```text
data/vec/anger/*.wav
data/vec/disgust/*.wav
data/vec/fear/*.wav
data/vec/happy/*.wav
data/vec/neutral/*.wav
data/vec/sad/*.wav
```
