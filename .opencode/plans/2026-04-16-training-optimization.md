# Training Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve emotion recognition accuracy from ~41% to 50%+ on combined RAVDESS + CREMA-D dataset through feature engineering and training strategy improvements.

**Architecture:** Enhance the existing `speech_emotion_classifier.py` with CMVN normalization, delta features, SpecAugment, label smoothing, LR warmup, Mixup augmentation, and weighted sampling. No architecture changes, no new files.

**Tech Stack:** PyTorch, torchaudio, librosa, numpy

---

## File Structure

All changes are in a single file:
- **Modify:** `src/transformer_mood/speech_emotion_classifier.py` — feature extraction, augmentation, training loop, data loading

Test files remain unchanged (existing tests should continue passing).

---

### Task 1: Add CMVN Normalization and Delta Features

**Files:**
- Modify: `src/transformer_mood/speech_emotion_classifier.py:54-56` (constants)
- Modify: `src/transformer_mood/speech_emotion_classifier.py:286-361` (SpeechEmotionDataset)

- [ ] **Step 1: Update global constants**

Change `N_MELS` usage and add feature dim constant:

```python
# Add after line 56 (MAX_SEQ_LEN = 200)
USE_DELTA = True          # 是否使用 delta 特征
FEATURE_DIM = N_MELS * 3 if USE_DELTA else N_MELS  # 384 or 128
```

- [ ] **Step 2: Modify `__getitem__` to add CMVN normalization and delta features**

Replace the current feature extraction section in `__getitem__` (lines 324-329):

```python
    def __getitem__(self, idx):
        sample = self.samples[idx]
        filepath = sample["filepath"]
        label = sample["emotion_idx"]

        # 1. 加载音频
        waveform = load_audio(filepath)

        # 2. 数据增强 - 音频级（仅训练时）
        if self.augment:
            waveform = self._augment(waveform)

        # 3. 提取 Mel-spectrogram
        mel_spec = extract_mel_spectrogram(waveform)  # (n_mels, time_frames)

        # 4. CMVN 归一化（per-utterance zero-mean unit-variance）
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        # 5. Delta 特征拼接
        if USE_DELTA:
            mel_np = mel_spec.numpy()
            delta1 = librosa.feature.delta(mel_np, width=9)
            delta2 = librosa.feature.delta(mel_np, order=2, width=9)
            feat_np = np.concatenate([mel_np, delta1, delta2], axis=0)  # (3*n_mels, time)
            feat = torch.from_numpy(feat_np).T.float()  # (time, 3*n_mels)
        else:
            feat = mel_spec.T  # (time, n_mels)

        seq_len = feat.shape[0]
        feat_dim = feat.shape[1]

        # 6. SpecAugment 数据增强（仅训练时，在频谱级别）
        if self.augment:
            feat = self._spec_augment(feat)

        # 7. 创建 padding mask
        mask = torch.ones(self.max_seq_len, dtype=torch.bool)

        # 8. 截断或填充至固定长度
        if seq_len > self.max_seq_len:
            feat = feat[:self.max_seq_len, :]
        elif seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            padding = torch.zeros(pad_len, feat_dim)
            feat = torch.cat([feat, padding], dim=0)
            mask[seq_len:] = False

        label_tensor = torch.tensor(label, dtype=torch.long)
        return feat, label_tensor, mask
```

- [ ] **Step 3: Add `_spec_augment` method to `SpeechEmotionDataset`**

Add after the existing `_augment` method (after line 360):

```python
    def _spec_augment(self, feat: torch.Tensor) -> torch.Tensor:
        """
        SpecAugment: 频率遮蔽 + 时间遮蔽。
        
        Args:
            feat: shape=(time, feat_dim)
        Returns:
            增强后的特征
        """
        feat = feat.clone()
        time_len, feat_dim = feat.shape
        
        # 频率遮蔽: 随机遮挡一段连续的频率维度
        f_mask_width = random.randint(0, min(27, feat_dim // 4))
        if f_mask_width > 0:
            f_start = random.randint(0, feat_dim - f_mask_width)
            feat[:, f_start:f_start + f_mask_width] = 0.0
        
        # 时间遮蔽: 随机遮挡一段连续的时间帧
        t_mask_width = random.randint(0, min(40, time_len // 4))
        if t_mask_width > 0:
            t_start = random.randint(0, time_len - t_mask_width)
            feat[t_start:t_start + t_mask_width, :] = 0.0
        
        return feat
```

- [ ] **Step 4: Update model instantiation to use FEATURE_DIM**

In `main()`, change line 919:
```python
# Before:
model = SpeechEmotionClassifier(input_dim=N_MELS).to(DEVICE)
# After:
model = SpeechEmotionClassifier(input_dim=FEATURE_DIM).to(DEVICE)
```

Also in predict mode, line 810:
```python
# Before:
model = SpeechEmotionClassifier(input_dim=N_MELS).to(DEVICE)
# After:
model = SpeechEmotionClassifier(input_dim=FEATURE_DIM).to(DEVICE)
```

- [ ] **Step 5: Update `predict_single` to use CMVN + delta**

Replace lines 621-623 in `predict_single`:
```python
    waveform = load_audio(audio_path)
    mel_spec = extract_mel_spectrogram(waveform)
    
    # CMVN 归一化
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
    
    # Delta 特征
    if USE_DELTA:
        mel_np = mel_spec.numpy()
        delta1 = librosa.feature.delta(mel_np, width=9)
        delta2 = librosa.feature.delta(mel_np, order=2, width=9)
        feat_np = np.concatenate([mel_np, delta1, delta2], axis=0)
        feat = torch.from_numpy(feat_np).T.float()  # (time, 3*n_mels)
    else:
        feat = mel_spec.T  # (time, n_mels)
```

Also update the padding section to use `feat.shape[1]` for pad width.

- [ ] **Step 6: Update model config save to include FEATURE_DIM**

In `main()`, update the config dict (line 1019):
```python
"input_dim": FEATURE_DIM,  # was N_MELS
```

- [ ] **Step 7: Run existing tests**

Run: `cd /home/chius/repo/gitea/transformer-mood && PYTHONPATH=src python -m pytest tests/ -v`
Expected: All 7 tests pass (no test touches feature extraction internals)

- [ ] **Step 8: Verify import and basic functionality**

Run: `cd /home/chius/repo/gitea/transformer-mood && PYTHONPATH=src python -c "from transformer_mood.speech_emotion_classifier import SpeechEmotionClassifier, FEATURE_DIM; print(f'FEATURE_DIM={FEATURE_DIM}'); m = SpeechEmotionClassifier(input_dim=FEATURE_DIM); print(f'Params: {sum(p.numel() for p in m.parameters()):,}')"`
Expected: `FEATURE_DIM=384`, params count slightly larger than before

- [ ] **Step 9: Commit**

```bash
git add src/transformer_mood/speech_emotion_classifier.py
git commit -m "feat: add CMVN normalization, delta features, and SpecAugment"
```

---

### Task 2: Improve Training Strategy (Label Smoothing + Warmup + Mixup)

**Files:**
- Modify: `src/transformer_mood/speech_emotion_classifier.py:548-577` (train_one_epoch)
- Modify: `src/transformer_mood/speech_emotion_classifier.py:928-953` (main training setup)

- [ ] **Step 1: Add Label Smoothing to CrossEntropyLoss**

In `main()`, change the criterion line (line 939):
```python
# Before:
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# After:
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
```

- [ ] **Step 2: Replace scheduler with Warmup + Cosine**

Replace the scheduler setup (line 941) with:
```python
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
```

- [ ] **Step 3: Add Mixup to `train_one_epoch`**

Replace the `train_one_epoch` function entirely:

```python
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    """单个 epoch 的训练（含 Mixup 增强）。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (features, labels, masks) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # Mixup 数据增强（50% 概率）
        use_mixup = random.random() < 0.5
        if use_mixup:
            lam = np.random.beta(0.2, 0.2)
            indices = torch.randperm(features.size(0), device=device)
            mixed_features = lam * features + (1 - lam) * features[indices]
            mixed_masks = masks | masks[indices]  # 两个 mask 取并集
            logits = model(mixed_features, mask=mixed_masks)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[indices])
        else:
            logits = model(features, mask=masks)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
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
```

- [ ] **Step 4: Increase early stopping patience**

In `main()`, change patience (line 951):
```python
# Before:
patience = 15
# After:
patience = 20
```

- [ ] **Step 5: Run existing tests**

Run: `cd /home/chius/repo/gitea/transformer-mood && PYTHONPATH=src python -m pytest tests/ -v`
Expected: All 7 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/transformer_mood/speech_emotion_classifier.py
git commit -m "feat: add label smoothing, LR warmup, and Mixup augmentation"
```

---

### Task 3: Add WeightedRandomSampler for Class Balancing

**Files:**
- Modify: `src/transformer_mood/speech_emotion_classifier.py:897-907` (DataLoader setup in main)

- [ ] **Step 1: Add WeightedRandomSampler import and setup**

Add import at top (around line 32):
```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
```

In `main()`, replace the train DataLoader setup (lines 902-903) with:

```python
    # 构建 WeightedRandomSampler 实现类别平衡采样
    train_labels = [s["emotion_idx"] for s in train_samples]
    class_sample_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    sample_weights = 1.0 / (class_sample_counts[train_labels] + 1e-6)
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
```

Note: `sampler` and `shuffle` are mutually exclusive — remove `shuffle=True` when using sampler.

- [ ] **Step 2: Run existing tests**

Run: `cd /home/chius/repo/gitea/transformer-mood && PYTHONPATH=src python -m pytest tests/ -v`
Expected: All 7 tests pass

- [ ] **Step 3: Commit**

```bash
git add src/transformer_mood/speech_emotion_classifier.py
git commit -m "feat: add WeightedRandomSampler for class-balanced training"
```

---

### Task 4: Integration Test — Full Training Run

**Files:**
- No code changes — this is a validation task

- [ ] **Step 1: Run a quick 10-epoch training to verify all changes work together**

Run: `cd /home/chius/repo/gitea/transformer-mood && PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode train --epochs 10`

Expected:
- No errors or crashes
- FEATURE_DIM=384 reported
- Warmup visible in first 5 epochs (LR increasing)
- Class weights computed correctly
- WeightedRandomSampler active
- Training loss decreasing

- [ ] **Step 2: Check output files generated**

Run: `ls -la /home/chius/repo/gitea/transformer-mood/output/`

Expected: best_model.pth, training_curves.png, confusion_matrix.png, emotion_distribution.png

- [ ] **Step 3: Run full 60-epoch training**

Run: `cd /home/chius/repo/gitea/transformer-mood && PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode train --epochs 60`

Expected: Test accuracy ≥ 50%

- [ ] **Step 4: Test predict mode with new model**

Run: `cd /home/chius/repo/gitea/transformer-mood && PYTHONPATH=src python -m transformer_mood.speech_emotion_classifier --mode predict --audio data/ravdess/Actor_01/03-01-01-01-01-01-01.wav`

Expected: Prediction with confidence scores, no errors

- [ ] **Step 5: Final test suite run**

Run: `cd /home/chius/repo/gitea/transformer-mood && PYTHONPATH=src python -m pytest tests/ -v`
Expected: All 7 tests pass

- [ ] **Step 6: Commit any final adjustments**

```bash
git add -A
git commit -m "feat: complete training optimization — CMVN, delta, SpecAugment, Mixup, warmup"
```
