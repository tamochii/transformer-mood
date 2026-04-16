# Bilingual README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add English and Chinese README pairs for every current README, with English as `README.md` and Chinese as `README.zh.md`.

**Architecture:** Keep the existing README set small and parallel. Each README pair uses the same section order and a minimal top language switch. Root and `data/` are updated independently, with no code or runtime changes.

**Tech Stack:** Markdown, Gitea/GitHub README rendering

---

### Task 1: Root README Pair

**Files:**
- Create: `README.zh.md`
- Modify: `README.md`

- [ ] Replace the current Chinese-first `README.md` with the English version.
- [ ] Add the top switch under the H1 in `README.md`: `[中文](README.zh.md) | English`.
- [ ] Create `README.zh.md` as the Chinese counterpart with the top switch: `中文 | [English](README.md)`.
- [ ] Keep the root README pair aligned section-by-section: summary, features, layout, quick start, dataset, training, CLI prediction, WebUI, notes.

### Task 2: Data README Pair

**Files:**
- Create: `data/README.zh.md`
- Modify: `data/README.md`

- [ ] Add the top switch under the H1 in `data/README.md`: `[中文](README.zh.md) | English`.
- [ ] Create `data/README.zh.md` with the top switch: `中文 | [English](README.md)`.
- [ ] Keep both `data/` README files short and structurally equivalent.

### Task 3: Verification

**Files:**
- Verify: `README.md`
- Verify: `README.zh.md`
- Verify: `data/README.md`
- Verify: `data/README.zh.md`

- [ ] Read all four README files and confirm that each switch link targets an existing sibling file.
- [ ] Confirm the English files are the canonical `README.md` files.
- [ ] Confirm commands, paths, and repository-specific names are preserved consistently across language pairs.
