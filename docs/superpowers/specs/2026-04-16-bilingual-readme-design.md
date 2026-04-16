# Bilingual README Design

## Goal

Add bilingual README coverage for every current README in the repository, with English as the default `README.md` and Chinese companion files named `README.zh.md`, plus a simple language switch at the top of each file.

## Current Scope

The repository currently has these README files:

- `/README.md`
- `/data/README.md`

After this change, the README set will be:

- `/README.md` (English)
- `/README.zh.md` (Chinese)
- `/data/README.md` (English)
- `/data/README.zh.md` (Chinese)

No other documentation files are included in this change.

## File Naming Rules

- English remains the default language for the canonical file name: `README.md`
- Chinese uses a sibling file named `README.zh.md`
- The naming pattern must be consistent in the repository root and subdirectories
- Existing relative links must remain valid inside each directory

## Top Language Switch

Each README pair will include a small language switch directly under the top-level heading.

English files use:

```md
[中文](README.zh.md) | English
```

Chinese files use:

```md
中文 | [English](README.md)
```

Formatting rules:

- Place the switch immediately below the `#` title
- Use plain Markdown links only
- The current language is plain text, not a self-link
- Links are always same-directory relative paths

## Content Structure Rules

Each language pair must be full counterparts with matching structure.

- `README.md` and `README.zh.md` describe the same project behavior, commands, and repository layout
- Section order stays aligned across both files
- Code blocks, commands, file paths, URLs, and filenames remain identical across languages unless the path text itself must change
- Product and tool names such as `Transformer Mood`, `FastAPI`, `RAVDESS`, and `ffmpeg` remain untranslated
- Short documents stay short in both languages; this is not a docs expansion pass

This structure is intended to make future maintenance mechanical: when one language changes, the corresponding section in the sibling file is easy to locate and update.

## Document-Specific Expectations

### Root README

The current root README is Chinese-first. It will be rewritten into English as `README.md`, preserving the current sections and commands:

- Project summary
- Features
- Repository layout
- Quick start
- Dataset
- Training
- CLI prediction
- WebUI
- Notes

`README.zh.md` will contain the Chinese version with the same section ordering.

### Data README

`/data/README.md` already exists in English and is short. It will remain English with the language switch added.

`/data/README.zh.md` will be added as a Chinese translation with the same concise structure:

- Dataset placement path
- Expected directory structure
- Reminder that the dataset is git-ignored and should not be committed

## Non-Goals

This change does not include:

- `ENVIRONMENT.md`
- Any non-README docs
- Any application code, UI, backend, or runtime changes
- Any documentation redesign beyond bilingual parity and the top language switch

## Risks And Mitigations

### Risk: English and Chinese files drift over time

Mitigation:

- Keep section order identical across language pairs
- Keep the files as direct counterparts rather than language-specific reorganizations

### Risk: Relative links break in subdirectories

Mitigation:

- Use same-directory links only: `README.md` and `README.zh.md`
- Avoid hardcoded root-relative paths for the language switch

### Risk: The default landing README changes user expectations

Mitigation:

- Preserve the existing content structure so only language defaults change
- Make the Chinese version immediately visible through the top switch

## Success Criteria

- Every current README has both English and Chinese versions
- English is the canonical `README.md` in each directory
- Chinese companion files are named `README.zh.md`
- Every README pair has a working top language switch
- Root and `data/` README pairs have matching structure across languages
