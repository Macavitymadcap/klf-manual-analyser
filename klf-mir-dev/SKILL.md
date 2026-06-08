---
name: klf-mir-dev
description: Development assistant for the KLF Manual Analyser project — a Python MIR (Music Information Retrieval) tool that scores MP3s against hit-song criteria from The KLF's 1988 book The Manual. Use this skill whenever working on any part of this codebase: writing analysis modules (librosa, essentia, madmom, msaf), the SQLite schema (db.py), the hybrid structure alignment logic, criteria TOML config files, LLM scoring prompts, the Jinja2 HTML report, or the CLI. Also use for debugging MIR library compatibility issues (madmom Python 3.11, msaf scipy), reviewing criteria logic, or discussing design decisions. The project uses Python 3.11+ with uv, SQLite, optional Qdrant, and Ollama (qwen2.5:14b, nomic-embed-text) running locally on Fedora with an ASUS ROG Strix / RTX 5070 Ti.
---

# KLF Manual Analyser — Development Skill

A context-loading skill for working on the KLF Manual Analyser project.
Read the relevant reference file before writing any code for a given module.

---

## Project summary

A Python CLI tool that:
1. Accepts a folder of MP3s named `Artist_Name-Song_Title.mp3`
2. Separates each track into stems (Demucs htdemucs)
3. Extracts acoustic features (librosa, essentia, madmom/fallback, msaf/fallback)
4. Transcribes lyrics (openai-whisper large-v3)
5. Aligns lyrics with acoustic structure (hybrid section labelling)
6. Optionally embeds feature summaries (nomic-embed-text → Qdrant)
7. Scores each track per criterion (deterministic + Ollama/qwen2.5:14b)
8. Aggregates across all tracks into a "recipe" for a matching song
9. Renders an HTML report with in-browser audio playback (Jinja2 + vanilla JS)

Three scoring modes: `1988`, `contemporary`, `1920s_1930s`.

---

## Reference files

Read the relevant file before writing code. Do not guess schema or module API.

| File | Read when working on |
|---|---|
| `references/architecture.md` | Pipeline, module structure, data flow |
| `references/schema.md` | SQLite tables, Qdrant collection, field types |
| `references/criteria.md` | Criterion definitions, rules, weights, prompt hints |
| `references/compatibility.md` | Known library issues and fallback strategies |

---

## Key conventions

**Language**: Python 3.11+, managed with uv. No TypeScript, no Bun.

**Field naming**: `verse_chorus_delta` (not `_db`) is normalised 0.0–1.0.
Physical unit fields: `bpm`, `duration`, `start`, `end`, `hook_first_appearance`.
Everything else normalised.

**Rule types in TOML**: `lte`, `gte`, `eq`, `range`, `exists`, `llm`.
`exists` checks whether any `sections` row for the track has `label == value`.
`db_field` (string) for single-field criteria; `db_fields` (array) for multi-field.
Mutually exclusive — validated at load time in `criteria.py`.

**Section labels**: `intro`, `verse`, `pre_chorus`, `chorus`, `breakdown`,
`double_chorus`, `bridge`, `outro`, `unknown`.

**Qdrant is optional**: always wrap Qdrant calls in try/except; skip gracefully
if unavailable. Core pipeline works without it.

**madmom and msaf are high-risk**: both may fail to install on Python 3.11.
Every module using them must have a librosa-based fallback. Test compatibility
before writing any dependent code.

**Chord detection accuracy**: ~70–75% on modern recordings, much lower on 1920s
material. Never treat chord data as ground truth. All harmony prompt hints must
acknowledge this explicitly.

**Scoring**: deterministic rules (`lte`, `gte`, `eq`, `range`, `exists`) require
no LLM call. `llm` rules call Ollama with a constructed prompt and expect JSON
response `{"score": 0-10, "reasoning": "..."}`. Score is normalised to 0.0–1.0.

**LLM calls**: use Ollama's OpenAI-compatible endpoint at `http://localhost:11434`.
No SDK required — plain `httpx` or `requests`. Default model: `qwen2.5:14b`.
Retry once with stricter JSON prompt if response fails to parse.

**HTML report aesthetic**: punky, anarchistic, KLF/Illuminatus! visual language.
Black, white, one aggressive accent colour. ALL CAPS proclamations. Jinja2
templates leave explicit hooks (element IDs, data attributes) for vanilla JS
audio player to attach to.

---

## Common tasks

### Adding a new criterion
1. Read `references/criteria.md` for the rule type schema
2. Add `[[criterion]]` block to the relevant `config/criteria_*.toml`
3. If `llm` rule: ensure all `db_fields` exist in the schema (`references/schema.md`)
4. If `exists` rule: ensure the label value is in the valid section labels list
5. Run `criteria.py` load validation

### Writing an analysis module
1. Read `references/architecture.md` for the module's expected inputs/outputs
2. Read `references/schema.md` for the exact DB fields it writes
3. Read `references/compatibility.md` for library-specific gotchas
4. Module writes directly to SQLite — no return values to merge
5. Include fallback if the primary library (madmom/msaf) is unavailable

### Writing a scoring prompt
1. Read `references/criteria.md` for the criterion's `prompt_hint`
2. Build prompt in `scoring/prompt.py` — inject actual field values from SQLite
3. System prompt must be framed in the mode's era context (1988 / streaming / 1920s)
4. Response must be JSON only: `{"score": int, "reasoning": str}`

### Debugging library compatibility
1. Read `references/compatibility.md` first
2. Test madmom and msaf import separately before testing together
3. If a fallback is needed, document it in `references/compatibility.md`

---

## Hardware context

- Fedora 44, Python 3.11, uv package manager
- ASUS ROG Strix G615LR, RTX 5070 Ti (12GB VRAM) — CUDA available
- Ollama 0.24.0 with: qwen2.5:14b, qwen2.5:7b, mistral-nemo:12b, nomic-embed-text
- Demucs and Whisper will use CUDA automatically if available
