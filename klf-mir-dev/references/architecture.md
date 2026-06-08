# Architecture Reference

## Pipeline stage → module mapping

| Stage | Module | Inputs | SQLite writes |
|---|---|---|---|
| 1: Decode | `audio/decode.py` | MP3 path | INSERT tracks (id, filename, artist, song_name, duration) |
| 2: Separate | `audio/separate.py` | full.wav | filesystem only (stems/) |
| 3a: Tempo | `analysis/tempo.py` | full.wav | UPDATE tracks (bpm, bpm_confidence, time_signature, tempo_stability) |
| 3a: Rhythm | `analysis/rhythm.py` | drums.wav | INSERT beat_patterns; UPDATE tracks (groove_feel) |
| 3a: Energy | `analysis/energy.py` | full.wav | UPDATE tracks (loudness_db, dynamic_range_db, verse_chorus_delta, energy_shape); INSERT tracks_timeseries |
| 3a: Harmony | `analysis/harmony.py` | other.wav + full.wav | UPDATE tracks (key, mode, key_confidence); INSERT sections (initial, label=unknown); INSERT chord_progressions |
| 3a: Groove | `analysis/groove.py` | full.wav | UPDATE tracks (danceability, self_similarity_score, beat_regularity, groove_consistency, repetition_score) |
| 3a: Structure | `analysis/structure.py` (pass 1) | full.wav | UPDATE sections (start, end, duration — boundaries only) |
| 3b: Whisper | `transcription/whisper.py` | vocals.wav | INSERT transcript_segments; UPDATE tracks (unique_word_ratio, hook_repetition_count, hook_first_appearance, hook_phrase) |
| 4: Alignment | `analysis/structure.py` (pass 2) | SQLite reads | UPDATE sections (label, label_confidence, label_source, mean_energy, lyric_density, repeated_phrase) |
| 5: Embed | `embedding/embed.py` | SQLite reads | UPDATE tracks (feature_summary); INSERT track_vectors |
| 6: Scoring | `scoring/` | SQLite reads | INSERT scores |
| 7: Aggregate | `aggregation/aggregate.py` | SQLite + Qdrant | — (outputs to render) |
| 8: Report | `report/render.py` | SQLite reads | writes HTML to data/reports/ |

## Module file structure

```
src/manual_analyser/
├── cli.py              -- typer entrypoint; orchestrates pipeline
├── pipeline.py         -- per-track orchestration; caching checks; progress
├── db.py               -- connection, schema CREATE, migrations
│
├── audio/
│   ├── decode.py       -- ffmpeg subprocess; filename parsing
│   └── separate.py     -- demucs Python API; CUDA auto-detection
│
├── analysis/
│   ├── tempo.py        -- librosa.beat.beat_track; madmom fallback
│   ├── rhythm.py       -- madmom BeatTrackingProcessor; librosa fallback
│   ├── structure.py    -- msaf; librosa.segment fallback; alignment pass
│   ├── energy.py       -- librosa.feature.rms; pyloudnorm for LUFS
│   ├── harmony.py      -- librosa chroma; chord estimation
│   └── groove.py       -- essentia Danceability; librosa self-similarity
│
├── transcription/
│   └── whisper.py      -- openai-whisper; hook phrase extraction
│
├── embedding/
│   ├── summarise.py    -- builds human-readable feature summary text
│   └── embed.py        -- nomic-embed-text via Ollama; Qdrant upsert
│
├── scoring/
│   ├── criteria.py     -- TOML loader; rule validation; db_field/db_fields check
│   ├── prompt.py       -- assembles LLM prompts from SQLite values + criteria
│   └── llm.py          -- Ollama HTTP calls; JSON parse; retry logic
│
├── aggregation/
│   └── aggregate.py    -- SQL aggregation; Qdrant cluster; recipe LLM call
│
└── report/
    ├── render.py       -- Jinja2 render; writes HTML to data/reports/
    ├── server.py       -- stdlib http.server; serves reports/ and stems/
    └── templates/
        ├── base.html
        ├── summary.html
        └── track.html
```

## Caching logic (pipeline.py)

```
for each track:
  if not stems_exist(track_id): run_demucs()
  if not track_in_db(track_id): run_analysis_modules()
  if not transcript_in_db(track_id): run_whisper()
  if not sections_labelled(track_id): run_alignment()
  if qdrant_available() and not vector_in_qdrant(track_id): run_embedding()
  if not scores_exist(track_id, mode): run_scoring()
```

`--no-cache` bypasses all checks. `track_id` = MD5 of `path + filename`.

## Error handling rules (see docs/ERROR_HANDLING.md for full policy)

- **Per-track isolation**: failure on one track never aborts others
- **Hard aborts**: ffmpeg missing, Ollama missing, Demucs model missing
- **Per-track skip**: decode failure, all analysis modules fail
- **Partial track**: individual module failure → null fields → partial score
- **Library fallback**: madmom/msaf/essentia fail → librosa fallback → log INFO
- **Qdrant**: always optional; wrap in try/except; log once if unavailable
- **LLM JSON parse fail**: retry once with stricter prompt; then write score=null
- **SQLite writes**: always use `with conn:` transaction pattern; auto-rollback on exception
- **Null fields in scoring**: treat as missing data, not zero; LLM prompts must handle

Track statuses: `pending`, `decoding`, `separating`, `analysing`, `transcribing`,
`aligning`, `embedding`, `scoring`, `complete`, `skipped`, `partial`

Log format: `[TRACK_ID_SHORT] [STAGE] message` — written to `data/pipeline.log`

## clean command path filtering

`manual-analyser clean [path] [--stems] [--features] [--reports]`

If `path` is provided, only remove records/files derived from MP3s in that path.
If omitted, remove all. Prevents accidental deletion across multiple input sets.
