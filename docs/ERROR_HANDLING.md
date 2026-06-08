# Error Handling Policy

KLF Manual Analyser — pipeline error handling, failure modes, and recovery
strategy.

This document defines how the pipeline behaves when things go wrong. All
pipeline code must follow these rules. Inconsistent error handling is worse
than no error handling — a partial SQLite record and a missing stem directory
left by a crashed run must never produce a silently corrupted report.

---

## Core principle: per-track isolation

**A failure on one track must never abort the run for other tracks.**

The pipeline processes 40 tracks. If track 12 crashes Whisper, tracks 1–11 and
13–40 should complete normally. Track 12 should be marked as failed and
reported clearly at the end. The report renders with 39 tracks; track 12 appears
in a "failed tracks" section with the error logged.

This is the single most important behavioural rule. It follows from the fact that
the most likely errors (Demucs OOM on an unusual track, Whisper failing on a
corrupt audio file, a chord detection library segfault) are track-specific, not
systemic.

---

## Failure modes and responses

### Stage 1: Decode (ffmpeg)

| Failure | Response |
|---|---|
| File not found | Skip track; log error; continue |
| ffmpeg not on PATH | Abort entire run with clear install message |
| ffmpeg returns non-zero | Skip track; log stderr; continue |
| Malformed MP3 (ffmpeg errors) | Skip track; log error; continue |
| Filename does not match convention | Accept track; store `artist=null`, `song_name=filename stem`; emit warning (not error) |

ffmpeg absence is a hard abort because no track can be processed without it.
All other decode failures are per-track skips.

---

### Stage 2: Separate (Demucs)

| Failure | Response |
|---|---|
| CUDA out of memory | Retry on CPU (set `device="cpu"` and retry once); if CPU also fails, skip track |
| Demucs model not found | Abort entire run with message to run `python -m demucs.separate --help` |
| Demucs produces incomplete stems | Skip track; log which stems are missing |
| Demucs segfault / unhandled exception | Skip track; log full traceback |

CUDA OOM is expected on very long tracks or certain encodings. The CPU retry
prevents losing valid tracks due to GPU memory pressure. The CPU fallback will
be slow (~5–10× slower) — emit a warning when it triggers.

---

### Stage 3a: Analysis modules

Each module (`tempo.py`, `rhythm.py`, `energy.py`, `harmony.py`, `groove.py`,
`structure.py`) runs independently. A failure in one module does not skip the
others.

| Failure | Response |
|---|---|
| Library import error (madmom/msaf/essentia) | Fall back to librosa implementation; log which fallback was used; continue |
| Numerical error (e.g. division by zero, NaN) | Write `null` for affected fields; log warning; continue |
| Unhandled exception in any module | Log full traceback; write `null` for all fields that module was responsible for; continue |
| All analysis modules fail for a track | Mark track as `analysis_failed`; skip scoring; include in failed tracks report |

**The null-field policy**: SQLite columns are nullable precisely to support
partial analysis. A track with `bpm=null` can still be scored on `structure`,
`chorus_hook`, and `groove`. The scoring layer must handle null fields
gracefully — treat them as missing data, not as zero.

---

### Stage 3b: Transcription (Whisper)

| Failure | Response |
|---|---|
| Whisper model not downloaded | Abort stage for this track; emit message to run `whisper --model large-v3`; continue with other stages |
| CUDA OOM | Retry on CPU once; if still fails, skip transcription for this track |
| Audio too short (< 2 seconds) | Skip transcription; write empty transcript; log warning |
| No speech detected | Write empty transcript; set `hook_phrase=null`; log info (not warning — instrumentals are valid) |
| Unhandled exception | Log traceback; skip transcription for this track; continue |

A track without transcription can still be scored on acoustic criteria.
LLM criteria that use transcript data (`chorus_hook`, `lyrics_economy`,
`hook_refrain`) will receive `null` for lyric fields — the LLM prompt
must acknowledge this explicitly.

---

### Stage 4: Hybrid structure alignment

| Failure | Response |
|---|---|
| No sections in DB for this track (analysis failed) | Skip alignment; log warning |
| All sections remain "unknown" after alignment | Accept; pass "unknown" labels to LLM with low confidence scores |
| Unhandled exception | Log traceback; leave sections with label="unknown"; continue |

The alignment pass produces low-confidence labels rather than errors. A track
where all sections are "unknown" will score poorly on `structure` criteria —
this is correct behaviour, not a bug.

---

### Stage 5: Embedding (Qdrant) — optional

| Failure | Response |
|---|---|
| Qdrant not running / connection refused | Skip stage for all tracks; log info once; continue |
| Qdrant returns error on upsert | Log warning for this track; continue |
| nomic-embed-text not available in Ollama | Skip stage for all tracks; log info once; continue |
| Unhandled exception | Log traceback; skip embedding for this track; continue |

Qdrant is optional. All Qdrant code is wrapped in try/except. A single
"Qdrant unavailable — similarity features will be absent" log line is
emitted once at pipeline start if Qdrant is not reachable.

---

### Stage 6: Scoring (LLM)

| Failure | Response |
|---|---|
| Ollama not running | Abort entire run with clear message to start Ollama |
| Model not pulled | Abort entire run with message to run `ollama pull qwen2.5:14b` |
| LLM response not valid JSON | Retry once with stricter prompt; if still fails, write `score=null, reasoning="parse_error"` |
| LLM response score out of range (< 0 or > 10) | Clamp to 0–10; log warning |
| Ollama timeout (> 30 seconds) | Retry once; if still times out, write `score=null, reasoning="timeout"` |
| All criteria fail to score for a track | Mark track as `scoring_failed`; include in failed tracks report |

Ollama absence is a hard abort for scoring (the whole stage is LLM-dependent).
Individual criterion failures produce null scores rather than aborting.

Deterministic criteria (`lte`, `gte`, `range`, `exists`) never produce LLM
errors — they are pure SQLite queries and can only fail if the relevant field
is null (see null-field policy above).

---

### Stage 7: Aggregation

| Failure | Response |
|---|---|
| Fewer than 2 scored tracks | Abort aggregation; emit warning; still render per-track report |
| LLM recipe generation fails | Include error message in recipe section of report; render report without recipe |
| Qdrant cluster query fails | Skip cluster features in report; log warning |

---

### Stage 8: Report render

| Failure | Response |
|---|---|
| Jinja2 template error | Abort render; log full traceback with template name and line |
| Output directory not writable | Abort render; emit clear error with path |
| Missing data for a track (nulls) | Render with "N/A" placeholders; do not abort |

---

## Track status model

Each track in the pipeline has one of these statuses, stored in memory
during a run (not persisted to SQLite):

| Status | Meaning |
|---|---|
| `pending` | Not yet processed |
| `decoding` | Stage 1 in progress |
| `separating` | Stage 2 in progress |
| `analysing` | Stage 3 in progress |
| `transcribing` | Stage 3b in progress |
| `aligning` | Stage 4 in progress |
| `embedding` | Stage 5 in progress |
| `scoring` | Stage 6 in progress |
| `complete` | All stages completed successfully |
| `skipped` | Decode failed — track unprocessable |
| `partial` | One or more stages failed; track has partial data; will appear in report with caveats |

`partial` tracks are included in the report. Their failed stages are clearly
marked. Criteria that depended on failed stages show "N/A" rather than a score.
They are excluded from aggregate statistics for criteria where their data is null.

---

## SQLite write atomicity

Each stage writes its data in a single transaction. If a stage fails
mid-write, the transaction is rolled back. The next run will re-run that
stage (because the caching check will find incomplete data).

```python
# Pattern all analysis modules must follow:
try:
    with conn:                          # context manager = auto commit/rollback
        conn.execute("UPDATE tracks ...")
        conn.execute("INSERT INTO beat_patterns ...")
except Exception as e:
    logger.error(f"[{track_id}] rhythm analysis failed: {e}")
    # Do not re-raise — caller continues to next module
```

The `with conn:` pattern uses SQLite's implicit transaction. If any statement
in the block raises, the entire block is rolled back automatically.

---

## Logging

All errors and warnings are logged via Python's standard `logging` module.

Log levels used:
- `DEBUG` — detailed per-field values, useful during development
- `INFO` — stage start/complete, optional feature skips (Qdrant, fallbacks)
- `WARNING` — per-track failures, null fields, fallback library used
- `ERROR` — stage failure with traceback
- `CRITICAL` — hard abort (ffmpeg missing, Ollama missing)

Log format: `[TRACK_ID_SHORT] [STAGE] message`
where `TRACK_ID_SHORT` is the first 8 characters of the MD5 track_id.

The Rich progress display in the terminal shows a summary count of
`complete / partial / skipped` at the end of each stage. Full logs
are written to `data/pipeline.log`.

---

## End-of-run summary

After all tracks are processed, the CLI emits a summary table:

```
Analysis complete
─────────────────────────────────────
  Complete:  37 tracks
  Partial:    2 tracks  (see below)
  Skipped:    1 track   (see below)
─────────────────────────────────────

Partial tracks:
  The_KLF-Doctorin_The_Tardis   whisper_failed, embedding_skipped
  Louis_Armstrong-Heebie_Jeebies  rhythm_fallback_used

Skipped tracks:
  corrupt_file.mp3  decode_failed: ffmpeg returned non-zero exit code
```

This gives the user a clear picture of what ran successfully before they open
the report.
