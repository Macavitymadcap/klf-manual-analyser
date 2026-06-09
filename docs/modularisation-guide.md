# Modularisation Guide: Remaining Splits

Three modules left: `analysis/harmony`, `analysis/structure`, and
`transcription`. This guide covers exactly what goes where, what the
`__init__.py` of each subpackage must re-export, and what test changes
are needed.

The pattern throughout is identical to what was done with `groove`:

1. Create a directory, move cohesive groups of functions into named files
2. Write an `__init__.py` that re-exports **everything the tests currently
   import** — so no test file needs editing
3. Delete the original flat `.py` file

---

## 1. `analysis/harmony/`

### Why split

`harmony.py` does three distinct jobs in sequence:

- **Key detection** — Krumhansl-Schmuckler profiles, pure signal processing
- **Chord estimation** — template matching per section, chord templates as data
- **Section / DB writes** — reads existing sections, writes skeletons and
  chord progressions

The key profiles and chord templates are large constant data; separating them
makes it obvious where to look when tuning them.

### New file layout

```
src/manual_analyser/analysis/harmony/
├── __init__.py       ← orchestration + re-exports
├── keys.py           ← key/mode detection
├── chords.py         ← chord templates + estimation
└── sections.py       ← section boundary helpers + SQLite writes
```

### `harmony/keys.py`

**Move here:**
- `NOTE_NAMES`
- `_MAJOR_PROFILE`, `_MINOR_PROFILE`
- `_detect_key(chroma)` — returns `(key, mode, confidence)`

**Imports needed:** `numpy`

---

### `harmony/chords.py`

**Move here:**
- `_CHORD_TEMPLATES` (the build loop at module level)
- `ChordEvent` dataclass
- `_match_chord(frame)` — single-frame template match; imported by tests
- `_estimate_chords(section_chroma, section_start, sr, hop_length)` — returns
  `list[ChordEvent]`
- `_chords_to_progression(chords)` — returns compact string; imported by tests

**Imports needed:** `numpy`, `NOTE_NAMES` from `.keys`

---

### `harmony/sections.py`

**Move here:**
- `_get_section_boundaries(db_path, track_id, duration, n_fallback)` —
  imported by tests
- `_write_result(db_path, track_id, result, short_id)` — writes to tracks,
  sections, chord_progressions
- `_write_nulls(db_path, track_id, short_id)`

**Imports needed:** `json`, `pathlib.Path`, `get_connection` from `db`,
`HarmonyResult`/`SectionHarmony` (or just the result types — forward-import
these from `__init__.py` or keep dataclasses in `__init__.py`)

> **Note on dataclass placement**: `ChordEvent` and `SectionHarmony` are used
> by both `chords.py` and `sections.py`. Put `ChordEvent` in `chords.py` and
> `SectionHarmony` in `__init__.py` (alongside `HarmonyResult`) to avoid a
> circular import. `sections.py` imports `SectionHarmony` from `..harmony`
> (the package `__init__`).

---

### `harmony/__init__.py`

Contains:
- `SectionHarmony` and `HarmonyResult` dataclasses (shared types)
- `analyse_harmony(...)` — public API function
- `_compute_harmony(...)` — orchestration; calls into the submodules

Re-exports for test compatibility (tests import all of these from
`manual_analyser.analysis.harmony`):

```python
from .keys import _detect_key
from .chords import ChordEvent, _match_chord, _chords_to_progression
from .sections import _get_section_boundaries
```

The test file `tests/analysis/test_harmony.py` imports:
```python
from manual_analyser.analysis.harmony import (
    ChordEvent,
    HarmonyResult,
    SectionHarmony,
    _chords_to_progression,
    _detect_key,
    _get_section_boundaries,
    _match_chord,
    analyse_harmony,
)
```

All of these must be importable from `manual_analyser.analysis.harmony` after
the split. The `__init__.py` re-exports handle this — no changes to the test
file.

### Import change in `harmony/__init__.py`

The `_compute_harmony` function currently calls `normalise_lyric_density`
indirectly via structure. It does **not** import from `utils` directly —
no normalise import change needed here.

---

## 2. `analysis/structure/`

### Why split

`structure.py` runs two completely separate pipeline passes that happen to
live in the same file:

- **Pass 1** (`segment_track`) — pure librosa segmentation, writes section
  skeleton to DB
- **Pass 2** (`align_sections`) — reads multiple DB tables, runs labelling
  heuristics, writes labels back

These passes run at different stages of the pipeline (Stage 3a and Stage 4
respectively). They share no runtime state.

### New file layout

```
src/manual_analyser/analysis/structure/
├── __init__.py       ← re-exports; constants
├── boundaries.py     ← pass 1: segmentation
└── alignment.py      ← pass 2: label assignment + helpers
```

### `structure/boundaries.py`

**Move here:**
- `DEFAULT_N_SEGMENTS`, `MIN_SEGMENT_DURATION`, `MIN_SEGMENTS`,
  `MAX_SEGMENTS`
- `segment_track(track_id, full_wav, data_dir, db_path)` — public API for
  pass 1
- `_segment_audio(y, sr)` — librosa agglomerative call
- `_write_sections(conn, track_id, boundaries, short_id)` — skeleton INSERT

**Imports needed:** `librosa`, `numpy`, `pathlib.Path`, `get_connection`

---

### `structure/alignment.py`

**Move here:**
- `HIGH_CONFIDENCE`, `MEDIUM_CONFIDENCE`, `LOW_CONFIDENCE`
- `SectionLabel` dataclass — imported by tests
- `align_sections(track_id, data_dir, db_path)` — public API for pass 2
- `_run_alignment(db_path, track_id, short_id)`
- `_compute_section_energies(sections, rms_array, duration)` — imported by
  tests
- `_compute_lyric_features(sections, transcript_rows)` — imported by tests
- `_assign_labels(sections, energies, lyric_data, duration, short_id)` —
  imported by tests
- `_find_repeated_phrase(words)` — imported by tests
- `_extract_phrases(words, n)`
- `_write_labels(db_path, track_id, labels, short_id)`

**Imports needed:** `json`, `numpy`, `Counter` from `collections`,
`normalise_lyric_density` from `analysis.normalise`, `get_connection`

---

### `structure/__init__.py`

Re-exports for test compatibility. The test file
`tests/analysis/test_structure.py` imports:

```python
from manual_analyser.analysis.structure import (
    SectionLabel,
    _assign_labels,
    _compute_lyric_features,
    _compute_section_energies,
    _find_repeated_phrase,
    align_sections,
    segment_track,
)
```

All of these must be importable from `manual_analyser.analysis.structure`.
Wire them up in `__init__.py`:

```python
from .boundaries import segment_track
from .alignment import (
    SectionLabel,
    align_sections,
    _assign_labels,
    _compute_lyric_features,
    _compute_section_energies,
    _find_repeated_phrase,
)
```

No test file changes needed.

### Import change in `structure/alignment.py`

```python
# REMOVE (currently imports from utils):
from manual_analyser.utils import normalise_lyric_density

# REPLACE WITH:
from manual_analyser.analysis.normalise import normalise_lyric_density
```

---

## 3. `transcription/` — two files, no subpackage

`whisper.py` is one logical module (Whisper transcription), but it currently
mixes the transcription engine with the hook extraction and lyric statistics
helpers. The split is into two sibling files rather than a subpackage — no
directory needed, no `__init__.py` required.

### New file layout

```
src/manual_analyser/transcription/
├── whisper.py    ← engine only (unchanged public API)
└── hooks.py      ← _extract_hook, _compute_unique_word_ratio
```

### `transcription/hooks.py`

**Move here:**
- `_extract_hook(segments, ngram_size)` — n-gram hook detection
- `_compute_unique_word_ratio(text)` — lyric repetition metric

**Imports needed:** `Counter` from `collections`, `numpy`,
`TranscriptSegment` (imported from `transcription.whisper` — or define a
minimal protocol/type alias here to avoid circular import; see note below)

> **Circular import note**: `hooks.py` needs `TranscriptSegment` to type its
> argument, but `whisper.py` will import from `hooks.py`. The simplest fix: put
> `TranscriptSegment` in a third file `transcription/types.py` and import it
> from there in both `whisper.py` and `hooks.py`. Alternatively, use a
> `TYPE_CHECKING` guard or just use `list` without the type annotation in
> `hooks.py`'s signature — the runtime behaviour is identical. The type alias
> approach is cleanest.

### `transcription/whisper.py`

**Remove from here:**
- `_extract_hook` and `_compute_unique_word_ratio` (now in `hooks.py`)

**Add at top:**
```python
from manual_analyser.transcription.hooks import (
    _extract_hook,
    _compute_unique_word_ratio,
)
```

**Also change the device import:**
```python
# REMOVE:
from manual_analyser.utils import get_torch_device

# REPLACE WITH:
from manual_analyser.audio.device import get_torch_device
```

The test file `tests/transcription/test_whisper.py` imports everything from
`manual_analyser.transcription.whisper` (not from `hooks`). Since `whisper.py`
re-imports `_extract_hook` and `_compute_unique_word_ratio` from `hooks.py`,
those names remain available at `manual_analyser.transcription.whisper._extract_hook`
etc. — no test changes needed.

---

## Deletion checklist

Once all splits are applied and tests pass:

- [ ] Delete `src/manual_analyser/utils.py`
- [ ] Delete `tests/test_utils.py`
- [ ] Delete `src/manual_analyser/analysis/harmony.py`
- [ ] Delete `src/manual_analyser/analysis/structure.py`

`whisper.py` is not deleted — it is edited in place.

---

## Test changes required

| Test file | Change needed |
|---|---|
| `tests/analysis/test_harmony.py` | None — `__init__.py` re-exports cover all imports |
| `tests/analysis/test_structure.py` | None — `__init__.py` re-exports cover all imports |
| `tests/transcription/test_whisper.py` | None — `whisper.py` re-imports from `hooks.py` |
| `tests/test_utils.py` | Delete — replaced by `test_device.py`, `test_normalise.py`, `test_decode_ingest.py` (already produced) |

---

## Summary of all files produced in this session vs remaining

### Done
- `audio/device.py` ✅
- `audio/decode.py` (absorbs ingest helpers) ✅
- `audio/separate.py` (import fix) ✅
- `analysis/normalise.py` ✅
- `analysis/groove/__init__.py` ✅
- `analysis/groove/danceability.py` ✅
- `analysis/groove/regularity.py` ✅
- `analysis/rhythm.py` (absorbs pattern/groove helpers) ✅
- `tests/audio/test_device.py` ✅
- `tests/audio/test_decode_ingest.py` ✅
- `tests/analysis/test_normalise.py` ✅

### Remaining (this guide)
- `analysis/harmony/__init__.py`
- `analysis/harmony/keys.py`
- `analysis/harmony/chords.py`
- `analysis/harmony/sections.py`
- `analysis/structure/__init__.py`
- `analysis/structure/boundaries.py`
- `analysis/structure/alignment.py`
- `transcription/hooks.py`
- `transcription/whisper.py` (edited — import changes only)

### One-line import fixes still needed (see `IMPORT_CHANGES.md`)
- `analysis/energy.py` — `utils` → `analysis.normalise`
- `analysis/structure/alignment.py` — `utils` → `analysis.normalise`
- `transcription/whisper.py` — `utils` → `audio.device`