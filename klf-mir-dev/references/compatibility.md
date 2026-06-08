# Compatibility Reference

Known issues, tested states, and fallback strategies for MIR libraries
on Python 3.11+ / Fedora 44 and macOS (Apple Silicon M3).

---

## madmom

**Status**: ❌ CONFIRMED FAILED — Python 3.11, macOS aarch64, tested 2025-06-08

**Failure**:
```
ImportError: cannot import name 'MutableSequence' from 'collections'
```
`collections.MutableSequence` was removed in Python 3.10. Hard failure —
do not use madmom on Python 3.10+.

**Resolution**: librosa is the primary implementation for all rhythm and
beat detection. No conditional import — librosa only.

**Fallback strategy** — `analysis/rhythm.py` (librosa primary):

```python
# BPM / beat tracking
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Drum pattern detection per frequency band (on drums stem)
onset_env_kick  = librosa.onset.onset_strength(y=y, sr=sr, fmax=200)
onset_env_snare = librosa.onset.onset_strength(y=y, sr=sr, fmin=200, fmax=2000)
onset_env_hihat = librosa.onset.onset_strength(y=y, sr=sr, fmin=4000)
# Quantise to 16-step grid — see analysis/rhythm.py for full implementation

# Groove feel — swing ratio from 8th-note subdivision timing
_, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
beat_times = librosa.frames_to_time(beats, sr=sr)
# ratio > 0.55 → swung; < 0.52 → straight; else unclear
```

---

## msaf

**Status**: ❌ CONFIRMED FAILED — Python 3.11, macOS aarch64, tested 2025-06-08

**Failure**:
```
ImportError: cannot import name 'inf' from 'scipy'
```
`scipy.inf` was removed in SciPy 1.11 (deprecated since 1.0; it was always
just `numpy.inf`). msaf's pymf dependency was never updated. Hard failure —
do not use msaf on SciPy 1.11+.

**Resolution**: `librosa.segment.agglomerative` is the primary implementation
for structural segmentation. No conditional import — librosa only.

**Primary strategy** — `analysis/structure.py` (librosa primary):

```python
def segment_track(y: np.ndarray, sr: int, n_segments: int = 8) -> np.ndarray:
    """Return segment boundary times in seconds using agglomerative clustering."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.vstack([chroma, mfcc])
    bounds = librosa.segment.agglomerative(features, k=n_segments)
    return librosa.frames_to_time(bounds, sr=sr)
```

Note: lower boundary accuracy than msaf, but reliable and actively maintained.
The hybrid lyric-alignment pass compensates significantly for boundary imprecision.

---

## essentia

**Status**: ✅ Installed cleanly (uv sync, 2025-06-08)

**Version**: 2.1b6.dev1389

**Test**:
```bash
uv add essentia
uv run python -c "import essentia.standard as es; es.Danceability(); print('essentia OK')"
```

**Fallback strategy** (if fails):
```python
def approximate_danceability(
    beat_regularity: float,
    tempo_stability: float,
    mean_energy: float
) -> float:
    return float(np.clip(
        (beat_regularity * 0.5) + (tempo_stability * 0.3) + (mean_energy * 0.2),
        0.0, 1.0
    ))
```

---

## openai-whisper

**Status**: ✅ Installed cleanly (uv sync, 2025-06-08)

**Device notes**:
- Fedora / CUDA (RTX 5070 Ti): uses CUDA
- macOS Apple Silicon (M3): uses MPS
- Use shared `get_torch_device()` — see Device detection below.

Model: large-v3 on GPU; medium on CPU.

---

## demucs

**Status**: ✅ Installed cleanly (uv sync, 2025-06-08)

**Device notes**: same as Whisper — use `get_torch_device()`.
Model: htdemucs.

Approximate runtime:
- RTX 5070 Ti (CUDA): ~10–30s per 3-minute track
- M3 (MPS): ~60–120s per 3-minute track

---

## librosa

**Status**: ✅ Installed cleanly (uv sync, 2025-06-08)

Primary implementation for ALL MIR analysis. Both madmom and msaf are
confirmed dead on Python 3.11+ — librosa is not a fallback, it is the stack.

---

## Device detection utility

Lives in `src/manual_analyser/utils.py`. Must be created before
`audio/separate.py` or `transcription/whisper.py`.

```python
import torch

def get_torch_device() -> str:
    """Return the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

---

## Summary

| Library | Status | Role |
|---|---|---|
| librosa | ✅ OK | Primary for ALL MIR analysis |
| demucs | ✅ OK | Stem separation |
| openai-whisper | ✅ OK | Transcription |
| madmom | ❌ Failed — `collections.MutableSequence` removed in Python 3.10 | Dead |
| msaf | ❌ Failed — `scipy.inf` removed in SciPy 1.11 | Dead |
| essentia | ✅ OK — 2.1b6.dev1389 | Danceability, groove descriptors |

## Impact on architecture

Both high-risk libraries failed as anticipated. This simplifies the codebase:

- `analysis/rhythm.py` — librosa only, no conditional import
- `analysis/structure.py` — librosa only, no conditional import
- No `try/except ImportError` branching anywhere in the analysis modules
- The skill compatibility reference and DESIGN.md fallback notes remain
  for historical context but the "fallback" is now simply the implementation

---

## Full stack verification

**Environment**: macOS Apple Silicon M3, Python 3.11.14, 2025-06-08

```
librosa       0.11.0
numpy         1.26.4
scipy         1.17.1
sklearn       1.9.0
essentia      2.1-beta6-dev
demucs        4.0.1
whisper       20250625
torch         2.12.0
httpx         0.28.1
qdrant_client OK
typer         0.26.7
jinja2        3.1.6
device:       mps (Apple Silicon)
```

All imports clean. Ready to write module code.