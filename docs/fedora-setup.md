# KLF Manual Analyser — Full Setup Guide (Fedora / RTX 5070 Ti)

## 1. System dependencies

```bash
# ffmpeg (required — MP3 decode & normalise)
sudo dnf install -y ffmpeg

# Docker (for Qdrant)
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and back in for the group to take effect
```

Verify ffmpeg is on your PATH:
```bash
ffmpeg -version
```

---

## 2. Python 3.11 + uv

The project requires Python 3.11 (pinned in `.python-version`).

```bash
# Install uv (Astral's fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # or restart your shell

# uv will pick up .python-version automatically, but install 3.11 if needed
uv python install 3.11
```

---

## 3. Install the project

```bash
cd klf-manual-analyser
uv sync
```

This installs everything from the lockfile — librosa, essentia, Demucs, openai-whisper (large-v3), torch with CUDA, and all supporting libraries. It will take a while the first time; torch alone is ~500 MB.

Verify the imports all work:
```bash
make verify
```

---

## 4. Ollama + models

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama

# Pull the required models
ollama pull qwen2.5:14b        # primary LLM scorer (~9 GB)
ollama pull nomic-embed-text   # embeddings for Qdrant (~300 MB)

# Optional: pull the fallback model
ollama pull mistral-nemo:12b
```

Check Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

---

## 5. Qdrant (optional but recommended)

Qdrant is optional — the pipeline will skip it gracefully if it's not running, you just lose the similarity/nearest-neighbour features in the report. Podman is already set up with qdrant, it may need booting:

```bash
podman start qdrant
```

Verify it's up:
```bash
curl http://localhost:6333/healthz; echo ""
# Should return: healthz check passed
```

---

## 6. Whisper

Whisper is installed as part of `uv sync` (it's in the lockfile as `openai-whisper`). It uses `large-v3` by default and will **download the model weights on first use** (~1.5 GB), cached to `~/.cache/whisper/`. No manual pull needed — it happens automatically the first time a track with vocals is processed.

If you want to pre-fetch it now:
```bash
uv run python -c "import whisper; whisper.load_model('large-v3')"
```

---

## 7. Getting MP3s

MP3s must follow the naming convention:

```
Artist_Name-Song_Title.mp3
```

Underscores within a name, hyphen between artist and title. Examples:
```
The_KLF-Doctorin_The_Tardis.mp3
Kylie_Minogue-Hand_On_Your_Heart.mp3
```

**For your first run / testing**, grab a few public domain tracks from the Internet Archive's 78rpm collection — these are the same ones intended for `1920s_1930s` mode:

```
https://archive.org/details/78rpm
```

Filter by pre-1928 recordings. Download MP3 format, rename to convention, drop into a folder (e.g. `~/music/test-tracks/`).

**For real 1988 mode runs**, put your own MP3s in a folder with the naming convention applied. The pipeline is designed around ~40 tracks.

---

## 8. Run the tests

```bash
make check
```

All tests run against in-memory SQLite with mocked audio, so no real MP3s are needed here.

---

### 9. First analysis run

Start with a small batch in `1920s_1930s` mode (public domain tracks, more forgiving of audio quality):

```bash
make analyse MODE=1920s_1930s PATH=~/music/test-tracks
```

Or for 1988 mode with your own tracks:
```bash
make analyse MODE=1988 PATH=~/music/klf-era
```

Then render and serve the report:
```bash
make report MODE=1988
make serve
# Browse to http://localhost:8000
```

---

## Quick-reference: what must be running before `make analyse`

| Service | How to check |
|---|---|
| ffmpeg | `which ffmpeg` |
| Ollama | `curl localhost:11434/api/tags` |
| Qdrant | `curl localhost:6333/healthz` (optional) |
| GPU/CUDA | `uv run python -c "import torch; print(torch.cuda.is_available())"` |

The only hard abort if missing is **Ollama** (scoring stage). Everything else degrades gracefully.

---

## Likely gotcha: torch + CUDA on Fedora

The lockfile has CUDA-enabled torch wheels. If `torch.cuda.is_available()` returns `False`, check your CUDA toolkit version matches what torch expects:

```bash
nvidia-smi           # shows driver + CUDA version
nvcc --version       # shows toolkit version
```

If there's a mismatch, the torch wheel from the lockfile may have been built against a different CUDA version. Let me know what `nvidia-smi` reports and I'll help you sort it.