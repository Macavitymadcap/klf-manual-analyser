"""
audio/device.py — Torch device detection for GPU-accelerated stages.

Used by audio/separate.py (Demucs) and transcription/whisper.py.
Kept in the audio package because both consumers are audio-processing
stages that need GPU access; it is not a general-purpose utility.

Import:
    from manual_analyser.audio.device import get_torch_device
"""


def get_torch_device() -> str:
    """
    Return the best available torch device: cuda > mps > cpu.

    Checks in priority order:
    - CUDA (Fedora / NVIDIA RTX — primary production environment)
    - MPS (macOS Apple Silicon — development environment)
    - CPU (fallback)

    torch import is deferred to avoid requiring it at module load time
    in contexts where it is not needed (criteria loading, reporting, etc.).
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
