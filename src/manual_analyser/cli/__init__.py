"""
cli/__init__.py — Typer entrypoint for manual-analyser.

Commands:
    analyse  <path> --mode  Run the full analysis pipeline
    report   --mode         Re-render the HTML report from the DB
    serve    --port         Start the local report server
    clean    [--stems] [--features] [--reports]  Remove cached data
"""

from pathlib import Path
from typing import Annotated

import typer

from manual_analyser.audio.decode import DecodeAbortError, check_ffmpeg
from manual_analyser.audio.separate import SeparateAbortError
from manual_analyser.cli import logging_setup
from manual_analyser.cli.clean import clean as run_clean
from manual_analyser.cli.output import console, print_error, print_info, print_run_summary
from manual_analyser.pipeline import run_pipeline
from manual_analyser.scoring.llm import OllamaUnavailableError, check_ollama

app = typer.Typer(
    name="manual-analyser",
    help="Score MP3s against The KLF's Manual hit-song criteria.",
    no_args_is_help=True,
)

_DEFAULT_DATA = Path("data")
_DEFAULT_MODE = "1988"
_DEFAULT_PORT = 8000
_VALID_MODES = ("1988", "contemporary", "1920s_1930s")


@app.command()
def analyse(
    path: Annotated[Path, typer.Argument(help="Folder of MP3s to analyse")],
    mode: Annotated[str, typer.Option(help="Scoring mode")] = _DEFAULT_MODE,
    model: Annotated[str, typer.Option(help="Ollama model")] = "qwen2.5:14b",
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Re-run all stages")] = False,
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = _DEFAULT_DATA,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Analyse a folder of MP3s and score against The Manual's criteria."""
    _validate_mode(mode)
    _validate_path(path)

    logging_setup.configure(data_dir, verbose)
    db_path = data_dir / "manual_analyser.db"

    mp3_paths = sorted(path.glob("*.mp3"))
    if not mp3_paths:
        print_error(f"No MP3 files found in {path}")
        raise typer.Exit(1)

    print_info(f"Found {len(mp3_paths)} MP3s — mode: {mode}")

    _check_hard_dependencies(model)

    try:
        summary = run_pipeline(
            mp3_paths=mp3_paths,
            mode=mode,
            db_path=db_path,
            data_dir=data_dir,
            no_cache=no_cache,
        )
    except (DecodeAbortError, SeparateAbortError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    print_run_summary(summary)

    if summary.total == 0 or len(summary.skipped) == summary.total:
        print_error("No tracks were successfully processed.")
        raise typer.Exit(1)

    print_info(f"Run 'manual-analyser report --mode {mode}' to render the HTML report.")


@app.command()
def report(
    mode: Annotated[str, typer.Option(help="Scoring mode")] = _DEFAULT_MODE,
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = _DEFAULT_DATA,
) -> None:
    """Re-render the HTML report from the database."""
    _validate_mode(mode)
    console.print("[yellow]report command: report renderer not yet implemented.[/yellow]")
    console.print("Run the pipeline first, then check back once report/render.py is complete.")


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port to serve on")] = _DEFAULT_PORT,
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = _DEFAULT_DATA,
) -> None:
    """Start the local report server."""
    console.print("[yellow]serve command: report server not yet implemented.[/yellow]")
    console.print("Run the pipeline first, then check back once report/server.py is complete.")


@app.command()
def clean(
    stems: Annotated[bool, typer.Option("--stems", help="Remove WAV stems")] = False,
    features: Annotated[bool, typer.Option("--features", help="Remove DB records")] = False,
    reports: Annotated[bool, typer.Option("--reports", help="Remove HTML reports")] = False,
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = _DEFAULT_DATA,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Remove cached stems, features, and/or reports."""
    remove_all = not any([stems, features, reports])
    target = (
        "everything"
        if remove_all
        else ", ".join(t for t, f in [("stems", stems), ("features", features), ("reports", reports)] if f)
    )

    if not yes:
        typer.confirm(f"Remove {target} from {data_dir}?", abort=True)

    run_clean(data_dir=data_dir, stems=stems, features=features, reports=reports)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        print_error(f"Unknown mode '{mode}'. Valid modes: {', '.join(_VALID_MODES)}")
        raise typer.Exit(1)


def _validate_path(path: Path) -> None:
    if not path.exists():
        print_error(f"Path does not exist: {path}")
        raise typer.Exit(1)
    if not path.is_dir():
        print_error(f"Path is not a directory: {path}")
        raise typer.Exit(1)


def _check_hard_dependencies(model: str) -> None:
    """Check ffmpeg and Ollama are available before starting."""
    try:
        check_ffmpeg()
    except DecodeAbortError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    try:
        check_ollama(model)
    except OllamaUnavailableError as exc:
        print_error(str(exc))
        raise typer.Exit(1)
