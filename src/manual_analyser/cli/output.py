"""cli/output.py — Rich terminal output helpers."""

from rich.console import Console

from manual_analyser.pipeline.types import RunSummary

console = Console()


def print_run_summary(summary: RunSummary) -> None:
    """Print the end-of-run summary table to the terminal."""
    console.print()
    console.rule("[bold]Analysis complete[/bold]")
    console.print(f"  [green]Complete:[/green]  {len(summary.complete)} tracks")
    console.print(f"  [yellow]Partial:[/yellow]   {len(summary.partial)} tracks")
    console.print(f"  [red]Skipped:[/red]   {len(summary.skipped)} tracks")
    console.rule()

    if summary.partial:
        console.print("\n[yellow]Partial tracks:[/yellow]")
        for state in summary.partial:
            label = state.mp3_path.split("/")[-1]
            issues = ", ".join(state.failed_stages) or "unknown"
            console.print(f"  {label}  {issues}")

    if summary.skipped:
        console.print("\n[red]Skipped tracks:[/red]")
        for state in summary.skipped:
            label = state.mp3_path.split("/")[-1]
            note = state.notes[0] if state.notes else "unknown"
            console.print(f"  {label}  {note}")


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_info(message: str) -> None:
    console.print(f"[cyan]{message}[/cyan]")


def print_warning(message: str) -> None:
    console.print(f"[yellow]Warning:[/yellow] {message}")
