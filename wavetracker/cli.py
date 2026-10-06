"""Command line interface: ``wavetracker --help``."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import __version__
from .config import Config

app = typer.Typer(
    name="wavetracker",
    help="Detect and track wave-type electric fish in multi-electrode recordings.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
console = Console(stderr=True)
log = logging.getLogger("wavetracker")


def _setup_logging(verbose: int) -> None:
    level = (
        logging.WARNING
        if verbose == 0
        else logging.INFO
        if verbose == 1
        else logging.DEBUG
    )
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=verbose > 1)],
        force=True,
    )
    logging.getLogger("wavetracker").setLevel(level)


def _version(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show version."
        ),
    ] = None,
) -> None:
    pass


Verbose = Annotated[
    int, typer.Option("--verbose", "-v", count=True, help="-v info, -vv debug.")
]
ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c", exists=True, dir_okay=False, help="YAML config file."
    ),
]


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[speed]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _summary_table(results, title: str, min_detections: int = 1) -> Table:
    table = Table(title=title, title_justify="left")
    for col in (
        "id",
        "detections",
        "median f [Hz]",
        "f range [Hz]",
        "start [s]",
        "end [s]",
    ):
        table.add_column(col, justify="right")
    t = results.times[results.idx_v]
    for i in results.ids():
        m = results.ident_v == i
        if m.sum() < min_detections:
            continue
        f = results.fund_v[m]
        table.add_row(
            f"{int(i)}",
            f"{m.sum()}",
            f"{np.median(f):.1f}",
            f"{f.min():.1f}-{f.max():.1f}",
            f"{t[m].min():.1f}",
            f"{t[m].max():.1f}",
        )
    n_nan = int(np.isnan(results.ident_v).sum())
    table.caption = f"{len(results.fund_v)} detections, {n_nan} unassigned"
    return table


@app.command()
def run(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            help="Recording file(s) or directories (fishgrid traces-grid*.raw or a "
            "sequence of audio files).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Results directory. With several inputs, one subdirectory per input.",
        ),
    ] = Path("wavetracker_output"),
    config: ConfigOpt = None,
    start: Annotated[
        float, typer.Option(help="Start of the analysed range [s].")
    ] = 0.0,
    duration: Annotated[
        float | None, typer.Option(help="Duration of the analysed range [s].")
    ] = None,
    device: Annotated[
        str, typer.Option(help="Torch device: auto, cuda, cuda:1, cpu, ...")
    ] = "auto",
    track: Annotated[
        bool, typer.Option(help="Track identities after detection.")
    ] = True,
    save_spec: Annotated[
        bool, typer.Option(help="Store the full-resolution spectrogram (large).")
    ] = False,
    interference: Annotated[
        bool | None,
        typer.Option(help="Remove interference combs (default: from config, on)."),
    ] = None,
    exclude: Annotated[
        list[int] | None,
        typer.Option(
            "--exclude-channel", "-x", help="Ignore this channel (0-based, repeatable)."
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", "-f", help="Overwrite existing results.")
    ] = False,
    verbose: Verbose = 0,
) -> None:
    """Detect fish fundamentals in recordings and track their identities."""
    from .io import recording_info
    from .pipeline import detect, track_results

    _setup_logging(verbose)
    cfg = Config.load(config)
    if save_spec:
        cfg.output.save_fine_spec = True
    if exclude:
        cfg.spectrogram.exclude_channels = sorted(set(exclude))
    if interference is not None:
        cfg.interference.enabled = interference

    jobs = [
        (p, output if len(inputs) == 1 else output / p.resolve().name) for p in inputs
    ]
    for path, out in jobs:
        if (out / "fund_v.npy").exists() and not overwrite:
            console.print(f"[yellow]Skipping {path}: results exist in {out} (use -f).")
            continue
        info = recording_info(path)
        span = (
            info.duration - start
            if duration is None
            else min(duration, info.duration - start)
        )
        console.print(
            f"[bold]{path}[/] → {out}  "
            f"[dim]({info.channels} ch, {info.rate:.0f} Hz, {span:.0f} of {info.duration:.0f} s)"
        )
        t0 = time.perf_counter()
        with _progress() as progress:
            task = progress.add_task("detect", total=None, speed="")

            def update(done: int, total: int, t0=t0, span=span, task=task) -> None:
                elapsed = time.perf_counter() - t0
                rt = span * done / total / max(elapsed, 1e-9)
                progress.update(
                    task, completed=done, total=total, speed=f"{rt:.0f}x realtime"
                )

            det = detect(path, out, cfg, start, duration, device, progress=update)
        tm = det.timings
        console.print(
            f"  {len(det.results.fund_v)} detections in {tm.total:.1f}s "
            f"[dim](read {tm.read:.1f}s, spectrogram {tm.spectrogram:.1f}s, "
            f"harmonic groups {tm.detection:.1f}s)"
        )
        if track:
            dt = track_results(det.results, cfg)
            det.results.save(out)
            console.print(f"  tracked {det.results.n_ids} identities in {dt:.1f}s")
            console.print(_summary_table(det.results, str(out), min_detections=10))


@app.command("track")
def track_cmd(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config: ConfigOpt = None,
    verbose: Verbose = 0,
) -> None:
    """(Re-)run identity tracking on existing detections."""
    from .pipeline import track_results
    from .results import Results

    _setup_logging(verbose)
    cfg = Config.load(config)
    results = Results.load(results_dir)
    with console.status("tracking"):
        dt = track_results(results, cfg)
    results.save(results_dir)
    console.print(f"tracked {results.n_ids} identities in {dt:.1f}s")


@app.command()
def summary(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    min_detections: Annotated[
        int, typer.Option(help="Hide identities with fewer detections.")
    ] = 10,
) -> None:
    """Show the identities found in a results directory."""
    from .results import Results

    Console().print(
        _summary_table(Results.load(results_dir), str(results_dir), min_detections)
    )


@app.command()
def info(
    inputs: Annotated[list[Path], typer.Argument(exists=True)],
) -> None:
    """Show channels, sampling rate and duration of recordings."""
    from .io import recording_info

    table = Table()
    for col in ("recording", "channels", "rate [Hz]", "duration"):
        table.add_column(col)
    for p in inputs:
        i = recording_info(p)
        h, rem = divmod(i.duration, 3600)
        table.add_row(
            str(p), str(i.channels), f"{i.rate:.0f}", f"{int(h)}h {rem / 60:.1f}min"
        )
    Console().print(table)


@app.command()
def config(
    output: Annotated[
        Path | None, typer.Argument(help="Write the default config here.")
    ] = None,
) -> None:
    """Print (or write) the default configuration as YAML."""
    cfg = Config()
    if output is None:
        print(cfg.to_yaml(), end="")
    else:
        cfg.save(output)
        console.print(f"wrote {output}")


@app.command()
def plot(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Image file (default: show).")
    ] = None,
    min_detections: Annotated[int, typer.Option(help="Hide shorter identities.")] = 10,
    fmin: Annotated[
        float | None, typer.Option(help="Lower frequency limit [Hz].")
    ] = None,
    fmax: Annotated[
        float | None, typer.Option(help="Upper frequency limit [Hz].")
    ] = None,
) -> None:
    """Plot tracked frequencies on top of the overview spectrogram."""
    from .plotting import plot_results

    plot_results(results_dir, output, min_detections=min_detections, flim=(fmin, fmax))
    if output:
        console.print(f"wrote {output}")


@app.command()
def synth(
    output: Annotated[Path, typer.Argument(help="Output .wav file.")],
    n_fish: Annotated[int, typer.Option("--fish", "-n")] = 3,
    duration: Annotated[float, typer.Option(help="[s]")] = 120.0,
    channels: Annotated[int, typer.Option()] = 8,
    rate: Annotated[float, typer.Option(help="[Hz]")] = 20000.0,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Generate a synthetic recording with known fish (ground truth saved alongside)."""
    from .synthetic import random_fish, save_recording, synthesize

    rng = np.random.default_rng(seed)
    fish = random_fish(n_fish, duration, rng=rng)
    rec = synthesize(fish, duration, rate=rate, channels=channels, rng=rng)
    truth = save_recording(rec, output)
    console.print(f"wrote {output} and {truth}")


# --- post-processing -------------------------------------------------------


@app.command()
def cleanup(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    n_fish: Annotated[int | None, typer.Option("--n-fish", "-n")] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="cleanup .cfg file")
    ] = None,
    stride: Annotated[float | None, typer.Option(help="Window size [min].")] = None,
    overlap: Annotated[float | None, typer.Option(help="Window overlap (0-1).")] = None,
    freq_tol: Annotated[float | None, typer.Option(help="[Hz]")] = None,
    time_tol: Annotated[float | None, typer.Option(help="[min]")] = None,
    density: Annotated[
        float | None, typer.Option(help="Min. detection density (0-1).")
    ] = None,
    show: Annotated[bool, typer.Option(help="Show result figures.")] = False,
) -> None:
    """Join and filter tracks, keeping the N most prominent fish.

    Parameters are read from cleanup_config.cfg in the results directory, or
    the packaged default; options override them.
    """
    from .postprocessing import cleanup as cu

    cu.show_results = show
    cu.main(
        results_dir,
        n_fish=n_fish,
        stride_minutes=stride,
        overlap_frac=overlap,
        freq_tolerance=freq_tol,
        time_tolerance_minutes=time_tol,
        density_threshold=density,
        config_path=config,
    )


@app.command()
def concat(
    day_folders: Annotated[list[Path], typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    gap: Annotated[float, typer.Option(help="Gap inserted between days [s].")] = 0.0,
) -> None:
    """Concatenate results of consecutive recordings into one dataset."""
    from .postprocessing.concat import concatenate_wavetracker_outputs

    concatenate_wavetracker_outputs(
        [str(f) for f in day_folders], str(output), gap_duration=gap, verbose=True
    )


@app.command("freq-analysis")
def freq_analysis(
    folders: Annotated[list[Path], typer.Argument(help="Results folders (globs ok).")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    analyze: Annotated[
        bool | None, typer.Option(help="Run temporal trace analysis.")
    ] = None,
    n_fish: Annotated[int | None, typer.Option("--n-fish", "-n")] = None,
    timepoints: Annotated[
        list[str] | None, typer.Option("--timepoint", "-t", help="HH:MM, repeatable.")
    ] = None,
    window: Annotated[float | None, typer.Option(help="Window [s].")] = None,
    method: Annotated[str | None, typer.Option(help="occurrence or power")] = None,
    freq_tolerance: Annotated[float | None, typer.Option(help="[Hz]")] = None,
    trace_tol: Annotated[float | None, typer.Option(help="[Hz]")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Extract the N most prominent frequencies at fixed times of day."""
    from types import SimpleNamespace

    from .postprocessing.frequency_analysis import run as fa_run

    fa_run(
        SimpleNamespace(
            folders=folders,
            config=config,
            run_analysis=analyze,
            n_fish=n_fish,
            timepoints=timepoints,
            window=window,
            method=method,
            freq_tolerance=freq_tolerance,
            trace_tol=trace_tol,
            output=output,
        )
    )


@app.command()
def sorter(
    results_dir: Annotated[Path | None, typer.Argument(file_okay=False)] = None,
) -> None:
    """Open the EOD sorter GUI for manual track correction (needs the \\[gui] extra)."""
    import sys

    try:
        from .gui import eodsorter
    except ImportError as e:  # pragma: no cover
        console.print(f"[red]{e}. Install with: uv sync --extra gui")
        raise typer.Exit(1) from e
    sys.argv = [sys.argv[0]] + ([str(results_dir)] if results_dir else [])
    eodsorter.main()


if __name__ == "__main__":
    app()
