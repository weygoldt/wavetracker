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
        disable=not console.is_terminal,  # no redraw spam in logs/pipes
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
    if cfg.default_frequency_range():
        hc = cfg.harmonic_groups
        console.print(
            f"[yellow]Note:[/] tracking the broad default fish range "
            f"{hc.min_freq:.0f}-{hc.max_freq:.0f} Hz. If you know the frequencies "
            "of your fish, set harmonic_groups.min_freq/max_freq to a narrow band "
            "(-c config.yaml): fewer false fish, better hum removal, faster."
        )
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
def harmonics(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    timescale: Annotated[
        float, typer.Option(help="Only modulations faster than this are compared [s].")
    ] = 30.0,
    min_overlap: Annotated[
        float, typer.Option(help="Minimum common time of two identities [s].")
    ] = 10.0,
    min_pattern: Annotated[
        float | None,
        typer.Option(help="Also accept electrode-pattern similarity >= this (grids)."),
    ] = None,
    remove: Annotated[
        bool, typer.Option(help="Untrack the harmonic identities (ident_v = NaN).")
    ] = False,
) -> None:
    """Find identities that are harmonics of other identities (co-modulation).

    Writes harmonics.csv to the results directory.
    """
    import numpy as np

    from .comodulation import ComodulationConfig, find_harmonics
    from .results import Results

    res = Results.load(results_dir)
    cfg = ComodulationConfig(
        timescale=timescale, min_overlap=min_overlap, min_pattern=min_pattern
    )
    with Console().status("scoring identity pairs"):
        found = find_harmonics(res, cfg)
    found.to_csv(results_dir / "harmonics.csv", index=False)
    table = Table(title=f"{len(found)} harmonic identities")
    for col in ("high", "f_high", "low", "f_low", "h", "overlap", "offset", "evidence"):
        table.add_column(col)
    for _, r in found.iterrows():
        table.add_row(
            f"{r.high:.0f}",
            f"{r.f_high:.1f}",
            f"{r.low:.0f}",
            f"{r.f_low:.1f}",
            str(r.h),
            f"{r.overlap:.0f}",
            f"{r.offset:+.2f}",
            r.evidence,
        )
    Console().print(table)
    if remove and len(found):
        res.ident_v[np.isin(res.ident_v, found.high.to_numpy())] = np.nan
        res.save(results_dir)
        Console().print(f"untracked {len(found)} identities")


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


@app.command("merge-by-position")
def merge_by_position_cmd(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    electrodes: Annotated[
        Path,
        typer.Option(
            "--electrodes",
            "-e",
            exists=True,
            dir_okay=False,
            help="Electrode positions over time (.npz: time, positions (T, E, 3); "
            ".csv: time, x0, y0, z0, x1, ...) in s and m, z <= 0 under water.",
        ),
    ],
    config: ConfigOpt = None,
    reference: Annotated[
        int | None,
        typer.Option(
            help="Reference electrode: channel i = electrode i - reference "
            "(default: from config)."
        ),
    ] = None,
    recording: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            help="Recording for the noise floor (default: the input in "
            "wavetracker.json, if it exists).",
        ),
    ] = None,
    noise_floor: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Noise floor .npz (freqs, power (F, channels)) instead of the recording.",
        ),
    ] = None,
    t_start: Annotated[
        float | None, typer.Option(help="Survey start \\[s] (default: config).")
    ] = None,
    t_end: Annotated[
        float | None, typer.Option(help="Survey end \\[s] (default: config).")
    ] = None,
    jobs: Annotated[
        int | None,
        typer.Option("--jobs", "-j", help="Parallel processes (default: all CPUs)."),
    ] = None,
    verbose: Verbose = 0,
) -> None:
    """Group track segments into fish by frequency and position (moving electrodes).

    For stationary fish recorded with moving electrodes: identities are split
    into segments, grouped by frequency, localised from their amplitudes and
    relative signs, split where the position does not fit, and merged where
    frequency and position agree. Writes fish_v.npy (fish of each detection),
    fish.csv, fish_segments.csv and position_merging.json; ident_v is kept.
    """
    from .position.electrodes import ElectrodeTrack
    from .position.merging import merge_by_position, survey_mask
    from .position.noise import DetectionNoiseFloor, SpectrumNoiseFloor
    from .results import Results

    _setup_logging(verbose)
    cfg = Config.load(config).position_merging
    if reference is not None:
        cfg.reference = reference
    if t_start is not None:
        cfg.t_start = t_start
    if t_end is not None:
        cfg.t_end = t_end
    results = Results.load(results_dir)
    track = ElectrodeTrack.load(electrodes, cfg.reference, cfg.channel_pairs)

    if noise_floor is not None:
        floor = SpectrumNoiseFloor.load(noise_floor)
    else:
        rec = recording
        if rec is None and isinstance(results.meta.get("input"), str):
            rec = Path(results.meta["input"])
        if rec is not None and rec.exists():
            survey = survey_mask(results.times, track, cfg)
            nfft = results.meta.get("config", {}).get("spectrogram", {}).get("nfft")
            if nfft is None:
                raise typer.BadParameter("nfft unknown: no config in wavetracker.json")
            with console.status(f"noise floor from {rec}"):
                floor = SpectrumNoiseFloor.from_recording(
                    rec,
                    results.times[survey],
                    int(nfft),
                    channels=results.meta.get("channels"),
                    quantile=cfg.noise_quantile,
                    max_freq=float(results.fund_v.max()) + 10.0,
                )
            floor.save(results_dir / "noise_floor.npz")
        else:
            console.print(
                "[yellow]No recording found: noise floor estimated from the "
                "detections (pass --recording or --noise-floor)."
            )
            floor = DetectionNoiseFloor(
                results.fund_v, results.sign_v, cfg.noise_quantile
            )

    t0 = time.perf_counter()
    with _progress() as progress:
        tasks: dict[str, int] = {}

        def update(stage: str, done: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = progress.add_task(stage, total=total, speed="")
            progress.update(tasks[stage], completed=done, total=total)

        out = merge_by_position(
            results, track, cfg, noise_floor=floor, n_jobs=jobs, progress=update
        )
    out.save(results_dir)
    st = out.stats
    console.print(
        f"{st['segments']} segments ({st['clutter_segments']} clutter) → "
        f"{st['initial_candidates']} frequency candidates, "
        f"{st['segments_split_off']} segments split off, {st['merges']} merges → "
        f"[bold]{st['fish']} fish[/] ({st['ambiguous']} ambiguous) in "
        f"{time.perf_counter() - t0:.0f}s"
    )
    table = Table(title=str(results_dir), title_justify="left")
    for col in (
        "fish",
        "f [Hz]",
        "x [m]",
        "y [m]",
        "depth [m]",
        "SE [m]",
        "ambiguous",
        "r",
        "segments",
        "detections",
    ):
        table.add_column(col, justify="right")
    for row in out.fish.itertuples():
        table.add_row(
            f"{row.fish}",
            f"{row.freq:.1f}",
            f"{row.x:.2f}",
            f"{row.y:.2f}",
            f"{row.depth:.2f}",
            f"{row.se_major:.2f}",
            "yes" if row.ambiguous else "",
            f"{row.r_fit:.2f}",
            f"{row.n_segments}",
            f"{row.n_det}",
        )
    Console().print(table)


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
    min_freq: Annotated[
        float | None,
        typer.Option(help="Ignore frequencies below [Hz] (default: none)."),
    ] = None,
    max_freq: Annotated[
        float | None,
        typer.Option(help="Ignore frequencies above [Hz] (default: none)."),
    ] = None,
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
            min_freq=min_freq,
            max_freq=max_freq,
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
