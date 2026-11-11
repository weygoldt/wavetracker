"""Evaluate a learned-detector model on ldev / ldev-field at several
detection thresholds, with the fixed tracker and linker of
``configs/learned-relinked.yaml`` (only ``detector.model`` and
``detector.threshold`` change), and print the comparison table.

    python benchmarks/dense/det_sweep.py run --model $WT_BENCHMARK_DATA/salience/B-det-x/model.pt \
        --tag B-det-x --thresholds 0.5 0.6 0.7
    python benchmarks/dense/det_sweep.py table --tags 'B-det-v1-s*' 'B-det-x-*'

Run tags are ``<tag>-t<threshold>``; each (threshold, split) runs as its own
process (``--jobs`` at a time).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA
from strata import frames, scores, strata

RUNS = DATA / "runs"
HERE = Path(__file__).parent
BASE = HERE / "configs" / "learned-relinked.yaml"
SPLITS = ("ldev", "ldev-field")


def run(args):
    base = yaml.safe_load(BASE.read_text())
    cfg_dir = DATA / "salience" / "sweep-configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for thr in args.thresholds:
        tag = f"{args.tag}-t{thr:g}"
        cfg = dict(base)
        cfg["detector"] = dict(base["detector"]) | {
            "model": str(Path(args.model).expanduser().resolve()),
            "threshold": float(thr),
        }
        path = cfg_dir / f"{tag}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        for sp in args.splits:
            jobs.append((tag, path, sp))

    def one(job):
        tag, path, sp = job
        log = RUNS / tag / f"{sp}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(HERE / "run.py"),
            "track",
            "--tag",
            tag,
            "-c",
            str(path),
            "--split",
            sp,
        ]
        with open(log, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return tag, sp, r.returncode

    with ThreadPoolExecutor(args.jobs) as ex:
        for tag, sp, rc in ex.map(one, jobs):
            print(f"{tag} {sp}: {'ok' if rc == 0 else f'failed ({rc})'}", flush=True)
    table(argparse.Namespace(tags=[f"{args.tag}-t*"], json=None))


COLS = [
    ("idf1", "IDF1"),
    ("snr 0-6 dB", "rec 0-6dB"),
    ("nn < 1.2 Hz", "rec nn<1.2"),
    ("nn 1.2-3 Hz", "rec nn1.2-3"),
    ("recall", "recall"),
    ("precision", "precision"),
    ("harmonic_per_frame", "harm/frame"),
    ("false_per_frame", "false/frame"),
    ("false_ids_30s", "false ids>=30s"),
]


def row(tag: str) -> dict | None:
    out = {}
    for sp in SPLITS:
        fr = frames(tag, sp)
        if not fr:
            return None
        sc = scores(tag, sp)
        if sc["n"] < 17:
            return None
        out[sp] = sc | strata(fr)
    return out


def table(args):
    tags = sorted(
        {
            p.name
            for pat in args.tags
            for p in RUNS.iterdir()
            if p.is_dir() and fnmatch.fnmatch(p.name, pat)
        }
    )
    rows = {t: r for t in tags if (r := row(t)) is not None}
    w = max([len(t) for t in rows] + [10])
    head = f"{'':{w}s} " + " ".join(f"{c[1]:>15s}" for c in COLS)
    print("lab / field; recall by nearest line only for frames >= 6 dB")
    print(head)
    for t, r in rows.items():
        cells = []
        for k, _ in COLS:
            a, b = r[SPLITS[0]][k], r[SPLITS[1]][k]
            fmt = "{:.1f}/{:.1f}" if k == "false_ids_30s" else "{:.3f}/{:.3f}"
            cells.append(f"{fmt.format(a, b):>15s}")
        mean = 0.5 * (r[SPLITS[0]]["idf1"] + r[SPLITS[1]]["idf1"])
        print(f"{t:{w}s} " + " ".join(cells) + f"   mean {mean:.4f}")
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return rows


def best(rows: dict, prefix: str) -> tuple[str, dict] | None:
    """The threshold of a model with the best mean IDF1 over lab and field."""
    cand = {t: r for t, r in rows.items() if t.startswith(prefix + "-t")}
    if not cand:
        return None
    t = max(cand, key=lambda t: cand[t]["ldev"]["idf1"] + cand[t]["ldev-field"]["idf1"])
    return t, cand[t]


def summary(args):
    """Per setting (models named <setting>-s<seed>): the best threshold of
    each seed, and mean and spread over seeds."""
    rows = table(argparse.Namespace(tags=["B-det-*"], json=None))
    models = sorted({t.rsplit("-t", 1)[0] for t in rows})
    settings: dict[str, list] = {}
    for m in models:
        b = best(rows, m)
        if b is None:
            continue
        setting = m.rsplit("-s", 1)[0] if "-s" in m else m
        settings.setdefault(setting, []).append((m, *b))
    print("\nbest threshold per model; mean (spread) over seeds")
    for s, lst in settings.items():
        lab = np.array([r["ldev"]["idf1"] for _, _, r in lst])
        fld = np.array([r["ldev-field"]["idf1"] for _, _, r in lst])
        names = ", ".join(t.removeprefix(s) for _, t, _ in lst)
        print(
            f"{s:32s} n={len(lst)} lab {lab.mean():.4f} ({np.ptp(lab):.4f}) "
            f"field {fld.mean():.4f} ({np.ptp(fld):.4f}) "
            f"mean {(lab.mean() + fld.mean()) / 2:.4f}  [{names}]"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--model", required=True)
    a.add_argument("--tag", required=True)
    a.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    a.add_argument("--splits", nargs="+", default=list(SPLITS))
    a.add_argument("--jobs", type=int, default=6)
    b = sub.add_parser("table")
    b.add_argument("--tags", nargs="+", required=True)
    b.add_argument("--json")
    sub.add_parser("summary")
    args = p.parse_args(argv)
    {"run": run, "table": table, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
