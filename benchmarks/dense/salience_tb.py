"""Mirror a salience_train.py log into TensorBoard events while it grows.

uv run --with tensorboard python benchmarks/dense/salience_tb.py --tag v1
uvx --from tensorboard tensorboard --logdir ~/wrk/data/wavetracker_benchmark/salience
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA

STEP = re.compile(r"^step (\d+): bce ([\d.]+) l1 ([\d.]+) \(([\d.]+) s/step\)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--poll", type=float, default=15.0)
    args = p.parse_args(argv)
    run = DATA / "salience" / args.tag
    log = run / "train.log"
    writer = SummaryWriter(str(run / "tb"))
    pos = 0
    while True:
        text = log.read_text() if log.exists() else ""
        new, pos = text[pos:], len(text)
        for line in new.splitlines():
            if m := STEP.match(line):
                s = int(m[1])
                writer.add_scalar("train/bce", float(m[2]), s)
                writer.add_scalar("train/offset_l1", float(m[3]), s)
                writer.add_scalar("train/seconds_per_step", float(m[4]), s)
            elif line.startswith("{"):
                row = json.loads(line)
                s = row["step"]
                for k, v in row.items():
                    if k.endswith(("_lab", "_field")):
                        metric, variant = k.rsplit("_", 1)
                        writer.add_scalar(f"val_{variant}/{metric}", v, s)
        writer.flush()
        if "done in" in new or "Traceback" in new:
            break
        time.sleep(args.poll)
    writer.close()


if __name__ == "__main__":
    main()
