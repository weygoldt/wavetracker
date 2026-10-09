#!/usr/bin/env python
"""Write a shields-style coverage badge from a coverage.py report.

    python scripts/coverage_badge.py coverage.json coverage.svg
    python scripts/coverage_badge.py coverage.xml coverage.svg

Reads the total from coverage's JSON report (`--cov-report=json`) or its
Cobertura XML report (`--cov-report=xml`) and writes a flat, two-part badge,
"coverage | 87%", coloured from red to bright green.  Standard library only,
so CI can run it with any Python and nothing installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

#: Advance widths of Verdana, in units of its 2048-unit em, for the
#: characters a badge is made of.  Text is laid out at 11 px, so a character
#: is `width * 11 / 2048` pixels wide.
VERDANA = {
    **dict.fromkeys("0123456789", 1303),
    "a": 1229, "b": 1276, "c": 1067, "d": 1276, "e": 1220, "f": 720,
    "g": 1276, "h": 1296, "i": 562, "j": 680, "k": 1196, "l": 562,
    "m": 1992, "n": 1296, "o": 1243, "p": 1276, "q": 1276, "r": 874,
    "s": 1067, "t": 807, "u": 1296, "v": 1196, "w": 1655, "x": 1194,
    "y": 1196, "z": 1067, "%": 2786, ".": 745, " ": 720, "-": 862,
}  # fmt: skip
FONT_SIZE = 11
#: horizontal padding either side of each half's text
PAD = 6

#: (lowest percentage, colour), shields' palette, best first
COLOURS = [
    (90, "#4c1"),  # brightgreen
    (80, "#97ca00"),  # green
    (70, "#a4a61d"),  # yellowgreen
    (60, "#dfb317"),  # yellow
    (50, "#fe7d37"),  # orange
    (0, "#e05d44"),  # red
]


def read_percent(path: Path) -> float:
    """Total line coverage, in percent, from a JSON or Cobertura XML report."""
    if path.suffix == ".xml":
        root = ET.parse(path).getroot()
        return 100.0 * float(root.attrib["line-rate"])
    with path.open() as f:
        return float(json.load(f)["totals"]["percent_covered"])


def text_width(text: str) -> float:
    return sum(VERDANA.get(c, 1303) for c in text) * FONT_SIZE / 2048


def colour(percent: float) -> str:
    return next(c for lo, c in COLOURS if percent >= lo)


def badge(label: str, value: str, fill: str) -> str:
    lw = round(text_width(label)) + 2 * PAD
    vw = round(text_width(value)) + 2 * PAD
    w = lw + vw
    title = escape(f"{label}: {value}")

    def text(x: float, s: str, length: float) -> str:
        s = escape(s)
        common = f'x="{x}" textLength="{length:.1f}" lengthAdjust="spacing"'
        return (
            f'<text {common} y="15" fill="#010101" fill-opacity=".3">{s}</text>'
            f'<text {common} y="14">{s}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" '
        f'role="img" aria-label="{title}">'
        f"<title>{title}</title>"
        '<linearGradient id="s" x2="0" y2="100%">'
        '<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        '<stop offset="1" stop-opacity=".1"/></linearGradient>'
        f'<clipPath id="r"><rect width="{w}" height="20" rx="3" fill="#fff"/>'
        "</clipPath>"
        f'<g clip-path="url(#r)"><rect width="{lw}" height="20" fill="#555"/>'
        f'<rect x="{lw}" width="{vw}" height="20" fill="{fill}"/>'
        f'<rect width="{w}" height="20" fill="url(#s)"/></g>'
        '<g fill="#fff" text-anchor="middle" '
        'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" '
        f'text-rendering="geometricPrecision" font-size="{FONT_SIZE}">'
        f"{text(lw / 2, label, text_width(label))}"
        f"{text(lw + vw / 2, value, text_width(value))}"
        "</g></svg>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="coverage.json or coverage.xml")
    parser.add_argument("output", type=Path, help="the SVG to write")
    parser.add_argument("--label", default="coverage")
    args = parser.parse_args(argv)

    # Coloured by the number shown, so "90%" is never drawn in plain green.
    percent = round(read_percent(args.report))
    value = f"{percent}%"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(badge(args.label, value, colour(percent)))
    print(f"{args.label}: {value} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
