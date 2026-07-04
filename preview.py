#!/usr/bin/env python3
"""Build a contact sheet (diffuse | albedo | normal | orm) for visual QA.

    python preview.py --data data/q3dm1_v6 --out preview.png --names a b c
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

TILE = 256
COLS = ("diffuse", "albedo", "normal", "orm")


def load(folder, name, kind, size):
    p = os.path.join(folder, f"{name}_{kind}.png")
    if not os.path.isfile(p):
        return Image.new("RGB", (size, size), (40, 40, 40))
    return Image.open(p).convert("RGB").resize((size, size), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="preview.png")
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args()

    names = args.names
    if not names:
        names = [os.path.basename(d)[: -len("_diffuse.png")]
                 for d in sorted(os.listdir(args.data))]
    header = 22
    rows = []
    for name in names:
        folder = os.path.join(args.data, name)
        if not os.path.isdir(folder):
            continue
        rows.append((name, folder))

    W = TILE * len(COLS)
    H = (TILE + header) * len(rows)
    sheet = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for r, (name, folder) in enumerate(rows):
        y = r * (TILE + header)
        draw.text((6, y + 5), f"{name}   [{'  '.join(COLS)}]", fill=(230, 230, 230))
        for c, kind in enumerate(COLS):
            sheet.paste(load(folder, name, kind, TILE), (c * TILE, y + header))
    sheet.save(args.out)
    print(f"wrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
