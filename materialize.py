#!/usr/bin/env python3
"""ioq3-materialize CLI.

Batch-generate PBR maps (albedo / normal / ORM) for every material folder under
a data directory, replacing the exporter's placeholder textures in place.

    python materialize.py --data data/q3dm1_v6
    python materialize.py --data data/q3dm1_v6 --only gothic_block@blocks17
    python materialize.py --data data/q3dm1_v6 --no-upscale     # fast Lanczos, no GPU
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass

from matgen import pipeline


@dataclass
class Config:
    scale: int = 4
    no_upscale: bool = False
    delight_strength: float = 0.8
    ao_strength: float = 0.7
    flip_green: bool = False
    device: str = "cuda"
    dry_run: bool = False


def find_materials(data_dir: str):
    for diffuse in sorted(glob.glob(os.path.join(data_dir, "*", "*_diffuse.png"))):
        folder = os.path.dirname(diffuse)
        name = os.path.basename(diffuse)[: -len("_diffuse.png")]
        yield folder, name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dir containing material subfolders")
    ap.add_argument("--only", default=None, help="process only this material name")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--no-upscale", action="store_true", help="skip Real-ESRGAN (Lanczos)")
    ap.add_argument("--delight-strength", type=float, default=0.8)
    ap.add_argument("--ao-strength", type=float, default=0.7)
    ap.add_argument("--flip-green", action="store_true", help="flip normal Y (DirectX)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--dry-run", action="store_true", help="process but don't write")
    args = ap.parse_args()

    cfg = Config(scale=args.scale, no_upscale=args.no_upscale,
                 delight_strength=args.delight_strength, ao_strength=args.ao_strength,
                 flip_green=args.flip_green, device="cpu" if args.cpu else "cuda",
                 dry_run=args.dry_run)

    mats = list(find_materials(args.data))
    if args.only:
        mats = [(f, n) for f, n in mats if n == args.only]
    if not mats:
        print("no materials found")
        return

    print(f"processing {len(mats)} materials (device={cfg.device}, "
          f"upscale={'lanczos' if cfg.no_upscale else f'x{cfg.scale} esrgan'})")
    t0 = time.time()
    ok = 0
    for i, (folder, name) in enumerate(mats, 1):
        ts = time.time()
        try:
            done = pipeline.process_material(folder, name, cfg)
            ok += bool(done)
            print(f"[{i:>3}/{len(mats)}] {name:<45} {time.time()-ts:5.1f}s")
        except Exception as e:
            print(f"[{i:>3}/{len(mats)}] {name:<45} FAILED: {e}")
    print(f"done: {ok}/{len(mats)} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
