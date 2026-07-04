"""Per-material orchestration: diffuse -> albedo / normal / orm.

Reads `<name>_diffuse.png` from a material folder and overwrites the exporter's
placeholder `<name>_albedo.png`, `<name>_normal.png`, `<name>_orm.png` with
generated PBR maps.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from . import channels, delight, normals, upscale


def _load_rgba(path: str):
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    return arr[..., :3], arr[..., 3]


def _save(path: str, rgb: np.ndarray, alpha: np.ndarray | None = None):
    if alpha is not None:
        out = np.dstack([rgb, alpha])
        Image.fromarray(out, "RGBA").save(path)
    else:
        Image.fromarray(rgb, "RGB").save(path)


def process_material(folder: str, name: str, cfg) -> bool:
    diffuse_path = os.path.join(folder, f"{name}_diffuse.png")
    if not os.path.isfile(diffuse_path):
        return False

    rgb, alpha = _load_rgba(diffuse_path)
    scale = 1 if cfg.no_upscale else cfg.scale

    # 1. upscale
    if scale > 1:
        big = upscale.upscale(rgb, scale=scale,
                              backend="lanczos" if cfg.no_upscale else "esrgan",
                              device=cfg.device)
        big_alpha = cv2.resize(alpha, (big.shape[1], big.shape[0]),
                               interpolation=cv2.INTER_LANCZOS4)
    else:
        big, big_alpha = rgb, alpha

    # 2. delight (at upscaled resolution so detail is preserved)
    albedo, shading = delight.delight(big, strength=cfg.delight_strength)

    # 3. normals
    nrm = normals.normals(albedo, device=cfg.device, flip_green=cfg.flip_green)

    # 4. ORM (shading ratio is same res as albedo here)
    orm = channels.pack_orm(albedo, shading, name, ao_strength=cfg.ao_strength)

    if cfg.dry_run:
        return True

    keep_alpha = big_alpha if (big_alpha < 250).any() else None
    _save(os.path.join(folder, f"{name}_albedo.png"), albedo, keep_alpha)
    _save(os.path.join(folder, f"{name}_normal.png"), nrm)
    _save(os.path.join(folder, f"{name}_orm.png"), orm)
    return True
