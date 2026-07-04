"""Delighting: remove baked-in illumination from a Q3 diffuse texture.

Q3 textures have lighting, directional shading, and ambient-occlusion painted
directly into the diffuse map. Feeding that straight into a PBR + SH-GI pipeline
double-darkens (the renderer adds its own light and AO on top). This module
flattens the low-frequency illumination field and compresses baked highlights,
while preserving surface detail and chroma.

It also returns the *shading ratio* it removed, which `channels.py` reuses as a
free ambient-occlusion prior — the crevice darkening we strip out of albedo is
exactly what belongs in the ORM occlusion channel.
"""

from __future__ import annotations

import cv2
import numpy as np

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ _LUMA


def delight(rgb: np.ndarray, strength: float = 0.8):
    """Flatten baked lighting.

    Args:
        rgb: HxWx3 uint8.
        strength: 0 = passthrough, 1 = fully flatten the low-frequency field.

    Returns:
        (albedo_uint8, shading_ratio_float) where shading_ratio in [0,1] is the
        multiplicative shading that was removed (1 = untouched, <1 = was darker).
    """
    f = rgb.astype(np.float32) / 255.0
    h, w = f.shape[:2]
    lum = _luminance(f) + 1e-4

    # Low-frequency illumination estimate: large-sigma blur captures the broad
    # lighting gradient (a wall lit from above, a corner in shadow) without
    # touching per-brick detail.
    sigma = max(h, w) / 8.0
    low = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ref = float(np.median(low))

    # Gain that pulls the low-freq field back to a flat reference level. Clamp so
    # we never blow out very dark or very bright regions.
    gain = ref / low
    gain = np.clip(gain, 0.5, 2.0)
    gain = 1.0 + strength * (gain - 1.0)

    flat = f * gain[..., None]

    # Compress baked specular highlights: pixels far above the local mean get
    # rolled back toward it (soft knee), so painted-in glints don't survive as
    # bright albedo.
    local_mean = cv2.GaussianBlur(flat, (0, 0), sigmaX=sigma / 2, sigmaY=sigma / 2)
    over = np.clip(flat - local_mean - 0.25, 0.0, None)
    flat = flat - strength * 0.5 * over

    albedo = np.clip(flat, 0.0, 1.0)

    # Shading ratio removed (for AO reuse): how much darker the original was
    # relative to the delit result, in luminance.
    new_lum = _luminance(albedo) + 1e-4
    ratio = np.clip(lum / new_lum, 0.0, 1.0)
    ratio = cv2.GaussianBlur(ratio, (0, 0), sigmaX=1.5, sigmaY=1.5)

    return (albedo * 255.0).round().astype(np.uint8), ratio
