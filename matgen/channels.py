"""Derive Occlusion / Roughness / Metalness and pack them into an ORM map.

These are heuristic — there is no ground truth in a Q3 diffuse texture — but they
are grounded in signals we already have: the shading ratio removed by delighting
(occlusion), surface micro-detail (roughness modulation), and the material name
(metalness classification). glTF ORM packing is R=occlusion, G=roughness,
B=metalness.
"""

from __future__ import annotations

import cv2
import numpy as np

# Material-name substrings that imply a metallic surface, with a base roughness.
# Q3 shader paths are descriptive enough that name classification is reliable.
_METAL_KEYS = ("iron", "metal", "steel", "rust", "gold", "silver", "copper",
               "brass", "chrome", "pipe", "grate", "grill", "wire", "fan",
               "vent", "duct", "tin", "alu")
_ORGANIC_KEYS = ("skin", "flesh", "meat", "throat", "organ", "brain")


def classify(name: str):
    """Return (metalness in [0,1], base_roughness) from the material name."""
    n = name.lower()
    if any(k in n for k in _METAL_KEYS):
        return 0.9, 0.35
    if any(k in n for k in _ORGANIC_KEYS):
        return 0.0, 0.55
    return 0.0, 0.75  # stone / brick / wood / concrete default


def _height(albedo: np.ndarray) -> np.ndarray:
    g = (albedo.astype(np.float32) / 255.0).mean(axis=2)
    return cv2.GaussianBlur(g, (0, 0), sigmaX=1.0)


def occlusion(albedo: np.ndarray, shading_ratio: np.ndarray,
              strength: float = 0.7) -> np.ndarray:
    """Cavity AO from height, biased by the shading the delighter removed."""
    h = _height(albedo)
    hw = max(h.shape)
    # Cavity: pixels below their local mean height are occluded.
    local = cv2.GaussianBlur(h, (0, 0), sigmaX=hw / 32.0)
    cavity = np.clip(1.0 - strength * np.clip(local - h, 0, None) * 4.0, 0.0, 1.0)
    # Combine with the low-frequency shading removed during delighting.
    ao = np.minimum(cavity, 0.4 + 0.6 * shading_ratio)
    return np.clip(ao, 0.0, 1.0)


def roughness(albedo: np.ndarray, base: float) -> np.ndarray:
    """Base roughness modulated by micro-detail: busy, high-frequency regions
    read as rougher; smooth flat regions slightly glossier."""
    g = (albedo.astype(np.float32) / 255.0).mean(axis=2)
    detail = np.abs(g - cv2.GaussianBlur(g, (0, 0), sigmaX=2.0))
    detail = np.clip(detail * 6.0, 0.0, 1.0)
    r = base + 0.15 * (detail - 0.3)
    r = cv2.GaussianBlur(r, (0, 0), sigmaX=1.0)
    return np.clip(r, 0.05, 1.0)


def pack_orm(albedo: np.ndarray, shading_ratio: np.ndarray, name: str,
             ao_strength: float = 0.7) -> np.ndarray:
    """Build the HxWx3 uint8 ORM map for a material."""
    metal, base_rough = classify(name)
    ao = occlusion(albedo, shading_ratio, ao_strength)
    rough = roughness(albedo, base_rough)
    metal_map = np.full(albedo.shape[:2], metal, dtype=np.float32)
    orm = np.stack([ao, rough, metal_map], axis=2)
    return (np.clip(orm, 0, 1) * 255.0).round().astype(np.uint8)
