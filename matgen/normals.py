"""Tangent-space normal map generation via DeepBump (color->normals ONNX).

DeepBump expects a single grayscale channel and operates on 256x256 tiles. We
tile the full-resolution (post-upscale) image with overlap and blend the tiles
with a Hann window to avoid seams. Output is an OpenGL-convention normal map
(+Y up), matching glTF tangent-space expectations.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "models",
                           "deepbump256.onnx")
_TILE = 256


@lru_cache(maxsize=1)
def _session(device: str):
    import onnxruntime as ort
    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(os.path.abspath(_MODEL_PATH), providers=providers)


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n).astype(np.float32)
    w = np.clip(w, 1e-3, None)
    return np.outer(w, w)


def _tile_starts(size: int, tile: int, overlap: int):
    step = tile - overlap
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, step))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def normals(gray_rgb: np.ndarray, device: str = "cuda",
            overlap: int = 32, flip_green: bool = False) -> np.ndarray:
    """Generate a normal map from an HxWx3 uint8 image. Returns uint8 RGB.

    Only luminance is used; color is ignored (DeepBump is a height-from-shading
    style predictor).
    """
    sess = _session(device)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    g = (gray_rgb.astype(np.float32) / 255.0).mean(axis=2)
    h, w = g.shape

    # Pad up to at least one tile.
    pad_h = max(0, _TILE - h)
    pad_w = max(0, _TILE - w)
    if pad_h or pad_w:
        g = np.pad(g, ((0, pad_h), (0, pad_w)), mode="reflect")
    H, W = g.shape

    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    win = _hann2d(_TILE)[..., None]

    for y in _tile_starts(H, _TILE, overlap):
        for x in _tile_starts(W, _TILE, overlap):
            tile = g[y:y + _TILE, x:x + _TILE][None, None]  # 1x1xTxT
            pred = sess.run([out_name], {in_name: tile.astype(np.float32)})[0]
            pred = pred[0].transpose(1, 2, 0)  # TxTx3
            acc[y:y + _TILE, x:x + _TILE] += pred * win
            wsum[y:y + _TILE, x:x + _TILE] += win

    nrm = acc / np.clip(wsum, 1e-6, None)
    nrm = nrm[:h, :w]

    # Re-normalize direction vectors for a clean, unit-length normal map.
    v = nrm * 2.0 - 1.0
    v /= np.clip(np.linalg.norm(v, axis=2, keepdims=True), 1e-6, None)
    if flip_green:
        v[..., 1] = -v[..., 1]
    out = (v * 0.5 + 0.5)
    return (np.clip(out, 0, 1) * 255.0).round().astype(np.uint8)
