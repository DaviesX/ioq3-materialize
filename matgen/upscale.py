"""Super-resolution of a low-res Q3 diffuse texture.

Primary backend: Real-ESRGAN (x4) loaded from the official .pth via `spandrel`,
run on the GPU through torch. Falls back to Lanczos (cv2) when torch/spandrel or
a CUDA device is unavailable, or when the caller passes `backend="lanczos"`.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "models",
                           "RealESRGAN_x4plus.pth")


@lru_cache(maxsize=1)
def _load_esrgan(device: str):
    """Load the Real-ESRGAN descriptor once. Returns (model, torch) or None."""
    try:
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
    except Exception as e:  # torch/spandrel not installed
        print(f"  [upscale] torch/spandrel unavailable ({e}); using Lanczos")
        return None
    if device == "cuda" and not torch.cuda.is_available():
        print("  [upscale] CUDA unavailable; using Lanczos")
        return None
    model = ModelLoader().load_from_file(os.path.abspath(_MODEL_PATH))
    if not isinstance(model, ImageModelDescriptor):
        print("  [upscale] unexpected model type; using Lanczos")
        return None
    model.to(device).eval()
    return model, torch


def upscale(rgb: np.ndarray, scale: int = 4, backend: str = "esrgan",
            device: str = "cuda") -> np.ndarray:
    """Upscale an HxWx3 uint8 RGB image by `scale`. Returns uint8 RGB.

    The Real-ESRGAN model is natively x4; if `scale` differs we resample the
    x4 result to the requested factor with Lanczos.
    """
    h, w = rgb.shape[:2]
    target = (w * scale, h * scale)

    loaded = _load_esrgan(device) if backend == "esrgan" else None
    if loaded is None:
        return cv2.resize(rgb, target, interpolation=cv2.INTER_LANCZOS4)

    model, torch = loaded
    inp = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    inp = inp.permute(2, 0, 1).unsqueeze(0).to(device)  # 1x3xHxW
    with torch.no_grad():
        out = model(inp).clamp(0.0, 1.0)
    out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

    if out.shape[1] != target[0] or out.shape[0] != target[1]:
        out = cv2.resize(out, target, interpolation=cv2.INTER_LANCZOS4)
    return out
