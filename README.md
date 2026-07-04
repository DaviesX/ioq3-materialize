# ioq3-materialize

Automated **Q3 diffuse → PBR** material generation. Fills the placeholder
`_albedo` / `_normal` / `_orm` textures that `ioq3-map-exporter` emits, so the
`sh-baker` → `sh-renderer` pipeline gets real PBR maps without hand-authoring
each material in Materialize.

It runs entirely locally on an NVIDIA GPU (developed on an RTX 4060 Ti 16GB),
through two neural models plus classical image ops.

## Pipeline

For every `<name>_diffuse.png` in a material folder:

```
diffuse
  → upscale        Real-ESRGAN x4      (spandrel + torch, CUDA)      models/RealESRGAN_x4plus.pth
  → delight        flatten baked light (low-freq illumination removal + highlight rolloff)
  → normal         DeepBump            (onnxruntime, CUDA, tiled)    models/deepbump256.onnx
  → occlusion      cavity-from-height biased by the shading delight removed
  → roughness      per-material base modulated by micro-detail
  → metalness      material-name classification (iron/metal/rust/... → metallic)
  → pack ORM       R=occlusion  G=roughness  B=metalness
```

Outputs overwrite the exporter's placeholders in place, at the upscaled
resolution (256² → 1024²). Original Q3 art is preserved as `_diffuse.png`.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install onnxruntime-gpu numpy pillow opencv-python-headless
pip install torch --index-url https://download.pytorch.org/whl/cu124   # upscaler backend
pip install spandrel
```

Model weights live in `models/` (git-ignored):
- `deepbump256.onnx` — https://github.com/HugoTini/DeepBump
- `RealESRGAN_x4plus.pth` — https://github.com/xinntao/Real-ESRGAN/releases

## Usage

```bash
# whole map
python materialize.py --data data/q3dm1_v6

# one material
python materialize.py --data data/q3dm1_v6 --only gothic_block@blocks17

# fast preview without the GPU upscaler (Lanczos)
python materialize.py --data data/q3dm1_v6 --no-upscale

# QA contact sheet (diffuse | albedo | normal | orm)
python preview.py --data data/q3dm1_v6 --out preview.png --names gothic_block@blocks17
```

Flags: `--scale`, `--delight-strength`, `--ao-strength`, `--flip-green`
(DirectX-convention normals), `--cpu`, `--dry-run`.

## Quality notes / knobs

- **Normals (DeepBump)** are the strongest output — convincing surface relief.
- **Metalness** is name-based (`matgen/channels.py::_METAL_KEYS`); extend the
  keyword lists for your texture set.
- **Delight** is intentionally conservative (better to under-correct than to
  wrongly flatten real albedo variation). Raise `--delight-strength` for
  textures with heavy directional baked lighting.
- **Roughness** is close to a per-material constant — the most "synthetic"
  channel and the first place to improve (e.g. swap in a learned roughness
  estimator) if surfaces look too uniform under the SH renderer.
