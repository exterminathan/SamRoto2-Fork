# Headless Roto Test (`test_roto.py`)

A non-GUI way to prove the full Sammie-Roto 2 pipeline works
(**frame extraction → SAM2 segmentation → tracking → matte → composite**) on a
single clip, without anyone having to click in the desktop app.

This was written to smoke-test a fresh checkout end to end. The normal,
intended way to rotoscope is the GUI (`run_sammie.bat`); this harness just
automates one object with a single seed point so the ML path can be verified.

---

## 1. One-time setup

### 1a. Install Python dependencies (uv)

The repo ships a private copy of `uv` under `.uv/`. Pick the PyTorch backend
that matches your hardware and sync it into `.venv`:

```bash
# NVIDIA RTX / recent driver  (this machine: RTX 2080 SUPER)
./.uv/uv.exe sync --extra cu130

# other options:
#   ./.uv/uv.exe sync --extra cu126   # older NVIDIA GPUs
#   ./.uv/uv.exe sync --extra xpu     # Intel Arc/Xe
#   ./.uv/uv.exe sync --extra cpu     # no GPU (slow)
```

> The GUI installer (`install.bat` → `manage.py`) does the same thing
> interactively. Running `uv sync` directly is just the scriptable equivalent.

### 1b. Download the SAM2 model

The app normally downloads models on demand via a Qt dialog. For the headless
test, grab the **Base** checkpoint directly (URL + MD5 come from
`sammie/model_downloader.py`):

```bash
curl.exe -L -o checkpoints/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
# expected MD5: ec7bd7d23d280d5e3cfa45984c02eda5
certutil -hashfile checkpoints/sam2.1_hiera_base_plus.pt MD5
```

(`checkpoints/1x_binary_mask_smooth.pth`, used for edge antialiasing, already
ships with the repo.)

---

## 2. Run the test

```bash
.venv/Scripts/python.exe test_roto.py files/original/clip.mp4
```

By default it places **one positive seed point at the center of frame 0** and
tracks that object through the whole clip. If the centered object isn't the one
you want, move the seed point (fractions of width/height):

```bash
# seed at 40% across, 35% down, placed on frame 0
.venv/Scripts/python.exe test_roto.py files/original/clip.mp4 --px 0.40 --py 0.35
```

Useful flags:

| flag | default | meaning |
|------|---------|---------|
| `--model` | `Base` | `Base`, `Large`, or `Efficient` |
| `--px` / `--py` | `0.5` | seed point as a fraction of width/height |
| `--seed-frame` | `0` | which frame to place the seed point on |
| `--green` | `0,255,0` | green-screen BGR color |
| `--no-antialias` | off | skip the edge-smoothing model |
| `--outdir` | `files/output` | where results are written |

---

## 3. Outputs (`files/output/`)

| file | what it is |
|------|------------|
| `masks/NNNNN.png` | raw binary mask, one per frame |
| `clip_matte.mp4` | antialiased grayscale matte (white = subject) |
| `clip_greenscreen.mp4` | subject composited over green |
| `preview_frame_00000.png` | RGBA cutout of the seed frame (transparent bg) |

To sanity-check, open `clip_greenscreen.mp4` — the tracked object should stay
isolated on green for the length of the clip.

---

## 4. How it maps to the real app

`test_roto.py` deliberately reuses the app's own code so the test reflects real
behavior:

- **Frame extraction** copies the colorspace handling from
  `sammie/sammie.py:load_video` (BT.709 / full-range PNGs).
- **Segmentation + tracking** use `sam2.build_sam.build_sam2_video_predictor`
  and `predictor.propagate_in_video` — the same calls as
  `SamManager.track_objects`.
- **Edge antialiasing** uses `sammie/smooth.py`, the same model the GUI's
  matte/alpha views use.

The only things the harness skips are the Qt GUI, interactive point editing,
and the optional MatAnyone/VideoMaMa matting refinement (segmentation masks
alone are enough to verify the core path).

---

## 5. Verified run (2026-06-27)

First end-to-end run on this machine, on `files/original/clip.mp4`:

```
device      : cuda — NVIDIA GeForce RTX 2080 SUPER (cc 7.5), fp16 autocast
clip        : 960x720, 29.68 fps, 334 frames (~11.25 s)
seed        : 1 positive point at (480,360) on frame 0  (subject is centered)
model       : SAM2 Base
tracking    : 334/334 frames @ ~6 fps  (~55 s)
total time  : ~114 s  (extraction + load + track + composite + encode)
outputs     : 334 masks, clip_matte.mp4, clip_greenscreen.mp4, preview_frame_00000.png
```

Result: the subject was cleanly isolated for the whole clip, including arm
movement near the end — no manual point editing needed. This confirms the
install, GPU path, model, and full segmentation→tracking→matte pipeline all
work on this machine.

---

## 6. Notes / gotchas

- **VRAM**: SAM2 Base on an 8 GB card is fine; the harness offloads decoded
  frames to CPU (`offload_video_to_cpu=True`) so memory scales with resolution,
  not clip length.
- **Turing GPUs (e.g. RTX 2080)** use fp16 autocast (no bf16); Ampere+ uses
  bf16. The harness picks this automatically, matching `core.DeviceManager`.
- Scratch frames/masks go in `temp_test_roto/` and can be deleted anytime.
  The desktop app uses its own `temp/` folder, so the two don't collide.
