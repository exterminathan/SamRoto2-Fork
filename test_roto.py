"""
test_roto.py — Headless end-to-end rotoscoping test for Sammie-Roto 2.

This is a NON-GUI test harness. It drives the same SAM2 video predictor and
edge-smoothing model that the desktop app uses, so it exercises the real
segmentation -> tracking -> matte pipeline without anyone having to click in
the GUI. The object to mask is chosen automatically with a single positive
point (default: the center of the first frame).

It mirrors the app's own logic:
  * frame extraction copies the colorspace handling from sammie.sammie.load_video
  * segmentation uses sam2.build_sam.build_sam2_video_predictor (Base model)
  * tracking uses predictor.propagate_in_video (same as SamManager.track_objects)
  * edge antialiasing uses sammie.smooth (same model the app's matte views use)

Outputs (under files/output/ by default):
  * masks/NNNNN.png            raw binary mask sequence (one per frame)
  * clip_matte.mp4             antialiased grayscale matte video
  * clip_greenscreen.mp4       subject composited over a green background
  * preview_frame_00000.png    RGBA cutout of the seed frame (transparent bg)

Usage:
    .venv/Scripts/python.exe test_roto.py files/original/clip.mp4
    .venv/Scripts/python.exe test_roto.py <video> --px 0.5 --py 0.4 --model Base
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import av
import cv2
import numpy as np
import torch

# sam2 registers its hydra config search path on import
import sam2  # noqa: F401
from sam2.build_sam import build_sam2_video_predictor
from sammie.smooth import prepare_smoothing_model, run_smoothing_model

# Locate ffmpeg (the repo machine has it at C:\ffmpeg; otherwise rely on PATH)
FFMPEG = r"C:\ffmpeg\ffmpeg.exe" if os.path.exists(r"C:\ffmpeg\ffmpeg.exe") else "ffmpeg"

MODEL_FILES = {
    "Large":     ("./checkpoints/sam2.1_hiera_large.pt",      "./configs/sam2.1/sam2.1_hiera_l.yaml"),
    "Base":      ("./checkpoints/sam2.1_hiera_base_plus.pt",  "./configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "Efficient": ("./checkpoints/efficienttam_s_512x512.pt",  "./configs/sam2.1/efficienttam_s_512x512.yaml"),
}


def log(msg):
    print(f"[roto] {msg}", flush=True)


def pick_device():
    """Mirror sammie.core.DeviceManager: prefer CUDA, set autocast dtype by arch."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        cc_major = torch.cuda.get_device_properties(0).major
        # Ampere+ (sm80) supports bf16; older (e.g. Turing 2080) uses fp16
        amp_dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
        log(f"device=cuda ({torch.cuda.get_device_name(0)}, cc={torch.cuda.get_device_capability()}), amp={amp_dtype}")
        if cc_major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return dev, amp_dtype
    if torch.backends.mps.is_available():
        log("device=mps")
        return torch.device("mps"), torch.bfloat16
    log("device=cpu (this will be slow)")
    return torch.device("cpu"), torch.float32


def extract_frames(video_file, frames_dir):
    """Decode the video to a PNG frame sequence.

    Colorspace handling copied from sammie.sammie.load_video so the test frames
    are byte-identical to what the GUI would produce.
    """
    os.makedirs(frames_dir, exist_ok=True)
    container = av.open(video_file)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    width, height = stream.width, stream.height
    fps = float(stream.average_rate)
    src_cs = int(stream.codec_context.colorspace)      # 1=BT.709, 5=BT.601, ...
    src_range = int(stream.codec_context.color_range)  # 1=limited, 2=full

    count = 0
    for frame in container.decode(stream):
        frame_rgb = frame.reformat(
            format="rgb24",
            src_colorspace=src_cs,
            dst_colorspace=1,    # always output BT.709
            src_color_range=src_range,
            dst_color_range=2,   # always output full range for PNG
        ).to_ndarray()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frames_dir, f"{count:05d}.png"), frame_bgr)
        count += 1
    container.close()
    log(f"extracted {count} frames @ {width}x{height}, {fps:.3f} fps")
    return count, width, height, fps


def main():
    ap = argparse.ArgumentParser(description="Headless rotoscoping test for Sammie-Roto 2")
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--model", default="Base", choices=list(MODEL_FILES), help="SAM2 model (default Base)")
    ap.add_argument("--px", type=float, default=0.5, help="seed point X as fraction of width (default 0.5)")
    ap.add_argument("--py", type=float, default=0.5, help="seed point Y as fraction of height (default 0.5)")
    ap.add_argument("--seed-frame", type=int, default=0, help="frame index to place the seed point on")
    ap.add_argument("--outdir", default="files/output", help="output directory")
    ap.add_argument("--workdir", default="temp_test_roto", help="scratch dir for frames/masks")
    ap.add_argument("--green", default="0,255,0", help="green-screen BGR color, comma separated")
    ap.add_argument("--no-antialias", action="store_true", help="skip the edge-smoothing model")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"input not found: {args.video}")

    ckpt, cfg = MODEL_FILES[args.model]
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint missing: {ckpt}\nDownload it first (see TESTING.md).")

    frames_dir = os.path.join(args.workdir, "frames")
    masks_dir = os.path.join(args.workdir, "masks")
    for d in (frames_dir, masks_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    device, amp_dtype = pick_device()

    # ---- 1. extract frames -------------------------------------------------
    n_frames, W, H, fps = extract_frames(args.video, frames_dir)
    if n_frames == 0:
        sys.exit("no frames decoded")

    seed_x = int(args.px * W)
    seed_y = int(args.py * H)
    seed_frame = max(0, min(args.seed_frame, n_frames - 1))
    obj_id = 1
    log(f"seed: positive point at ({seed_x},{seed_y}) on frame {seed_frame}, object {obj_id}")

    # ---- 2. build SAM2 predictor ------------------------------------------
    log(f"loading SAM2 {args.model} model...")
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)

    smoothing_model = None
    if not args.no_antialias:
        sm_path = "./checkpoints/1x_binary_mask_smooth.pth"
        if os.path.exists(sm_path):
            smoothing_model = prepare_smoothing_model(sm_path, device)
            log("loaded edge-smoothing model")
        else:
            log("edge-smoothing weights not found; masks will be hard-edged")

    autocast_ctx = (torch.autocast("cuda", dtype=amp_dtype)
                    if device.type == "cuda" else torch.autocast(device.type, dtype=amp_dtype)
                    if device.type in ("mps", "xpu") else _nullctx())

    with torch.inference_mode(), autocast_ctx:
        # ---- 3. seed point + segment first frame --------------------------
        log("initializing inference state...")
        state = predictor.init_state(
            video_path=frames_dir, async_loading_frames=True, offload_video_to_cpu=True
        )
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=seed_frame,
            obj_id=obj_id,
            points=np.array([[seed_x, seed_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

        # ---- 4. propagate across all frames -------------------------------
        log("tracking through video...")
        tracked = 0
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            i = out_obj_ids.index(obj_id) if obj_id in out_obj_ids else 0
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(masks_dir, f"{out_frame_idx:05d}.png"), mask)
            tracked += 1
            if out_frame_idx % 25 == 0:
                log(f"  frame {out_frame_idx}/{n_frames}")
        log(f"tracked {tracked} frames")

    # ---- 5. composite outputs ---------------------------------------------
    green = tuple(int(c) for c in args.green.split(","))  # BGR
    gs_dir = os.path.join(args.workdir, "_green")
    mt_dir = os.path.join(args.workdir, "_matte")
    masks_out = os.path.join(args.outdir, "masks")
    for d in (gs_dir, mt_dir, masks_out):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    log("compositing matte + green-screen...")
    bg = np.full((H, W, 3), green, dtype=np.uint8)
    for idx in range(n_frames):
        mpath = os.path.join(masks_dir, f"{idx:05d}.png")
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) if os.path.exists(mpath) else None
        if mask is None:
            mask = np.zeros((H, W), dtype=np.uint8)

        # antialias edges with the same model the app uses for matte views
        if smoothing_model is not None:
            m3 = np.stack([mask] * 3, axis=-1)
            m3 = run_smoothing_model(m3, smoothing_model, device)
            soft = m3[:, :, 0]
        else:
            soft = mask

        # raw (binary) mask copy to output
        cv2.imwrite(os.path.join(masks_out, f"{idx:05d}.png"), mask)
        # antialiased matte frame
        cv2.imwrite(os.path.join(mt_dir, f"{idx:05d}.png"), soft)

        frame = cv2.imread(os.path.join(frames_dir, f"{idx:05d}.png"))
        alpha = (soft.astype(np.float32) / 255.0)[:, :, None]
        comp = (frame.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        cv2.imwrite(os.path.join(gs_dir, f"{idx:05d}.png"), comp)

        # RGBA cutout sample on the seed frame
        if idx == seed_frame:
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = soft
            cv2.imwrite(os.path.join(args.outdir, f"preview_frame_{idx:05d}.png"), rgba)

    # ---- 6. encode preview videos -----------------------------------------
    def encode(src_dir, out_name, grayscale=False):
        out_path = os.path.join(args.outdir, out_name)
        cmd = [
            FFMPEG, "-y", "-framerate", f"{fps}",
            "-i", os.path.join(src_dir, "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            out_path,
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            log(f"ffmpeg failed for {out_name}:\n{r.stderr[-600:]}")
        else:
            log(f"wrote {out_path}")

    encode(mt_dir, "clip_matte.mp4")
    encode(gs_dir, "clip_greenscreen.mp4")

    # cleanup intermediate composite frame dirs (keep masks_out + mp4s)
    shutil.rmtree(gs_dir, ignore_errors=True)
    shutil.rmtree(mt_dir, ignore_errors=True)

    dt = time.time() - t0
    log(f"DONE in {dt:.1f}s")
    log(f"outputs in {args.outdir}:")
    for f in sorted(os.listdir(args.outdir)):
        log(f"  {f}")


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
