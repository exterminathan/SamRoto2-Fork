# sammie/depth.py
"""
Monocular depth estimation (Depth Anything V2).

Runs a per-frame depth pass over the loaded video and stores a whole-frame
grayscale depth map per frame in ``temp/depth/{frame:05d}.png`` (near = white,
far = black). Unlike segmentation/matting, depth is a whole-frame property and
is therefore NOT stored per-object. The subject confinement (Depth-Matte) is
done at render time by multiplying the depth map by the segmentation mask
(see sammie.sammie._handle_depth_matte_view).

This mirrors the matting workflow: a "Run Depth" button triggers a precompute
pass; afterwards the live preview and export just read the cached PNGs.
"""
import os
import gc
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt

from sammie import core
from sammie.settings_manager import get_settings_manager
from sammie.model_downloader import ensure_models
from depth_anything_v2.dpt import DepthAnythingV2


# Encoder configurations for the supported model sizes. Each maps to a registry
# key (sammie/model_downloader.py) + checkpoint filename + constructor kwargs.
MODEL_CONFIGS = {
    "Small": {
        "registry_key": "depth_small",
        "checkpoint": "./checkpoints/depth_anything_v2_vits.pth",
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "Large": {
        "registry_key": "depth_large",
        "checkpoint": "./checkpoints/depth_anything_v2_vitl.pth",
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


class DepthManager:
    """Loads Depth Anything V2 and runs a per-frame depth precompute pass."""

    def __init__(self):
        self.model = None
        self.loaded_model_name = None
        self.propagated = False  # whether depth maps have been computed

    # -- model lifecycle ----------------------------------------------------

    def load_depth_model(self, parent_window=None):
        """Build the depth model and load its checkpoint onto the active device."""
        settings_mgr = get_settings_manager()
        model_size = settings_mgr.get_session_setting("depth_model", "Large")
        config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["Large"])

        core.DeviceManager.clear_cache()
        device = core.DeviceManager.get_device()

        if not ensure_models(config["registry_key"], parent=parent_window):
            return False

        try:
            model = DepthAnythingV2(
                encoder=config["encoder"],
                features=config["features"],
                out_channels=config["out_channels"],
            )
            state = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            self.model = model.to(device).eval()
            self.loaded_model_name = model_size
        except Exception as e:
            print(f"Failed to load Depth Anything V2 ({model_size}): {e}")
            self.model = None
            return False

        return True

    def unload_depth_model(self):
        """Release the model and free VRAM."""
        self.model = None
        self.loaded_model_name = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded Depth model")

    # -- inference ----------------------------------------------------------

    def _infer_depth(self, bgr_image, device, input_size):
        """Run the model on a BGR frame, returning an HxW float32 depth array.

        Mirrors DepthAnythingV2.infer_image but forces the active device so the
        force_cpu setting is honored (infer_image picks cuda-if-available).
        """
        image_t, (h, w) = self.model.image2tensor(bgr_image, input_size)
        image_t = image_t.to(device)
        with torch.no_grad():
            depth = self.model(image_t)
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.detach().cpu().numpy()

    @staticmethod
    def _normalize_depth(depth):
        """Normalize a float depth map to uint8 grayscale (near = white)."""
        d_min = float(depth.min())
        d_max = float(depth.max())
        norm = (depth - d_min) / (d_max - d_min + 1e-8)
        return (norm * 255.0).clip(0, 255).astype(np.uint8)

    def run_depth(self, parent_window=None):
        """Compute depth for every frame in the in/out range into temp/depth/.

        Does NOT require points/tracking (depth is whole-frame). Only requires a
        loaded video (extracted frames). Returns 1 on success, 0 on cancel/fail.
        """
        if self.model is None:
            print("Depth model is not loaded")
            return 0

        settings_mgr = get_settings_manager()
        input_size = settings_mgr.get_session_setting("depth_res", 518)
        device = core.DeviceManager.get_device()

        frame_count = core.VideoInfo.total_frames
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        total = end_frame - start_frame + 1
        if total <= 0:
            print("No frames to process for depth")
            return 0

        os.makedirs(core.depth_dir, exist_ok=True)
        extension = core.get_frame_extension()

        progress_dialog = QProgressDialog("Running depth estimation...", "Cancel", 0, 100, parent_window)
        progress_dialog.setWindowTitle("Depth Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()
        pbar = tqdm(total=total, desc="Depth Progress", unit="frame")

        cancelled = False
        try:
            for i, frame_number in enumerate(range(start_frame, end_frame + 1)):
                if progress_dialog.wasCanceled():
                    cancelled = True
                    break

                frame_path = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")
                if not os.path.exists(frame_path):
                    continue
                bgr = cv2.imread(frame_path)
                if bgr is None:
                    continue

                depth = self._infer_depth(bgr, device, input_size)
                depth_u8 = self._normalize_depth(depth)

                out_path = os.path.join(core.depth_dir, f"{frame_number:05d}.png")
                cv2.imwrite(out_path, depth_u8)
                core.DeviceManager.clear_cache()

                # Periodically advance the on-screen frame so the user sees progress
                if frame_number % 10 == 0 and parent_window is not None:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                pbar.update(1)
                progress_dialog.setValue(int((i + 1) * 100 / total))
                QApplication.processEvents()
        finally:
            pbar.close()
            core.DeviceManager.clear_cache()

        if cancelled:
            progress_dialog.close()
            self.propagated = False
            return 0

        self.propagated = True
        return 1

    # -- data management ----------------------------------------------------

    def clear_depth(self):
        """Remove all computed depth maps."""
        if os.path.exists(core.depth_dir):
            shutil.rmtree(core.depth_dir)
        os.makedirs(core.depth_dir, exist_ok=True)
        self.propagated = False
        print("Depth data cleared")
