# Copyright (c) Meta Platforms, Inc. and affiliates.
# This adapter follows BoxeR's CC-BY-NC 4.0 licensed loader interfaces.

"""Fire3D RGB-D loader for the released BoxeR inference code.

The Imaginarium camera JSON stores OpenCV-style camera-to-world transforms:
x right, y down, z forward.  BoxeR's pinhole camera uses the same convention,
while the world is already z-up, so no axis permutation is required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


def _camera_to_world(frame: dict) -> np.ndarray:
    eye = np.asarray(frame["eye"], dtype=np.float32)
    lookat = np.asarray(frame["lookat"], dtype=np.float32)
    up_hint = np.asarray(frame["up"], dtype=np.float32)

    forward = lookat - eye
    forward /= np.linalg.norm(forward).clip(1e-8)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right).clip(1e-8)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up).clip(1e-8)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.column_stack([right, -up, forward])
    transform[:3, 3] = eye
    return transform


class ImaginariumLoader(BaseLoader):
    """Yield one Fire3D scene in BoxeR's per-frame datum format."""

    camera = "rgb"
    device_name = "Imaginarium"

    def __init__(
        self,
        dataset_root: str,
        scene_name: str,
        video_id: int = 0,
        skip_frames: int = 1,
        max_frames: int | None = None,
        start_frame: int = 1,
        sdp_points: int = 10_000,
        seed: int = 20260727,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.scene_name = scene_name
        self.video_id = int(video_id)
        self.sdp_points = int(sdp_points)
        self.seed = int(seed)
        self.resize = None

        render_root = self.dataset_root / "renders" / scene_name
        self.camera_path = render_root / f"{self.video_id}.json"
        self.depth_dir = render_root / f"{self.video_id}_depth"
        self.rgb_source = os.environ.get("FF_BOXER_RGB_SOURCE", "native")
        if self.rgb_source == "native":
            self.frames_dir = render_root / f"{self.video_id}_frames"
        elif self.rgb_source == "updated":
            self.frames_dir = (
                self.dataset_root
                / "renders_updated"
                / scene_name
                / f"{self.video_id}_frames"
            )
        else:
            raise ValueError(
                "FF_BOXER_RGB_SOURCE must be 'native' or 'updated', found "
                f"{self.rgb_source!r}"
            )
        if not self.camera_path.is_file():
            raise FileNotFoundError(self.camera_path)
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(self.frames_dir)

        camera = json.loads(self.camera_path.read_text(encoding="utf-8"))
        self.K = np.asarray(camera["K"], dtype=np.float32).reshape(3, 3)
        self.source_width = int(camera["width"])
        self.source_height = int(camera["height"])
        self.camera_frames = camera["frames"]

        frame_paths = sorted(
            path
            for path in self.frames_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        depth_paths = sorted(self.depth_dir.glob("*.npz"))
        if len(frame_paths) != len(depth_paths):
            raise ValueError(
                f"{scene_name}: {len(frame_paths)} RGB frames but "
                f"{len(depth_paths)} depth frames"
            )
        if len(frame_paths) != len(self.camera_frames):
            raise ValueError(
                f"{scene_name}: {len(frame_paths)} RGB frames but "
                f"{len(self.camera_frames)} camera records"
            )

        start = max(int(start_frame) - 1, 0)
        indices = list(range(start, len(frame_paths), max(int(skip_frames), 1)))
        if max_frames is not None:
            indices = indices[: int(max_frames)]
        self.records = [
            (index, frame_paths[index], depth_paths[index]) for index in indices
        ]
        self.length = len(self.records)
        self.index = 0
        print(
            f"ImaginariumLoader: {scene_name}, {self.length} / "
            f"{len(frame_paths)} frames, rgb_source={self.rgb_source}"
        )
        self._init_prefetch()

    @staticmethod
    def _read_depth(path: Path) -> np.ndarray:
        with np.load(path) as archive:
            if "depth" in archive:
                return archive["depth"].astype(np.float32)
            if "arr_0" in archive:
                return archive["arr_0"].astype(np.float32)
            raise KeyError(f"Unsupported depth keys in {path}: {archive.files}")

    def _sample_sdp(
        self,
        depth: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        c2w: np.ndarray,
        frame_index: int,
    ) -> torch.Tensor:
        height, width = depth.shape
        step = max(1, int(np.sqrt(height * width / (self.sdp_points * 2))))
        yy, xx = np.mgrid[0:height:step, 0:width:step]
        yy = yy.ravel()
        xx = xx.ravel()
        zz = depth[yy, xx]
        valid = np.isfinite(zz) & (zz > 0) & (zz <= 50.0)
        yy, xx, zz = yy[valid], xx[valid], zz[valid]

        if len(zz) > self.sdp_points:
            rng = np.random.default_rng(self.seed + int(frame_index))
            chosen = rng.choice(len(zz), size=self.sdp_points, replace=False)
            yy, xx, zz = yy[chosen], xx[chosen], zz[chosen]
        if len(zz) == 0:
            return torch.zeros(0, 3, dtype=torch.float32)

        camera_points = np.stack(
            [
                (xx.astype(np.float32) - cx) * zz / fx,
                (yy.astype(np.float32) - cy) * zz / fy,
                zz,
            ],
            axis=-1,
        )
        world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
        result = torch.from_numpy(world_points.astype(np.float32))
        if len(result) < self.sdp_points:
            padding = torch.full(
                (self.sdp_points - len(result), 3),
                float("nan"),
                dtype=torch.float32,
            )
            result = torch.cat([result, padding], dim=0)
        return result

    def load(self, idx: int) -> dict:
        frame_index, frame_path, depth_path = self.records[idx]
        image = cv2.imread(os.fspath(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        source_height, source_width = image.shape[:2]
        depth = self._read_depth(depth_path)
        if depth.shape != (source_height, source_width):
            depth = cv2.resize(
                depth,
                (source_width, source_height),
                interpolation=cv2.INTER_NEAREST,
            )

        output_height = source_height
        output_width = source_width
        if self.resize is not None:
            output_height = output_width = int(self.resize)
            image = cv2.resize(
                image,
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            )
            depth = cv2.resize(
                depth,
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            )

        scale_x = output_width / source_width
        scale_y = output_height / source_height
        fx = float(self.K[0, 0] * scale_x)
        fy = float(self.K[1, 1] * scale_y)
        cx = float(self.K[0, 2] * scale_x)
        cy = float(self.K[1, 2] * scale_y)
        c2w = _camera_to_world(self.camera_frames[frame_index])

        cam = self.pinhole_from_K(
            output_width,
            output_height,
            fx,
            fy,
            cx,
            cy,
            valid_radius=(output_width, output_height),
        )
        transform_data = torch.tensor(
            [*c2w[:3, :3].reshape(-1), *c2w[:3, 3]],
            dtype=torch.float32,
        )
        datum = {
            "img0": self.img_to_tensor(image),
            "cam0": cam.float(),
            "T_world_rig0": PoseTW(transform_data),
            "sdp_w": self._sample_sdp(depth, fx, fy, cx, cy, c2w, frame_index),
            "time_ns0": int(frame_index) * 100_000_000,
            "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
            "obbs": ObbTW(torch.zeros(0, 165)),
            "gt_labels": [],
            "frame_index": int(frame_index),
            "frame_path": os.fspath(frame_path),
            "depth_path": os.fspath(depth_path),
            "rgb_source": self.rgb_source,
            "source_width": int(source_width),
            "source_height": int(source_height),
            "camera_to_world": c2w,
        }
        return datum
