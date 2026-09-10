"""Build a `single_image` protocol scene from one uploaded RGB image.

The released single-image scenes are a directory per scene:

```
<root>/single_image_valid.txt        scene whitelist the loader intersects with
<root>/data/<scene_id>/rgb.jpeg      the input frame
<root>/data/<scene_id>/aligned_pcd.ply   (H/2, W/2) organized lattice
<root>/data/<scene_id>/camera.json   the camera the render stage replays
```

This module writes that layout for an upload. Two details are load-bearing:

* The RGB resolution must be a multiple of 32 so the half-resolution lattice is
  a multiple of 16, which is what the DINO patch grid and the sampled point grid
  agree on. `fit_image_size` guarantees it, which also makes the driver's centre
  crop a no-op -- so the camera written here stays valid.
* `camera.json` is not decoration: the protocol's view profile is
  `native-input-camera`, and the render stage reads exactly this file through
  `utils.data_single_image.scene_camera`.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from fire3d.webui.config import (
    SCENE_DATA_DIR,
    SCENE_ID_PREFIX,
    WebUIConfig,
)
from fire3d.webui.depth import (
    DepthEstimator,
    back_project,
    depth_preview,
    extent_scale,
    intrinsics_from_fov,
    room_height,
    room_height_scale,
    write_organized_ply,
)
from fire3d.webui.resources import sync_scene_whitelist

CAMERA_SCHEMA = "fire3d_single_image_camera_v1"
PREVIEW_DIR = "previews"

# The lattice stride the protocol center-crops to is 16 on the half-resolution
# grid, so the RGB side has to be a multiple of 32.
IMAGE_MULTIPLE = 32

# The world frame this module writes into: camera at the origin, +y forward,
# +z up. `up` is a world vector here, exactly as `scene_camera` records it.
WORLD_FORWARD = [0.0, 1.0, 0.0]
WORLD_UP = [0.0, 0.0, 1.0]
WORLD_EYE = [0.0, 0.0, 0.0]


@dataclass(frozen=True)
class SceneLayout:
    """Every path one generated scene owns."""

    scene_id: str
    scene_root: Path
    image_dir: Path
    rgb_path: Path
    point_cloud_path: Path
    camera_path: Path
    preview_path: Path


@dataclass(frozen=True)
class SceneSummary:
    """What the UI needs to explain how a scene was built."""

    scene_id: str
    layout: SceneLayout
    height: int
    width: int
    grid_height: int
    grid_width: int
    point_count: int
    valid_points: int
    fov_degrees: float
    depth_model: str
    depth_metric: bool
    depth_device: str
    scale: float
    room_height_before: float
    room_height_after: float
    max_depth: float

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "rgb_size": [self.height, self.width],
            "grid_size": [self.grid_height, self.grid_width],
            "point_count": self.point_count,
            "valid_points": self.valid_points,
            "fov_degrees": self.fov_degrees,
            "depth_model": self.depth_model,
            "depth_metric": self.depth_metric,
            "depth_device": self.depth_device,
            "scale": self.scale,
            "room_height_before_m": self.room_height_before,
            "room_height_after_m": self.room_height_after,
            "max_depth_m": self.max_depth,
        }

    def to_markdown(self) -> str:
        coverage = 100.0 * self.valid_points / max(self.point_count, 1)
        model = self.depth_model if len(self.depth_model) <= 48 else self.depth_model[-45:]
        return "\n".join(
            [
                f"- 场景 ID：`{self.scene_id}`",
                f"- 输入尺寸：{self.width}×{self.height}（点云网格 "
                f"{self.grid_width}×{self.grid_height}）",
                f"- 有效点：{self.valid_points}/{self.point_count}"
                f"（{coverage:.1f}%），视场角 {self.fov_degrees:.0f}°",
                f"- 深度模型：`{model}`（{'metric' if self.depth_metric else 'relative'}，"
                f"{self.depth_device}）",
                f"- 尺度标定：×{self.scale:.3f}，房间高度 "
                f"{self.room_height_before:.2f} m → {self.room_height_after:.2f} m",
            ]
        )


def _noop(_message: str) -> None:
    return None


def new_scene_id() -> str:
    """Timestamped, unique, filesystem-safe scene id.

    The published scenes use zero-padded numeric ids; the WebUI prefix keeps
    uploaded scenes identifiable (and retention-safe) inside a shared root.
    """

    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{SCENE_ID_PREFIX}_{stamp}_{uuid.uuid4().hex[:6]}"


def scene_layout(scene_root: Path, scene_id: str) -> SceneLayout:
    image_dir = Path(scene_root) / SCENE_DATA_DIR / scene_id
    return SceneLayout(
        scene_id=scene_id,
        scene_root=Path(scene_root),
        image_dir=image_dir,
        rgb_path=image_dir / "rgb.jpeg",
        point_cloud_path=image_dir / "aligned_pcd.ply",
        camera_path=image_dir / "camera.json",
        preview_path=Path(scene_root) / PREVIEW_DIR / f"{scene_id}_depth.png",
    )


def fit_image_size(
    height: int,
    width: int,
    *,
    max_side: int,
    multiple: int = IMAGE_MULTIPLE,
) -> tuple[int, int]:
    """Scale an upload down and snap it to the protocol's lattice multiple.

    Only ever shrinks: a larger image would raise the point count, and hence the
    perception cost, above the range the released protocols were validated in.
    """

    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    if multiple < 2:
        raise ValueError("multiple must be at least 2")
    limit = max(int(multiple), int(max_side))
    scale = min(1.0, limit / float(max(height, width)))
    out_height = int(round(height * scale))
    out_width = int(round(width * scale))
    out_height -= out_height % multiple
    out_width -= out_width % multiple
    if out_height < multiple or out_width < multiple:
        raise ValueError(
            f"image {width}x{height} is too extreme to fit {limit}px with a "
            f"{multiple}px lattice; crop it before uploading"
        )
    return out_height, out_width


def load_image(source: np.ndarray | str | Path) -> np.ndarray:
    """Read an upload as HxWx3 RGB uint8."""

    import cv2

    if isinstance(source, (str, Path)):
        buffer = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if buffer is None:
            raise ValueError(f"could not decode image: {source}")
        return cv2.cvtColor(buffer, cv2.COLOR_BGR2RGB)
    array = np.asarray(source)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"expected an HxWx3 image, got shape {array.shape}")
    return np.ascontiguousarray(array[:, :, :3], dtype=np.uint8)


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if image.shape[:2] == (height, width):
        return image
    interpolation = cv2.INTER_AREA if height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def camera_payload(
    scene_id: str,
    height: int,
    width: int,
    fov_degrees: float,
    *,
    provenance: dict | None = None,
) -> dict:
    """The `fire3d_single_image_camera_v1` record for a generated scene.

    The `reconstruction` entry mirrors `utils.data_single_image.scene_camera`
    exactly, including its centre crop, so the record stays correct even if a
    caller ever hand-builds a scene whose lattice is not a multiple of 16.
    """

    focal_x, focal_y, center_x, center_y = intrinsics_from_fov(height, width, fov_degrees)
    frame = {"eye": list(WORLD_EYE), "lookat": list(WORLD_FORWARD), "up": list(WORLD_UP)}
    common = {
        "frame": frame,
        "forward": list(WORLD_FORWARD),
        "source_size": [int(height), int(width)],
    }

    grid_height, grid_width = height // 2, width // 2
    crop_height = (grid_height // 16) * 16
    crop_width = (grid_width // 16) * 16
    top = (grid_height - crop_height) // 2
    left = (grid_width - crop_width) // 2

    native = {
        **common,
        "K": [
            [focal_x, 0.0, center_x],
            [0.0, focal_y, center_y],
            [0.0, 0.0, 1.0],
        ],
        "width": int(width),
        "height": int(height),
        "crop_top": 0,
        "crop_left": 0,
    }
    reconstruction = {
        **common,
        "K": [
            [focal_x, 0.0, center_x - 2.0 * left],
            [0.0, focal_y, center_y - 2.0 * top],
            [0.0, 0.0, 1.0],
        ],
        "width": int(2 * crop_width),
        "height": int(2 * crop_height),
        "crop_top": int(2 * top),
        "crop_left": int(2 * left),
    }
    payload = {
        "schema": CAMERA_SCHEMA,
        "scene_id": scene_id,
        "native": native,
        "reconstruction": reconstruction,
    }
    if provenance:
        payload["webui"] = provenance
    return payload


def build_scene(
    config: WebUIConfig,
    estimator: DepthEstimator,
    image: np.ndarray | str | Path,
    *,
    scene_id: str | None = None,
    register: bool = True,
    log: Callable[[str], None] | None = None,
) -> SceneSummary:
    """Write a complete single-image scene for one RGB frame.

    `estimator` is injected rather than constructed here so the scene layout can
    be tested, and exercised on a machine without the depth checkpoint, without
    loading torch.
    """

    import cv2

    emit = log or _noop
    rgb = load_image(image)
    height, width = fit_image_size(
        int(rgb.shape[0]), int(rgb.shape[1]), max_side=config.max_side
    )
    rgb = resize_image(rgb, height, width)

    emit("[scene] 估计单目深度 …")
    raw_depth = np.asarray(estimator.estimate(rgb), dtype=np.float32)
    if raw_depth.shape[:2] != (height, width):
        raw_depth = cv2.resize(
            raw_depth, (width, height), interpolation=cv2.INTER_NEAREST
        )

    points = back_project(
        raw_depth,
        fov_degrees=config.fov_degrees,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
    )
    before = room_height(points)
    scale = 1.0
    if config.calibrate_room_height or not estimator.metric:
        # A relative backend has no absolute scale at all, so it always needs
        # the room-height anchor; a metric one only needs it as a sanity net.
        scale = room_height_scale(
            points,
            target=config.target_room_height,
            always=not estimator.metric,
        )
    if scale != 1.0:
        points = points * scale
    shrink = extent_scale(points)
    if shrink != 1.0:
        emit(f"[scene] 点云超出 token 范围，整体缩小 ×{shrink:.3f}")
        points = points * shrink
        scale *= shrink

    finite = np.isfinite(points).all(axis=-1)
    valid_points = int(finite.sum())
    if valid_points < 256:
        raise ValueError(
            "深度估计没有得到足够的有效点；请换一张室内照片，"
            "或调整视场角/最大深度后重试。"
        )

    layout = scene_layout(config.scene_root, scene_id or new_scene_id())
    layout.image_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(layout.rgb_path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    ):
        raise OSError(f"could not write {layout.rgb_path}")
    write_organized_ply(layout.point_cloud_path, points)
    layout.camera_path.write_text(
        json.dumps(
            camera_payload(
                layout.scene_id,
                height,
                width,
                config.fov_degrees,
                provenance={
                    "generator": "fire3d-webui",
                    "depth_model": config.depth_model,
                    "depth_metric": bool(estimator.metric),
                    "depth_device": config.depth_device,
                    "scale": scale,
                    "max_depth_m": config.max_depth,
                },
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    layout.preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(layout.preview_path), cv2.cvtColor(depth_preview(raw_depth), cv2.COLOR_RGB2BGR))

    if register:
        entries = sync_scene_whitelist(config.scene_root)
        if layout.scene_id not in entries:
            raise RuntimeError(f"scene {layout.scene_id} was not registered in the whitelist")
        emit(f"[scene] 已登记 {layout.scene_id}（清单共 {len(entries)} 个场景）")

    after = room_height(points)
    return SceneSummary(
        scene_id=layout.scene_id,
        layout=layout,
        height=height,
        width=width,
        grid_height=height // 2,
        grid_width=width // 2,
        point_count=int(finite.size),
        valid_points=valid_points,
        fov_degrees=float(config.fov_degrees),
        depth_model=config.depth_model,
        depth_metric=bool(estimator.metric),
        depth_device=config.depth_device,
        scale=float(scale),
        room_height_before=float(before),
        room_height_after=float(after),
        max_depth=float(config.max_depth),
    )
