"""Monocular depth for the Fire3D WebUI, and its back-projection to a cloud.

The released `single_image` protocol conditions perception *and* reconstruction
on an organized point cloud, not on the RGB image alone: every scene ships
`data/<scene_id>/aligned_pcd.ply`, an `(H/2, W/2)` lattice that the perception
loader subsamples and the reconstruction center-crops. So an arbitrary upload
needs one extra step before the frozen pipeline can run -- estimate depth and
back-project it into the same gravity-aligned world frame the released scenes
use.

Frame contract (this is the part worth getting exactly right):

* The protocol's world frame has **z up**: `utils.single_image_room_walls`
  reads the floor and ceiling from the z percentiles, and the camera records
  `up` as a vector in that frame.
* The camera looks along **+y**, matching `scene_camera`'s
  `forward = rotation @ (0, 0, 1)` with the camera at the origin.
* Object coordinates are **metres**, because the conditioning cloud is compared
  against `scene_scale = 24.0` and the room-box fit thresholds are metric.

So back-projection maps OpenCV camera axes to the world as
`(x, y, z)_cam -> (x, z, -y)_world`, which puts +z up and +y forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from fire3d.webui.config import WebUIConfig

LogFn = Callable[[str], None]

# The frozen protocol's position token range. A cloud larger than this cannot be
# represented by the scene normalization, so it is only ever scaled down.
SCENE_EXTENT_LIMIT_M = 24.0

# Plausible indoor floor-to-ceiling band. Heights inside the band are trusted;
# the metric checkpoints land here on their own.
MIN_ROOM_HEIGHT_M = 1.8
MAX_ROOM_HEIGHT_M = 5.0

PLY_HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "comment Fire3D WebUI organized depth lattice\n"
    "element vertex {count}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "end_header\n"
)


class DepthEstimator(Protocol):
    """Anything that maps an HxWx3 uint8/float RGB image to an HxW depth map."""

    metric: bool

    def describe(self) -> str: ...

    def estimate(self, image: np.ndarray) -> np.ndarray: ...


@dataclass
class FlatDepthEstimator:
    """Constant-depth fallback: a flat wall at a fixed distance.

    Only useful for exercising the WebUI plumbing without the depth checkpoint.
    The reconstruction it produces is a scene at one depth, which is not a real
    reconstruction -- the UI says so.
    """

    distance: float = 2.5
    metric: bool = True

    def describe(self) -> str:
        return f"平面占位深度（{self.distance:.1f} m）"

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = _image_shape(image)
        return np.full((height, width), float(self.distance), dtype=np.float32)


@dataclass
class ArrayDepthEstimator:
    """Wrap a precomputed depth map; used by tests and depth-map uploads."""

    depth: np.ndarray
    metric: bool = True
    name: str = "预置深度图"

    def describe(self) -> str:
        return self.name

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = _image_shape(image)
        depth = np.asarray(self.depth, dtype=np.float32)
        if depth.shape[:2] != (height, width):
            import cv2

            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        return depth


class HuggingFaceDepthEstimator:
    """Depth Anything V2 through `transformers`, loaded lazily.

    Metric checkpoints (the `-Metric-` family) return metres directly. Relative
    checkpoints return affine depth, which this class converts to a
    proportional depth map and leaves for `room_height_scale` to pin to a
    plausible room.
    """

    def __init__(self, model_dir: Path, *, device: str, relative: bool) -> None:
        self.model_dir = Path(model_dir)
        self.device_requested = device
        self.metric = not relative
        self._relative = relative
        self._processor = None
        self._model = None
        self._device = ""

    def describe(self) -> str:
        kind = "metric" if self.metric else "relative"
        return f"Depth Anything V2（{kind}）@ `{self.model_dir.name}`"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:  # pragma: no cover - dependency is in [webui]
            raise RuntimeError(
                "transformers is required for depth estimation; "
                "run `python -m pip install -e '.[webui]'`"
            ) from error
        device = resolve_device(self.device_requested)
        self._processor = AutoImageProcessor.from_pretrained(str(self.model_dir))
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(str(self.model_dir))
            .to(device)
            .eval()
        )
        self._device = device

    def estimate(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        import torch

        height, width = _image_shape(image)
        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            predicted = self._model(**inputs).predicted_depth
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        depth = predicted.squeeze().float().cpu().numpy()
        if self._relative:
            depth = relative_to_metric_scale(depth)
        return np.asarray(depth, dtype=np.float32)


def _image_shape(image: np.ndarray) -> tuple[int, int]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    return int(image.shape[0]), int(image.shape[1])


def resolve_device(requested: str) -> str:
    """Resolve `auto` to CUDA when it is available, else CPU."""

    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_depth_model_dir(config: WebUIConfig, on_log: LogFn | None = None) -> Path:
    """Return the local depth snapshot, downloading it if a run skipped bootstrap."""

    model_dir = config.depth_model_dir
    if (model_dir / "config.json").is_file():
        return model_dir
    from fire3d.webui.resources import download_depth_snapshot

    if on_log is not None:
        on_log(f"[depth] 本地缺少 {config.depth_model}，正在下载到 {model_dir}")
    return download_depth_snapshot(config.depth_model, model_dir, on_log or (lambda _: None))


def load_depth_estimator(
    config: WebUIConfig, on_log: LogFn | None = None
) -> DepthEstimator:
    """Load the configured monocular depth model.

    `FIRE3D_WEBUI_DEPTH_MODEL=none` (or `plane`) selects the flat placeholder so
    the WebUI can be exercised on a machine without the depth checkpoint.
    """

    if config.depth_is_placeholder:
        return FlatDepthEstimator()
    model_dir = resolve_depth_model_dir(config, on_log)
    estimator = HuggingFaceDepthEstimator(
        model_dir,
        device=config.depth_device,
        relative=not config.depth_metric,
    )
    if on_log is not None:
        on_log(f"[depth] 使用 {estimator.describe()}（设备 {resolve_device(config.depth_device)}）")
    return estimator


def relative_to_metric_scale(predicted: np.ndarray) -> np.ndarray:
    """Turn affine depth into a proportional depth map.

    The relative checkpoints are trained on inverse depth (bright is near), so
    depth is `1 / predicted`, rescaled so the median is 1.0 metre. The absolute
    scale is meaningless at this point by construction; `room_height_scale`
    fixes it from the reconstructed room height.
    """

    predicted = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(predicted) & (np.abs(predicted) > 1e-6)
    if not finite.any():
        return np.ones_like(predicted, dtype=np.float32)
    positives = predicted[finite] > 0.0
    depth = np.ones_like(predicted, dtype=np.float64)
    if positives.any():
        depth[finite] = 1.0 / np.abs(predicted[finite])
    else:
        depth[finite] = np.abs(predicted[finite])
    reference = float(np.median(depth[finite]))
    if not np.isfinite(reference) or reference <= 0.0:
        reference = 1.0
    return (depth / reference).astype(np.float32)


def intrinsics_from_fov(
    height: int, width: int, fov_degrees: float
) -> tuple[float, float, float, float]:
    """Square-pixel pinhole intrinsics from a horizontal field of view.

    The upload carries no EXIF-derived calibration and the protocol consumes
    only the cloud, so a nominal field of view is the honest input here.
    """

    if not 0.0 < fov_degrees < 180.0:
        raise ValueError("fov_degrees must be in (0, 180)")
    focal = 0.5 * float(width) / math.tan(math.radians(fov_degrees) / 2.0)
    return focal, focal, 0.5 * float(width), 0.5 * float(height)


def block_mean(depth: np.ndarray, grid_height: int, grid_width: int) -> np.ndarray:
    """NaN-aware 2x2 block mean onto the half-resolution lattice.

    The lattice the protocol consumes is exactly half the RGB resolution, and
    invalid samples are NaN. A plain resize would smear NaN across neighbours,
    and a plain stride would throw away half the measurements, so the block mean
    of the finite samples is used instead.
    """

    trimmed = depth[: grid_height * 2, : grid_width * 2]
    blocks = trimmed.reshape(grid_height, 2, grid_width, 2).astype(np.float64)
    finite = np.isfinite(blocks)
    counts = finite.sum(axis=(1, 3))
    totals = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    out = np.full((grid_height, grid_width), np.nan, dtype=np.float64)
    usable = counts > 0
    out[usable] = totals[usable] / counts[usable]
    return out


def back_project(
    depth: np.ndarray,
    *,
    fov_degrees: float,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Back-project a full-resolution depth map to a world-frame lattice.

    Returns a `(H//2, W//2, 3)` float32 array with NaN where the depth is
    missing or outside `[min_depth, max_depth]`.
    """

    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth map, got shape {depth.shape}")
    height, width = int(depth.shape[0]), int(depth.shape[1])
    grid_height, grid_width = height // 2, width // 2
    if grid_height < 16 or grid_width < 16:
        raise ValueError(
            f"image {width}x{height} is too small; the protocol needs at least a 32x32 input"
        )
    grid_depth = block_mean(np.asarray(depth, dtype=np.float32), grid_height, grid_width)
    valid = (
        np.isfinite(grid_depth)
        & (grid_depth >= float(min_depth))
        & (grid_depth <= float(max_depth))
    )
    grid_depth = np.where(valid, grid_depth, np.nan)

    focal_x, focal_y, center_x, center_y = intrinsics_from_fov(height, width, fov_degrees)
    # Pixel k spans [k, k+1), so its centre is k + 0.5. The lattice cell (i, j)
    # averages the 2x2 block at full-resolution (2j, 2i), whose centre is
    # (2j + 0.5, 2i + 0.5).
    columns = (2.0 * np.arange(grid_width, dtype=np.float64) + 0.5)[None, :]
    rows = (2.0 * np.arange(grid_height, dtype=np.float64) + 0.5)[:, None]
    camera_x = (columns - center_x) * grid_depth / focal_x
    camera_y = (rows - center_y) * grid_depth / focal_y

    points = np.empty((grid_height, grid_width, 3), dtype=np.float32)
    points[..., 0] = camera_x
    # +y forward, +z up: the released single_image frame reads the floor and
    # ceiling off z, and its camera looks along +y (see the module docstring).
    points[..., 1] = grid_depth
    points[..., 2] = -camera_y
    points[~valid] = np.nan
    return points


def room_height(points: np.ndarray) -> float:
    """Robust floor-to-ceiling spread of a world-frame cloud, in metres.

    The 1st/99th z percentiles ignore the specular outliers that depth models
    leave on mirrors and windows, which a min/max would take at face value.
    """

    finite = np.isfinite(points).all(axis=-1)
    heights = points[..., 2][finite]
    if heights.size < 2:
        return 0.0
    low, high = np.percentile(heights, 1.0), np.percentile(heights, 99.0)
    return float(high - low)


def room_height_scale(
    points: np.ndarray,
    *,
    target: float,
    min_height: float = MIN_ROOM_HEIGHT_M,
    max_height: float = MAX_ROOM_HEIGHT_M,
    always: bool = False,
    max_scale: float = 4.0,
) -> float:
    """Isotropic scale that brings the reconstructed room height into range.

    Applied about the camera at the origin, so it is exactly a rescaling of the
    depth map. `always` is set for relative-depth backends, whose absolute scale
    is undefined by construction.
    """

    finite = np.isfinite(points).all(axis=-1)
    if int(finite.sum()) < 256:
        return 1.0
    height = room_height(points)
    if not np.isfinite(height) or height <= 1e-6:
        return 1.0
    if not always and min_height <= height <= max_height:
        return 1.0
    return float(np.clip(float(target) / height, 1.0 / max_scale, max_scale))


def extent_scale(points: np.ndarray, *, limit: float = SCENE_EXTENT_LIMIT_M) -> float:
    """Shrink-only scale keeping the cloud inside the protocol's token range."""

    finite = np.isfinite(points).all(axis=-1)
    if finite.sum() < 256:
        return 1.0
    values = points[finite]
    low = np.percentile(values, 0.5, axis=0)
    high = np.percentile(values, 99.5, axis=0)
    longest = float(np.max(high - low))
    if not np.isfinite(longest) or longest <= limit:
        return 1.0
    return float(limit / longest)


def depth_preview(depth: np.ndarray) -> np.ndarray:
    """Colorized depth map for the UI; invalid samples stay black."""

    import cv2

    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0)
    preview = np.zeros(values.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(values[valid], [2.0, 98.0])
        if high <= low:
            high = low + 1e-6
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        # NaN depths stay NaN through the arithmetic, and casting NaN to uint8
        # is undefined, so the finite values are scaled and the rest default to 0.
        scaled = np.where(np.isfinite(normalized), normalized * 255.0, 0.0)
        preview = scaled.astype(np.uint8)
    colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def write_organized_ply(path: Path, points: np.ndarray) -> Path:
    """Write an `(H, W, 3)` lattice as a binary little-endian PLY.

    Vertex order is the lattice's row-major order and invalid samples stay NaN.
    Both matter: `utils.data_single_image.get_inference_data` asserts the vertex
    count equals `(H/2) * (W/2)` and filters NaN rows positionally, so dropping,
    reordering, or zero-filling vertices would silently bend the geometry.
    """

    points = np.asarray(points)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 lattice, got shape {points.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.ascontiguousarray(points.reshape(-1, 3), dtype="<f4")
    header = PLY_HEADER.format(count=vertices.shape[0]).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())
    return path
