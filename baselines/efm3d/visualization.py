"""Shared point-cloud and OBB visualization helpers for EFM3D adapters."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


EDGE_VERTEX_IDS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def load_cloud(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    cloud = trimesh.load(path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float32)
    colors = np.asarray(cloud.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    if len(points) > max_points:
        indices = np.random.default_rng(0).choice(
            len(points), size=max_points, replace=False
        )
        points = points[indices]
        colors = colors[indices]
    return points, colors


def camera_basis(frame: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray(frame["eye"], dtype=np.float64)
    lookat = np.asarray(frame["lookat"], dtype=np.float64)
    up_hint = np.asarray(frame["up"], dtype=np.float64)
    forward = lookat - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    return right, up, forward


def effective_pinhole(
    intrinsics: list[list[float]],
    source_width: int,
    source_height: int,
    render_width: int,
    render_height: int,
) -> tuple[float, float, float, float]:
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    if min(fx, fy, source_width, source_height, render_width, render_height) <= 0:
        raise ValueError("Camera focal lengths and dimensions must be positive")
    focal = (
        fy * float(render_height) / float(source_height)
        if fy >= fx
        else fx * float(render_width) / float(source_width)
    )
    shift_x = (float(source_width) * 0.5 - cx) / float(source_width)
    shift_y = (cy - float(source_height) * 0.5) / float(source_height)
    return (
        focal,
        focal,
        float(render_width) * (0.5 - shift_x),
        float(render_height) * (0.5 + shift_y),
    )


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    frame: dict[str, object],
    pinhole: tuple[float, float, float, float],
    width: int,
    height: int,
    point_size: float,
) -> tuple[Image.Image, int]:
    eye = np.asarray(frame["eye"], dtype=np.float64)
    right, up, forward = camera_basis(frame)
    relative = np.asarray(points, dtype=np.float64) - eye[None, :]
    x = relative @ right
    y = relative @ up
    z = relative @ forward
    valid = np.isfinite(z) & (z > 1e-5)
    source_indices = np.flatnonzero(valid)
    image = Image.new("RGB", (width, height), (7, 9, 12))
    if not len(source_indices):
        return image, 0

    fx, fy, cx, cy = pinhole
    z_valid = z[valid]
    px = fx * x[valid] / z_valid + cx
    py = cy - fy * y[valid] / z_valid
    onscreen = (
        np.isfinite(px)
        & np.isfinite(py)
        & (px >= -point_size)
        & (px < float(width) + point_size)
        & (py >= -point_size)
        & (py < float(height) + point_size)
    )
    projected = np.stack([px[onscreen], py[onscreen]], axis=1)
    depth = z_valid[onscreen]
    source_indices = source_indices[onscreen]
    colors_u8 = np.clip(colors[source_indices] * 255.0, 0, 255).astype(np.uint8)
    draw = ImageDraw.Draw(image)
    radius = max(0.75, point_size * 0.5)
    for index in np.argsort(depth)[::-1]:
        point_x, point_y = projected[index]
        color = tuple(int(value) for value in colors_u8[index])
        draw.ellipse(
            (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
            fill=color,
        )
    return image, int(len(source_indices))


def _font(size: int, *, bold: bool) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ):
        path = root / filename
        if path.is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def add_label(
    image: Image.Image, title: str, subtitle: str, prediction: bool
) -> Image.Image:
    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = (140, 59, 34, 218) if prediction else (32, 36, 42, 218)
    draw.rectangle((0, 0, 390, 70), fill=color)
    draw.text((14, 8), title, fill="white", font=_font(24, bold=True))
    draw.text((14, 40), subtitle, fill=(228, 232, 238), font=_font(15, bold=False))
    return Image.alpha_composite(result, overlay).convert("RGB")


def assemble_sheet(
    panels: list[dict[str, Image.Image]], width: int, height: int, output: Path
) -> None:
    rows = math.ceil(len(panels) / 2)
    sheet = Image.new("RGB", (width * 4, height * rows), (7, 9, 12))
    for rank, panel in enumerate(panels):
        row = rank // 2
        pair_column = rank % 2
        sheet.paste(panel["gt"], (pair_column * width * 2, row * height))
        sheet.paste(panel["pred"], (pair_column * width * 2 + width, row * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def load_predictions(path: Path | None, threshold: float) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if float(row.get("prob", 1.0)) >= threshold
        ]


def _obb_corners_world(row: dict[str, str]) -> np.ndarray:
    quaternion = np.asarray(
        [
            float(row["qw_world_object"]),
            float(row["qx_world_object"]),
            float(row["qy_world_object"]),
            float(row["qz_world_object"]),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    else:
        quaternion /= norm
    w, x, y, z = quaternion
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    half_extent = 0.5 * np.asarray(
        [float(row["scale_x"]), float(row["scale_y"]), float(row["scale_z"])]
    )
    signs = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    center = np.asarray(
        [
            float(row["tx_world_object"]),
            float(row["ty_world_object"]),
            float(row["tz_world_object"]),
        ]
    )
    return (rotation @ (signs * half_extent).T).T + center


def _project_obb_edges(
    row: dict[str, str],
    frame: dict[str, object],
    pinhole: tuple[float, float, float, float],
) -> list[tuple[float, tuple[float, float], tuple[float, float]]]:
    corners = _obb_corners_world(row)
    eye = np.asarray(frame["eye"], dtype=np.float64)
    right, up, forward = camera_basis(frame)
    relative = corners - eye[None, :]
    camera_x = relative @ right
    camera_y = relative @ up
    camera_z = relative @ forward
    fx, fy, cx, cy = pinhole
    projected = np.full((len(corners), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera_z) & (camera_z > 1e-5)
    projected[valid, 0] = fx * camera_x[valid] / camera_z[valid] + cx
    projected[valid, 1] = cy - fy * camera_y[valid] / camera_z[valid]
    return [
        (
            float((camera_z[start] + camera_z[end]) * 0.5),
            tuple(float(value) for value in projected[start]),
            tuple(float(value) for value in projected[end]),
        )
        for start, end in EDGE_VERTEX_IDS
        if valid[start] and valid[end]
    ]


def overlay_efm_obbs(
    image: Image.Image,
    predictions: list[dict[str, str]],
    frame: dict[str, object],
    pinhole: tuple[float, float, float, float],
    line_width: int,
) -> tuple[Image.Image, int, int]:
    palette = (
        (0, 212, 255),
        (255, 92, 138),
        (141, 220, 74),
        (255, 181, 51),
        (183, 117, 255),
        (58, 224, 184),
    )
    rendered = image.copy()
    segments = []
    visible_boxes = 0
    for index, row in enumerate(predictions):
        box_segments = _project_obb_edges(row, frame, pinhole)
        visible_boxes += bool(box_segments)
        color_index = int(row.get("instance", index))
        segments.extend((*segment, palette[color_index % len(palette)]) for segment in box_segments)
    draw = ImageDraw.Draw(rendered)
    for _depth, start, end, color in sorted(segments, reverse=True):
        draw.line((start, end), fill=(5, 5, 5), width=line_width + 2)
        draw.line((start, end), fill=color, width=line_width)
    return rendered, visible_boxes, len(segments)


__all__ = [
    "add_label",
    "assemble_sheet",
    "effective_pinhole",
    "load_cloud",
    "load_predictions",
    "overlay_efm_obbs",
    "render_point_cloud",
]
