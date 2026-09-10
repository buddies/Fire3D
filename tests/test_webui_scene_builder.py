"""Contract tests for the WebUI's single-image scene builder.

These run without torch, transformers, or a GPU: the depth estimator is
injected, so what is under test is the scene contract the frozen protocol
consumes -- the whitelist, the organized lattice, and `camera.json`.
"""

import json

import numpy as np
import pytest

from fire3d.webui import scene_builder
from fire3d.webui.config import SCENE_DATA_DIR, WHITELIST_NAME, WebUIConfig
from fire3d.webui.depth import (
    ArrayDepthEstimator,
    FlatDepthEstimator,
    back_project,
    block_mean,
    depth_preview,
    relative_to_metric_scale,
    room_height_scale,
    write_organized_ply,
)

HEIGHT = 96
WIDTH = 128


def make_config(tmp_path, **overrides) -> WebUIConfig:
    config = WebUIConfig.from_env(depth_model="plane")
    return config.replace(
        scene_root=tmp_path / "webui/single_image",
        output_root=tmp_path / "out",
        cache_root=tmp_path / "cache",
        **overrides,
    )


def synthetic_image(height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    columns = np.linspace(0, 255, width, dtype=np.uint8)
    rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    return np.stack(
        [
            np.broadcast_to(columns, (height, width)),
            np.broadcast_to(rows, (height, width)),
            np.full((height, width), 128, dtype=np.uint8),
        ],
        axis=-1,
    )


def test_fit_image_size_snaps_to_the_lattice_multiple():
    height, width = scene_builder.fit_image_size(1000, 1500, max_side=1280)
    assert height % scene_builder.IMAGE_MULTIPLE == 0
    assert width % scene_builder.IMAGE_MULTIPLE == 0
    assert max(height, width) <= 1280


def test_fit_image_size_never_upscales_a_small_upload():
    assert scene_builder.fit_image_size(96, 128, max_side=1280) == (96, 128)


def test_fit_image_size_rejects_a_degenerate_aspect_ratio():
    with pytest.raises(ValueError, match="too extreme"):
        scene_builder.fit_image_size(8, 4000, max_side=1280)


def test_camera_payload_is_readable_by_the_released_loader(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    scene_id = "webui_test_camera"
    layout = scene_builder.scene_layout(config.scene_root, scene_id)
    layout.image_dir.mkdir(parents=True)
    payload = scene_builder.camera_payload(
        scene_id, HEIGHT, WIDTH, config.fov_degrees
    )
    layout.camera_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("FF_SINGLE_IMAGE_ROOT", str(config.scene_root))

    from utils.data_single_image import scene_camera

    native = scene_camera(scene_id, native_resolution=True)
    assert native["frame"]["eye"] == [0.0, 0.0, 0.0]
    assert native["frame"]["lookat"] == [0.0, 1.0, 0.0]
    assert native["frame"]["up"] == [0.0, 0.0, 1.0]
    assert (native["width"], native["height"]) == (WIDTH, HEIGHT)
    assert native["K"][0][0] == pytest.approx(
        0.5 * WIDTH / np.tan(np.radians(config.fov_degrees) / 2.0)
    )
    # The lattice is a multiple of 16 on both sides, so the driver's centre crop
    # is a no-op and the reconstruction camera must equal the native one.
    assert scene_camera(scene_id)["width"] == WIDTH
    assert scene_camera(scene_id)["K"] == native["K"]
    assert scene_camera(scene_id)["crop_top"] == 0
    assert scene_camera(scene_id)["crop_left"] == 0


def test_write_organized_ply_preserves_order_count_and_nan(tmp_path):
    import trimesh

    points = np.zeros((4, 5, 3), dtype=np.float32)
    points[..., 1] = 2.5
    points[..., 0] = np.arange(5)[None, :]
    points[..., 2] = np.arange(4)[:, None]
    points[0, 0] = np.nan

    path = write_organized_ply(tmp_path / "aligned_pcd.ply", points)
    loaded = np.asarray(trimesh.load(path, process=False).vertices)

    assert loaded.shape == (20, 3)
    assert np.isnan(loaded[0]).all()
    assert loaded[1].tolist() == pytest.approx([1.0, 2.5, 0.0])
    # Row-major order is what the loader's positional masking depends on.
    assert loaded[5].tolist() == pytest.approx([0.0, 2.5, 1.0])


def test_back_project_uses_a_z_up_camera_at_the_origin():
    depth = np.full((HEIGHT, WIDTH), 4.0, dtype=np.float32)
    points = back_project(
        depth, fov_degrees=60.0, min_depth=0.1, max_depth=20.0
    )
    assert points.shape == (HEIGHT // 2, WIDTH // 2, 3)
    top_left = points[0, 0]
    bottom_right = points[-1, -1]
    # +y is forward, so every ray sits at the measured depth.
    assert top_left[1] == pytest.approx(4.0)
    assert bottom_right[1] == pytest.approx(4.0)
    # +z is up and +x is right, matching the released single_image frame.
    assert top_left[2] > 0.0 and bottom_right[2] < 0.0
    assert top_left[0] < 0.0 and bottom_right[0] > 0.0


def test_back_project_marks_out_of_range_depth_as_nan():
    depth = np.full((HEIGHT, WIDTH), 4.0, dtype=np.float32)
    depth[:, : WIDTH // 2] = 50.0
    points = back_project(depth, fov_degrees=60.0, min_depth=0.1, max_depth=12.0)
    assert np.isnan(points[:, : WIDTH // 4]).all()
    assert np.isfinite(points[:, WIDTH // 4 :]).all()


def test_block_mean_averages_only_the_finite_samples():
    all_invalid = np.array(
        [[1.0, 3.0, np.nan, np.nan], [1.0, 3.0, np.nan, np.nan]], dtype=np.float32
    )
    averaged = block_mean(all_invalid, 1, 2)
    assert averaged[0, 0] == pytest.approx(2.0)
    assert np.isnan(averaged[0, 1])

    uniform = np.full((2, 2), 5.0, dtype=np.float32)
    assert block_mean(uniform, 1, 1)[0, 0] == pytest.approx(5.0)


def test_relative_depth_is_rescaled_to_a_proportional_map():
    predicted = np.array([[0.25, 0.5, 1.0]], dtype=np.float32)
    scaled = relative_to_metric_scale(predicted)
    # 1 / predicted, then normalised so the median is 1.0: the absolute scale of
    # a relative checkpoint is meaningless, only the ratios are.
    assert scaled[0].tolist() == pytest.approx([2.0, 1.0, 0.5])


def test_room_height_scale_corrects_heights_outside_the_plausible_band():
    grid = np.zeros((32, 32, 3), dtype=np.float32)
    grid[..., 1] = 3.0  # forward
    grid[..., 2] = np.linspace(0.0, 2.4, 32)[:, None]
    assert room_height_scale(grid, target=2.6) == 1.0

    # Too short (a collapsed relative-depth cloud) and too tall both indicate a
    # broken scale, so both are rescaled -- clamped to max_scale either way.
    squashed = grid.copy()
    squashed[..., 2] *= 0.1
    assert room_height_scale(squashed, target=2.6) == pytest.approx(4.0)

    stretched = grid.copy()
    stretched[..., 2] *= 5.0
    assert room_height_scale(stretched, target=2.6) == pytest.approx(0.25)

    # `always` forces the correction even for a plausible-looking height, which
    # is what a relative checkpoint needs.
    assert room_height_scale(grid, target=1.2, always=True) == pytest.approx(0.5)


def test_build_scene_writes_the_protocol_layout(tmp_path):
    config = make_config(tmp_path)
    summary = scene_builder.build_scene(
        config, FlatDepthEstimator(distance=3.0), synthetic_image()
    )

    assert summary.layout.rgb_path.is_file()
    assert summary.layout.point_cloud_path.is_file()
    assert summary.layout.camera_path.is_file()
    assert summary.layout.preview_path.is_file()

    assert (summary.height, summary.width) == (HEIGHT, WIDTH)
    assert summary.point_count == (HEIGHT // 2) * (WIDTH // 2)
    assert summary.valid_points == summary.point_count
    assert summary.scale == 1.0

    import trimesh

    loaded = trimesh.load(summary.layout.point_cloud_path, process=False)
    assert np.asarray(loaded.vertices).shape == (summary.point_count, 3)

    import cv2

    reloaded = cv2.imread(str(summary.layout.rgb_path), cv2.IMREAD_COLOR)
    assert reloaded.shape[:2] == (HEIGHT, WIDTH)

    whitelist = (config.scene_root / WHITELIST_NAME).read_text().splitlines()
    assert summary.scene_id in whitelist
    assert summary.scene_id.startswith("webui_")


def test_build_scene_registers_the_scene_only_once(tmp_path):
    config = make_config(tmp_path)
    first = scene_builder.build_scene(
        config, FlatDepthEstimator(), synthetic_image()
    )
    scene_builder.build_scene(
        config,
        FlatDepthEstimator(),
        synthetic_image(),
        scene_id=first.scene_id,
    )
    whitelist = (config.scene_root / WHITELIST_NAME).read_text().splitlines()
    assert whitelist.count(first.scene_id) == 1


def test_build_scene_keeps_whitelist_entries_for_scenes_on_disk(tmp_path):
    config = make_config(tmp_path)
    released = config.scene_root / SCENE_DATA_DIR / "003025"
    released.mkdir(parents=True)
    # A downloaded release archive can ship its own whitelist; the sync must
    # union rather than truncate, or our generated scene would vanish.
    (config.scene_root / WHITELIST_NAME).write_text("003025\n")

    summary = scene_builder.build_scene(
        config, FlatDepthEstimator(), synthetic_image()
    )
    whitelist = (config.scene_root / WHITELIST_NAME).read_text().splitlines()
    assert whitelist == sorted({"003025", summary.scene_id})


def test_build_scene_rejects_an_empty_depth_map(tmp_path):
    config = make_config(tmp_path)
    estimator = ArrayDepthEstimator(np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="有效点"):
        scene_builder.build_scene(config, estimator, synthetic_image())


def test_depth_preview_blackens_invalid_samples():
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    depth[0, 0] = np.nan
    preview = depth_preview(depth)
    assert preview.shape == (8, 8, 3)
    assert preview[0, 0].tolist() == [0, 0, 0]
    assert preview[4, 4].sum() > 0


def test_released_loader_accepts_a_generated_scene(tmp_path, monkeypatch):
    """The strongest offline check: the frozen perception loader reads our scene.

    If the lattice shape, vertex order, or resolution multiple drifts, this is
    where it fails -- before any GPU is involved.
    """

    config = make_config(tmp_path)
    summary = scene_builder.build_scene(
        config, FlatDepthEstimator(distance=3.0), synthetic_image()
    )
    monkeypatch.setenv("FF_SINGLE_IMAGE_ROOT", str(config.scene_root))

    from utils.data_single_image import get_inference_data, load_single_image_data

    assert load_single_image_data() == [summary.scene_id]
    payload = get_inference_data([summary.scene_id], 0, image_downsample=16)

    assert payload["rgbs"].shape == (1, HEIGHT, WIDTH, 3)
    expected_points = (HEIGHT // 16) * (WIDTH // 16)
    assert payload["points"].shape == (expected_points, 3)
    assert payload["points_rgbs"].shape == (expected_points, 3)
    assert payload["preprocess_transform"].shape == (4, 4)
    assert np.isfinite(payload["points"]).all()
    # `point_normalize` only translates, so the metric scale we produced has to
    # survive it. A flat-depth cloud collapses the forward axis but must still
    # span a room-height vertical extent (~2.5 m for a 3 m wall at 60 degrees).
    vertical = payload["points"][:, 2]
    assert float(vertical.max() - vertical.min()) > 1.0
