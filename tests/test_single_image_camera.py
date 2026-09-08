import json

from utils import data_single_image


def test_published_single_image_camera_is_self_contained(tmp_path, monkeypatch):
    root = tmp_path / "single_image"
    camera_path = root / "data/003025/camera.json"
    camera_path.parent.mkdir(parents=True)
    expected = {
        "frame": {"eye": [0, 0, 0], "lookat": [0, 0, 1], "up": [0, 1, 0]},
        "forward": [0, 0, 1],
        "K": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
        "width": 100,
        "height": 80,
        "crop_top": 0,
        "crop_left": 0,
        "source_size": [80, 100],
    }
    camera_path.write_text(
        json.dumps(
            {
                "schema": "fire3d_single_image_camera_v1",
                "scene_id": "003025",
                "native": expected,
                "reconstruction": expected,
            }
        )
    )
    monkeypatch.setenv("FF_SINGLE_IMAGE_ROOT", str(root))
    assert data_single_image.scene_camera("003025", native_resolution=True) == expected
