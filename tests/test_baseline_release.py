import numpy as np
from PIL import Image

from baselines.efm3d.visualization import (
    effective_pinhole,
    overlay_efm_obbs,
    render_point_cloud,
)


def test_efm3d_visualization_projects_points_and_oriented_boxes():
    frame = {
        "eye": [0.0, 0.0, 0.0],
        "lookat": [0.0, 0.0, 1.0],
        "up": [0.0, 1.0, 0.0],
    }
    pinhole = effective_pinhole(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        100,
        100,
        100,
        100,
    )
    image, visible_points = render_point_cloud(
        np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        frame,
        pinhole,
        100,
        100,
        2.0,
    )
    assert visible_points == 1
    assert image.getpixel((50, 50))[0] > 200

    prediction = {
        "tx_world_object": "0",
        "ty_world_object": "0",
        "tz_world_object": "2",
        "qw_world_object": "1",
        "qx_world_object": "0",
        "qy_world_object": "0",
        "qz_world_object": "0",
        "scale_x": "1",
        "scale_y": "1",
        "scale_z": "1",
        "instance": "0",
    }
    rendered, visible_boxes, visible_edges = overlay_efm_obbs(
        Image.new("RGB", (100, 100), "white"),
        [prediction],
        frame,
        pinhole,
        2,
    )
    assert rendered.size == (100, 100)
    assert visible_boxes == 1
    assert visible_edges == 12
