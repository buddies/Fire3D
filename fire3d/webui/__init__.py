"""Gradio web interface for single-image Fire3D reconstruction.

The WebUI is deliberately a *thin* layer on top of the released pipeline. It
turns one uploaded RGB image into a `single_image` protocol scene directory and
then runs `fire3d infer` unchanged, so every result is produced by the frozen
`fire3d_single_image_v1` protocol rather than by a parallel code path that could
drift from it.

Pipeline of the application:

1. `fire3d.webui.resources` downloads the model bundle, the DINOv3 source
   checkout, and the monocular depth checkpoint at start-up.
2. `fire3d.webui.depth` estimates metric depth for the uploaded image and
   back-projects it into the gravity-aligned world frame the protocol expects.
3. `fire3d.webui.scene_builder` writes `rgb.jpeg`, `aligned_pcd.ply`, and
   `camera.json` under the WebUI's own `single_image` root.
4. `fire3d.webui.runner` runs `fire3d infer --dataset single_image` and collects
   the composed scene GLB.
5. `fire3d.webui.app` renders all of that with Gradio.

Only `fire3d.webui.config` is imported here: importing the package must stay
cheap enough for the CLI help path, and must not require `gradio`, `torch`, or
`transformers`.
"""

from fire3d.webui.config import WebUIConfig

__all__ = ["WebUIConfig"]
