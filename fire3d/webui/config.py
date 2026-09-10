"""Paths and runtime settings for the Fire3D WebUI.

Every path the WebUI writes to lives under the repository or under a root the
operator can override, and every setting can be supplied either by an
environment variable or by a CLI flag. Keeping this in one frozen dataclass
means the Gradio app, the bootstrap, and the runner cannot disagree about where
things are -- which matters because the pipeline resolves its own input root
from `FF_SINGLE_IMAGE_ROOT` while the WebUI writes the scene files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# `fire3d download --models` installs the bundle here, and the frozen protocols
# resolve their checkpoints relative to it.
DEFAULT_MODEL_ROOT = REPO_ROOT / "checkpoints/Fire3D"

# DINOv3 has to be a source checkout: the perception and flow models load it
# through `torch.hub.load(repo_dir, "dinov3_vitl16", source="local")`, which
# needs hubconf.py next to the package. The revision matches scripts/install.sh.
DEFAULT_DINOV3_REPO = REPO_ROOT / "third_party/dinov3"
DEFAULT_DINOV3_COMMIT = "31703e4cbf1ccb7c4a72daa1350405f86754b6d1"

# The WebUI keeps its generated scenes in its own `single_image` root so that
# uploaded images never mix with a published dataset download.
DEFAULT_WORK_ROOT = REPO_ROOT / "data/webui"
DEFAULT_SCENE_ROOT = DEFAULT_WORK_ROOT / "single_image"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/webui"
DEFAULT_CACHE_ROOT = REPO_ROOT / ".cache/fire3d_webui"

# Metric-indoor Depth Anything V2 is the right prior here: the release targets
# indoor rooms, the conditioning cloud has to be metric (the frozen protocol
# uses `scene_scale` 24.0 metres and the wall/room-box fit reads metres), and
# the metric checkpoint needs no affine calibration. The relative checkpoint is
# the fallback when the metric one is unavailable.
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
FALLBACK_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# 60 degrees horizontal is a neutral guess for a phone or webcam frame. The
# reconstruction only sees the cloud, so the field of view trades the scene's
# proportions against its depth extent; wrong values still produce a scene, just
# a differently proportioned one.
DEFAULT_FOV_DEGREES = 60.0

# Image long side cap. The release examples are ~970x1296 RGB, and the
# perception cost grows with the point count, so this keeps an upload in the
# same regime as the validated scenes.
DEFAULT_MAX_SIDE = 1280
SCENE_ID_PREFIX = "webui"

# The `single_image` native-input layout the frozen protocol demands, shared by
# the code that provisions the root and the code that writes scenes into it.
WHITELIST_NAME = "single_image_valid.txt"
SCENE_DATA_DIR = "data"


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off", ""}


def _env_path(name: str, default: Path) -> Path:
    raw = _env_str(name)
    return default if raw is None else Path(raw).expanduser()


def _parse_auth(raw: str | None) -> tuple[str, str] | None:
    if raw is None or not raw.strip():
        return None
    user, separator, password = raw.partition(":")
    if not separator or not user or not password:
        raise ValueError("auth must be supplied as 'user:password'")
    return (user, password)


def depth_dir_name(model_id: str) -> str:
    """Filesystem-safe directory name for a Hugging Face model id."""

    return model_id.replace("/", "__")


@dataclass(frozen=True)
class WebUIConfig:
    """Resolved WebUI settings. Build one with :meth:`from_env`."""

    model_root: Path
    scene_root: Path
    output_root: Path
    cache_root: Path
    dino_repo: Path
    dino_commit: str
    depth_model: str
    depth_model_dir: Path
    depth_device: str
    gpu: str
    render_preview: bool
    max_side: int
    fov_degrees: float
    max_depth: float
    min_depth: float
    calibrate_room_height: bool
    target_room_height: float
    keep_scenes: int
    concurrency: int
    host: str
    port: int
    share: bool
    auth: tuple[str, str] | None
    bootstrap: bool
    example_data: bool

    @property
    def data_root(self) -> Path:
        """`--data-root` for `fire3d infer`.

        The CLI validates `<data_root>/single_image` and then hands that same
        path to the driver as `--protocol-input-root`, so the data root has to
        be the parent of the scene root; deriving it keeps the two in lockstep.
        """

        return self.scene_root.parent

    @property
    def depth_is_placeholder(self) -> bool:
        """True for the flat-depth backend, which needs no checkpoint at all."""

        return self.depth_model.lower() in {"none", "plane", "flat"}

    @property
    def depth_metric(self) -> bool:
        """True when the depth checkpoint returns metres, not affine depth."""

        return "Metric" in self.depth_model

    @property
    def skip_render(self) -> bool:
        """Rendering needs Blender, so it stays opt-in."""

        return not self.render_preview

    def replace(self, **overrides: object) -> WebUIConfig:
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean)

    @classmethod
    def from_env(cls, **overrides: object) -> WebUIConfig:
        """Build a config from `FIRE3D_WEBUI_*` variables plus CLI overrides.

        `None` overrides mean "not supplied on the command line", so they fall
        through to the environment and then to the built-in default.
        """

        def pick(name: str, default: object) -> object:
            value = overrides.get(name)
            return default if value is None else value

        cache_root = _env_path("FIRE3D_WEBUI_CACHE_ROOT", DEFAULT_CACHE_ROOT)
        depth_model = str(
            pick(
                "depth_model",
                _env_str("FIRE3D_WEBUI_DEPTH_MODEL", DEFAULT_DEPTH_MODEL),
            )
        )
        depth_model_dir = Path(
            str(
                pick(
                    "depth_model_dir",
                    cache_root / "depth" / depth_dir_name(depth_model),
                )
            )
        )
        auth = pick("auth", None)
        base = cls(
            model_root=_env_path("FIRE3D_MODEL_ROOT", DEFAULT_MODEL_ROOT),
            scene_root=_env_path("FIRE3D_WEBUI_SCENE_ROOT", DEFAULT_SCENE_ROOT),
            output_root=_env_path("FIRE3D_WEBUI_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT),
            cache_root=cache_root,
            dino_repo=_env_path(
                "FIRE3D_DINOV3_REPO",
                _env_path("FIRE3D_WEBUI_DINO_REPO", DEFAULT_DINOV3_REPO),
            ),
            dino_commit=str(
                _env_str("FIRE3D_WEBUI_DINO_COMMIT", DEFAULT_DINOV3_COMMIT)
            ),
            depth_model=depth_model,
            depth_model_dir=depth_model_dir,
            depth_device=str(
                pick("depth_device", _env_str("FIRE3D_WEBUI_DEPTH_DEVICE", "auto"))
            ),
            gpu=str(pick("gpu", _env_str("FIRE3D_WEBUI_GPU", "0"))),
            render_preview=bool(
                pick(
                    "render_preview",
                    _env_bool("FIRE3D_WEBUI_RENDER_PREVIEW", False),
                )
            ),
            max_side=int(
                pick("max_side", _env_int("FIRE3D_WEBUI_MAX_SIDE", DEFAULT_MAX_SIDE))
            ),
            fov_degrees=float(
                pick("fov_degrees", _env_float("FIRE3D_WEBUI_FOV", DEFAULT_FOV_DEGREES))
            ),
            max_depth=float(_env_float("FIRE3D_WEBUI_MAX_DEPTH", 12.0)),
            min_depth=float(_env_float("FIRE3D_WEBUI_MIN_DEPTH", 0.05)),
            calibrate_room_height=bool(
                _env_bool("FIRE3D_WEBUI_CALIBRATE_ROOM_HEIGHT", True)
            ),
            target_room_height=float(
                _env_float("FIRE3D_WEBUI_TARGET_ROOM_HEIGHT", 2.6)
            ),
            keep_scenes=int(
                pick("keep_scenes", _env_int("FIRE3D_WEBUI_KEEP_SCENES", 50))
            ),
            concurrency=int(
                pick("concurrency", _env_int("FIRE3D_WEBUI_CONCURRENCY", 1))
            ),
            host=str(pick("host", _env_str("FIRE3D_WEBUI_HOST", "0.0.0.0"))),
            port=int(pick("port", _env_int("FIRE3D_WEBUI_PORT", 7860))),
            share=bool(pick("share", _env_bool("FIRE3D_WEBUI_SHARE", False))),
            auth=(
                _parse_auth(str(auth))
                if auth
                else _parse_auth(_env_str("FIRE3D_WEBUI_AUTH"))
            ),
            bootstrap=bool(
                pick("bootstrap", _env_bool("FIRE3D_WEBUI_BOOTSTRAP", True))
            ),
            example_data=bool(
                pick("example_data", _env_bool("FIRE3D_WEBUI_EXAMPLE_DATA", False))
            ),
        )
        if base.max_side < 64:
            raise ValueError("max_side must be at least 64 pixels")
        if not 5.0 <= base.fov_degrees <= 170.0:
            raise ValueError("fov_degrees must be between 5 and 170 degrees")
        if base.min_depth <= 0.0 or base.max_depth <= base.min_depth:
            raise ValueError("max_depth must be greater than min_depth")
        if base.keep_scenes < 1:
            raise ValueError("keep_scenes must be at least 1")
        if base.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        return base
