"""Start-up resource provisioning for the Fire3D WebUI.

A fresh clone cannot reconstruct anything until four things exist: the Fire3D
model bundle, a DINOv3 source checkout, a monocular depth checkpoint, and the
compiled CUDA extensions. The first three are pure downloads and are fetched
automatically here; the fourth has to be compiled against the local CUDA
toolkit by `scripts/install.sh`, so this module verifies it and reports exactly
what is missing instead of pretending it can install it.

Checks are sentinel-based: every download step is skipped when its expected
files are already in place, so restarting the server is cheap and an interrupted
download is resumed rather than restarted.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fire3d.download import download_data, download_models
from fire3d.webui.config import (
    FALLBACK_DEPTH_MODEL,
    SCENE_DATA_DIR,
    WHITELIST_NAME,
    WebUIConfig,
    depth_dir_name,
)

LogFn = Callable[[str], None]

# Exactly the files `fire3d.cli.validate_release_inputs` refuses to run without,
# so the WebUI never reports "ready" for a bundle the pipeline will reject.
MODEL_SENTINELS = (
    "perception/config.yaml",
    "perception/model.pt",
    "reconstruction/flows/ss/config.yaml",
    "external/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "external/trellis2/shape_dec_next_dc_f16c32_fp16.safetensors",
    "external/trellis2/tex_dec_next_dc_f16c32_fp16.safetensors",
)

# Import names, not distribution names. The first group is required by the
# frozen inference path; the second only by the WebUI itself.
INFERENCE_MODULES = (
    "torch",
    "trellis2",
    "cumesh",
    "o_voxel",
    "flex_gemm",
    "nvdiffrast",
    "utils3d",
    "spconv",
)
WEBUI_MODULES = ("gradio", "transformers")

EXAMPLE_SCENE_IDS = ("003025",)


@dataclass
class BootstrapStep:
    name: str
    status: str
    detail: str

    OK_STATUSES = frozenset({"ready", "downloaded", "skipped", "warning"})

    @property
    def ok(self) -> bool:
        return self.status in self.OK_STATUSES

    def to_markdown(self) -> str:
        icon = {
            "ready": "✅",
            "downloaded": "⬇️",
            "skipped": "➖",
            "warning": "⚠️",
            "missing": "❌",
        }.get(self.status, "•")
        return f"{icon} **{self.name}** — {self.detail}"


@dataclass
class BootstrapReport:
    steps: list[BootstrapStep] = field(default_factory=list)
    depth_model: str = ""
    depth_model_dir: Path | None = None

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)

    @property
    def blockers(self) -> list[BootstrapStep]:
        return [step for step in self.steps if not step.ok]

    def add(self, step: BootstrapStep, on_log: LogFn) -> BootstrapStep:
        self.steps.append(step)
        on_log(step.to_markdown())
        return step

    def to_markdown(self) -> str:
        lines = [step.to_markdown() for step in self.steps]
        if not self.ok:
            lines.append("")
            lines.append(
                "**自动下载未能完成全部资源，请按上面 ❌ 的提示处理后重启服务。**"
            )
        return "\n\n".join(lines)


def _log(on_log: LogFn, message: str) -> None:
    on_log(message)


def model_bundle_ready(model_root: Path) -> bool:
    return all((model_root / name).is_file() for name in MODEL_SENTINELS)


def ensure_models(
    model_root: Path, on_log: LogFn, *, enabled: bool = True
) -> BootstrapStep:
    """Fetch the released Fire3D model bundle used by every frozen protocol."""

    if model_bundle_ready(model_root):
        return BootstrapStep(
            "Fire3D 模型权重",
            "ready",
            f"已存在：`{model_root}`",
        )
    if not enabled:
        return BootstrapStep(
            "Fire3D 模型权重",
            "missing",
            f"`{model_root}` 缺少模型文件，且已禁用自动下载（`--no-bootstrap`）。",
        )
    _log(on_log, f"[bootstrap] 下载 Fire3D 模型权重到 {model_root} …")
    download_models(model_root)
    if not model_bundle_ready(model_root):
        raise RuntimeError(f"Downloaded model bundle at {model_root} is incomplete")
    return BootstrapStep(
        "Fire3D 模型权重",
        "downloaded",
        f"已下载到 `{model_root}`",
    )


def dinov3_ready(repo: Path) -> bool:
    return (repo / "hubconf.py").is_file() and (repo / "dinov3").is_dir()


def ensure_dinov3(
    repo: Path, commit: str, on_log: LogFn, *, enabled: bool = True
) -> BootstrapStep:
    """Clone the pinned DINOv3 source tree the hub loader imports.

    The weights come with the model bundle; what has to exist here is the source
    tree, because `torch.hub.load(repo, "dinov3_vitl16", source="local")` runs
    `hubconf.py` from it. No compilation is involved.
    """

    repo = repo.expanduser()
    if dinov3_ready(repo):
        return BootstrapStep("DINOv3 源码", "ready", f"已存在：`{repo}`")
    if not enabled:
        return BootstrapStep(
            "DINOv3 源码",
            "missing",
            f"`{repo}` 缺少 DINOv3 源码，且已禁用自动下载（`--no-bootstrap`）。",
        )
    if shutil.which("git") is None:
        raise RuntimeError(
            f"DINOv3 source is missing at {repo} and git is not on PATH; "
            "install git or clone it manually at " + commit
        )
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").is_dir():
        _log(on_log, f"[bootstrap] 克隆 DINOv3 到 {repo} …")
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/dinov3.git", str(repo)],
            check=True,
        )
    _log(on_log, f"[bootstrap] 切换到 DINOv3 修订 {commit[:12]} …")
    subprocess.run(["git", "-C", str(repo), "fetch", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", commit], check=True)
    if not dinov3_ready(repo):
        raise RuntimeError(f"DINOv3 checkout at {repo} is incomplete after checkout")
    return BootstrapStep(
        "DINOv3 源码",
        "downloaded",
        f"已克隆到 `{repo}`（{commit[:12]}）",
    )


def depth_snapshot_ready(model_dir: Path) -> bool:
    return (model_dir / "config.json").is_file()


def download_depth_snapshot(model_id: str, model_dir: Path, on_log: LogFn) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency is in [webui]
        raise RuntimeError(
            "huggingface_hub is required to fetch the depth model; "
            "run scripts/install.sh first"
        ) from error
    _log(on_log, f"[bootstrap] 下载深度模型 {model_id} 到 {model_dir} …")
    snapshot_download(repo_id=model_id, local_dir=str(model_dir))
    if not depth_snapshot_ready(model_dir):
        raise RuntimeError(f"Depth snapshot at {model_dir} has no config.json")
    return model_dir


def ensure_depth_model(
    config: WebUIConfig, on_log: LogFn, *, enabled: bool = True
) -> BootstrapStep:
    """Fetch the monocular depth checkpoint, falling back to the relative one.

    The metric-indoor checkpoint is preferred because the conditioning cloud has
    to be metric. If it cannot be fetched (older hub mirror, offline cache, a
    renamed repo) the relative checkpoint still works: `depth.py` rescales it
    from the reconstructed room height.
    """

    model_dir = config.depth_model_dir
    if config.depth_is_placeholder:
        return BootstrapStep(
            "单目深度模型",
            "skipped",
            f"`{config.depth_model}` 使用平面占位深度，无需下载（仅供链路验证）",
        )
    if depth_snapshot_ready(model_dir):
        return BootstrapStep(
            "单目深度模型",
            "ready",
            f"`{config.depth_model}` 已存在于 `{model_dir}`",
        )
    if not enabled:
        return BootstrapStep(
            "单目深度模型",
            "missing",
            f"`{model_dir}` 缺少深度模型，且已禁用自动下载（`--no-bootstrap`）。",
        )
    try:
        download_depth_snapshot(config.depth_model, model_dir, on_log)
    except Exception as error:  # noqa: BLE001 - fallback must catch hub errors
        if config.depth_model == FALLBACK_DEPTH_MODEL:
            raise
        _log(
            on_log,
            f"[bootstrap] {config.depth_model} 下载失败（{error}），"
            f"改用 {FALLBACK_DEPTH_MODEL}",
        )
        return BootstrapStep(
            "单目深度模型",
            "warning",
            f"`{config.depth_model}` 不可用（{error}）；"
            f"请改用 `{FALLBACK_DEPTH_MODEL}` 并配合房间高度标定。",
        )
    return BootstrapStep(
        "单目深度模型",
        "downloaded",
        f"`{config.depth_model}` 已下载到 `{model_dir}`",
    )


def ensure_scene_root(scene_root: Path, on_log: LogFn) -> BootstrapStep:
    """Create the `single_image` native-input layout the protocol insists on."""

    data_dir = scene_root / SCENE_DATA_DIR
    whitelist = scene_root / WHITELIST_NAME
    created = not scene_root.is_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    if not whitelist.is_file():
        whitelist.write_text("", encoding="utf-8")
    return BootstrapStep(
        "WebUI 场景目录",
        "downloaded" if created else "ready",
        f"`{scene_root}`（{WHITELIST_NAME} + {SCENE_DATA_DIR}/）",
    )


def read_whitelist(scene_root: Path) -> list[str]:
    path = scene_root / WHITELIST_NAME
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sync_scene_whitelist(scene_root: Path) -> list[str]:
    """Union the whitelist with the scene directories that actually exist.

    `utils.data_single_image.load_single_image_data` intersects the whitelist
    with the directory listing, so an over-broad whitelist is harmless while a
    narrow one silently hides scenes. Taking the union therefore makes the file
    self-healing: a released example archive can overwrite it, and the next
    registration restores every scene that is still on disk.
    """

    data_dir = scene_root / SCENE_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    on_disk = sorted(child.name for child in data_dir.iterdir() if child.is_dir())
    entries = sorted(set(read_whitelist(scene_root)) | set(on_disk))
    (scene_root / WHITELIST_NAME).write_text(
        "".join(f"{entry}\n" for entry in entries), encoding="utf-8"
    )
    return entries


def ensure_example_data(
    config: WebUIConfig, on_log: LogFn, *, enabled: bool
) -> BootstrapStep:
    """Optionally install the released single-image example beside our scenes."""

    if not enabled:
        return BootstrapStep("官方单图示例数据", "skipped", "未启用（`--example-data` 可开启）")
    _log(on_log, f"[bootstrap] 下载 single_image 官方示例 {EXAMPLE_SCENE_IDS} …")
    download_data(
        config.data_root,
        ["single_image"],
        scene_ids=list(EXAMPLE_SCENE_IDS),
    )
    sync_scene_whitelist(config.scene_root)
    return BootstrapStep(
        "官方单图示例数据",
        "downloaded",
        f"{', '.join(EXAMPLE_SCENE_IDS)} → `{config.scene_root}`",
    )


def blender_available() -> bool:
    """Blender is only needed by the optional static render preview."""

    if shutil.which("blender") is not None:
        return True
    return _module_available("bpy")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def verify_runtime(on_log: LogFn) -> list[BootstrapStep]:
    """Report the compiled CUDA extensions and WebUI dependencies."""

    steps: list[BootstrapStep] = []
    missing_inference = [name for name in INFERENCE_MODULES if not _module_available(name)]
    if missing_inference:
        steps.append(
            BootstrapStep(
                "推理运行时",
                "missing",
                f"缺少 {', '.join(missing_inference)}；"
                "请在仓库根目录运行 `bash scripts/install.sh`（需要 CUDA 12.8 与 C++ 编译器）。",
            )
        )
    else:
        steps.append(BootstrapStep("推理运行时", "ready", "全部 CUDA 扩展可导入"))

    missing_webui = [name for name in WEBUI_MODULES if not _module_available(name)]
    if missing_webui:
        steps.append(
            BootstrapStep(
                "WebUI 依赖",
                "missing",
                f"缺少 {', '.join(missing_webui)}；"
                "运行 `python -m pip install -e '.[webui]'`。",
            )
        )
    else:
        steps.append(BootstrapStep("WebUI 依赖", "ready", "gradio 与 transformers 可用"))

    if _module_available("torch"):
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            steps.append(BootstrapStep("GPU", "ready", f"{name}（CUDA {torch.version.cuda}）"))
        else:
            steps.append(
                BootstrapStep("GPU", "warning", "未检测到 CUDA 设备，推理会极慢或失败")
            )
    for step in steps:
        _log(on_log, step.to_markdown())
    return steps


def prune_scenes(
    scene_root: Path, keep: int, *, prefix: str, on_log: LogFn
) -> list[str]:
    """Drop the oldest WebUI scenes, never a released example scene.

    Only directories carrying the WebUI prefix are candidates, so a downloaded
    example scene or a hand-placed scene is never removed by retention.
    """

    data_dir = scene_root / SCENE_DATA_DIR
    if not data_dir.is_dir():
        return []
    candidates = sorted(
        (child for child in data_dir.iterdir() if child.is_dir() and child.name.startswith(prefix)),
        key=lambda child: child.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for child in candidates[keep:]:
        shutil.rmtree(child, ignore_errors=True)
        removed.append(child.name)
    for scene_id in removed:
        _log(on_log, f"[retention] 清理旧场景 {scene_id}")
    if removed:
        sync_scene_whitelist(scene_root)
    return removed


def bootstrap(
    config: WebUIConfig,
    *,
    on_log: LogFn | None = None,
    download: bool = True,
) -> BootstrapReport:
    """Provision every resource the first inference needs.

    Raises on a failure that cannot be degraded -- a missing model bundle or a
    missing DINOv3 checkout -- because starting the server anyway would only
    move the same error to the first user request.
    """

    log: LogFn = on_log or (lambda message: print(message, flush=True))
    report = BootstrapReport()
    report.depth_model = config.depth_model
    report.depth_model_dir = config.depth_model_dir
    config.scene_root.mkdir(parents=True, exist_ok=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.cache_root.mkdir(parents=True, exist_ok=True)

    report.add(ensure_scene_root(config.scene_root, log), log)
    report.add(ensure_models(config.model_root, log, enabled=download), log)
    report.add(ensure_dinov3(config.dino_repo, config.dino_commit, log, enabled=download), log)
    depth_step = ensure_depth_model(config, log, enabled=download)
    if depth_step.status != "warning":
        report.add(depth_step, log)
    else:
        # The metric checkpoint is unavailable: retry with the relative one,
        # which `depth.py` rescales from the reconstructed room height. A total
        # failure here is reported rather than raised, so the operator can still
        # reach the UI and read `outputs/webui/bootstrap.json`; `--depth-model
        # none` remains a working (flat-depth) way to exercise the pipeline.
        report.depth_model = FALLBACK_DEPTH_MODEL
        report.depth_model_dir = (
            config.cache_root / "depth" / depth_dir_name(FALLBACK_DEPTH_MODEL)
        )
        log(f"[bootstrap] 回退到 {FALLBACK_DEPTH_MODEL}")
        try:
            fallback_step = ensure_depth_model(
                config.replace(
                    depth_model=report.depth_model,
                    depth_model_dir=report.depth_model_dir,
                ),
                log,
                enabled=download,
            )
        except Exception as error:  # noqa: BLE001 - reported through the UI
            fallback_step = BootstrapStep(
                "单目深度模型",
                "missing",
                f"`{FALLBACK_DEPTH_MODEL}` 也不可下载（{error}）；"
                "可先用 `--depth-model none` 验证链路。",
            )
        report.add(fallback_step, log)
    report.add(
        BootstrapStep(
            "场景清单",
            "ready",
            f"{len(sync_scene_whitelist(config.scene_root))} 个场景已登记",
        ),
        log,
    )
    report.add(ensure_example_data(config, log, enabled=config.example_data), log)
    if config.render_preview:
        report.add(
            BootstrapStep(
                "Blender 渲染",
                "ready" if blender_available() else "warning",
                (
                    "已检测到 Blender"
                    if blender_available()
                    else "未找到 blender/bpy；渲染预览会失败，可运行 "
                    "`bash scripts/install_blender.sh` 或关闭渲染预览。"
                ),
            ),
            log,
        )
    for step in verify_runtime(log):
        report.steps.append(step)
    return report


def write_bootstrap_report(report: BootstrapReport, path: Path) -> Path:
    """Persist the report so a failed container start can be diagnosed offline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ok": report.ok,
                "depth_model": report.depth_model,
                "depth_model_dir": (
                    str(report.depth_model_dir) if report.depth_model_dir else None
                ),
                "steps": [
                    {"name": step.name, "status": step.status, "detail": step.detail}
                    for step in report.steps
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "BootstrapReport",
    "BootstrapStep",
    "SCENE_DATA_DIR",
    "WHITELIST_NAME",
    "blender_available",
    "bootstrap",
    "ensure_depth_model",
    "ensure_dinov3",
    "ensure_example_data",
    "ensure_models",
    "ensure_scene_root",
    "model_bundle_ready",
    "prune_scenes",
    "read_whitelist",
    "sync_scene_whitelist",
    "verify_runtime",
    "write_bootstrap_report",
]
