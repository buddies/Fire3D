"""Run the frozen Fire3D `single_image` pipeline for one WebUI scene.

The WebUI never re-implements the pipeline. It shells out to `fire3d infer`,
which drives perception, reconstruction, and (optionally) rendering through the
frozen `fire3d_single_image_v1` protocol. That keeps a server-side result
byte-comparable with the released command line, and it means the WebUI inherits
the release's provenance recording for free: each run writes its resolved
protocol and the exact subprocess commands next to its outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from fire3d.webui.config import REPO_ROOT, WebUIConfig

LogFn = Callable[[str], None]

SCENE_GLB = "predicted_textured_world_scene.glb"
TRAINING_GLB = "predicted_textured_training_scene.glb"


@dataclass
class InferenceResult:
    """Everything one run produced, whether or not it succeeded."""

    scene_id: str
    returncode: int
    output_root: Path
    log_path: Path
    scene_glb: Path | None = None
    training_glb: Path | None = None
    object_meshes: list[Path] = field(default_factory=list)
    renders: list[Path] = field(default_factory=list)
    summary: dict | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.scene_glb is not None

    @property
    def downloadable_glb(self) -> Path | None:
        return self.scene_glb or self.training_glb

    def scene_status(self) -> str:
        scenes = (self.summary or {}).get("scenes") or {}
        record = scenes.get(self.scene_id) or {}
        return str(record.get("status", "unknown"))

    def to_markdown(self) -> str:
        lines = [f"- 返回码：`{self.returncode}`，场景状态：`{self.scene_status()}`"]
        if self.scene_glb is not None:
            size_mb = self.scene_glb.stat().st_size / (1024 * 1024)
            lines.append(f"- 场景 GLB：`{self.scene_glb.name}`（{size_mb:.1f} MB）")
        if self.training_glb is not None and self.scene_glb is None:
            lines.append(f"- 降级输出：`{self.training_glb.name}`（规范化坐标系）")
        if self.object_meshes:
            lines.append(f"- 逐物体网格：{len(self.object_meshes)} 个 `.ply`")
        if self.renders:
            lines.append(f"- 渲染预览：{len(self.renders)} 张")
        if not self.ok:
            lines.append("- 完整日志见输出目录下的 `logs/`")
        return "\n".join(lines)


def scene_output_root(config: WebUIConfig, scene_id: str) -> Path:
    return config.output_root / scene_id


def build_infer_command(
    config: WebUIConfig, scene_id: str, *, skip_render: bool | None = None
) -> list[str]:
    """The `fire3d infer` invocation for one generated scene.

    `python -m fire3d` is the released entry point, and `--data-root` must be
    the parent of the scene root because the CLI validates
    `<data_root>/single_image` and forwards it as the protocol input root.
    """

    skip = config.skip_render if skip_render is None else skip_render
    command = [
        sys.executable,
        "-m",
        "fire3d",
        "infer",
        "--dataset",
        "single_image",
        "--scene-id",
        scene_id,
        "--data-root",
        str(config.data_root),
        "--output-root",
        str(scene_output_root(config, scene_id)),
        "--gpu",
        str(config.gpu),
    ]
    if skip:
        command.append("--skip-render")
    return command


def subprocess_environment(config: WebUIConfig) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    environment["PYTHONUNBUFFERED"] = "1"
    # The reconstruction batches are large; expandable segments is what the
    # driver itself sets for its children, and it avoids a fragmentation OOM.
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("FF_SINGLE_IMAGE_ROOT", str(config.scene_root))
    return environment


def _stream(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    on_log: LogFn | None,
) -> int:
    """Run a child process, mirroring its output to the log file and the UI."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    emit: LogFn = on_log or (lambda _line: None)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            emit(line.rstrip("\n"))
        handle.flush()
        return process.wait()


def run_inference(
    config: WebUIConfig,
    scene_id: str,
    *,
    on_log: LogFn | None = None,
    skip_render: bool | None = None,
) -> InferenceResult:
    """Run perception + reconstruction (+ optional render) for one scene."""

    output_root = scene_output_root(config, scene_id)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / "webui_infer.log"
    command = build_infer_command(config, scene_id, skip_render=skip_render)
    emit: LogFn = on_log or (lambda _line: None)
    emit("[infer] " + " ".join(command))
    returncode = _stream(
        command,
        environment=subprocess_environment(config),
        log_path=log_path,
        on_log=on_log,
    )
    return collect_outputs(scene_id, output_root, returncode=returncode, log_path=log_path)


def find_scene_glbs(output_root: Path, scene_id: str) -> tuple[Path | None, Path | None]:
    """Locate the composed scene GLBs under a run's output root.

    The driver writes them under `reconstruction/<scene_id>/appearance/`, but the
    lookup is a glob so a layout change in the driver degrades to "found the GLB"
    rather than to a crash.
    """

    root = output_root / "reconstruction" / scene_id
    if not root.is_dir():
        root = output_root
    scene_glb = _first(root.rglob(SCENE_GLB))
    training_glb = _first(root.rglob(TRAINING_GLB))
    if scene_glb is None and training_glb is None:
        candidates = sorted(
            (
                path
                for path in root.rglob("*.glb")
                if path.is_file() and path.stat().st_size > 0
            ),
            key=lambda path: path.stat().st_size,
            reverse=True,
        )
        if candidates:
            scene_glb = candidates[0]
    return scene_glb, training_glb


def _first(paths: Iterator[Path]) -> Path | None:
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def collect_outputs(
    scene_id: str,
    output_root: Path,
    *,
    returncode: int,
    log_path: Path,
) -> InferenceResult:
    scene_glb, training_glb = find_scene_glbs(output_root, scene_id)
    renders_dir = output_root / "renders"
    renders = sorted(
        (
            path
            for path in renders_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        key=lambda path: path.name,
    )
    objects_dir = output_root / "reconstruction" / scene_id
    object_meshes = sorted(
        path
        for path in objects_dir.rglob("pred_*.ply")
        if path.is_file() and path.stat().st_size > 0
    ) if objects_dir.is_dir() else []
    return InferenceResult(
        scene_id=scene_id,
        returncode=int(returncode),
        output_root=output_root,
        log_path=log_path,
        scene_glb=scene_glb,
        training_glb=training_glb,
        object_meshes=object_meshes,
        renders=renders,
        summary=read_summary(output_root),
    )


def read_summary(output_root: Path) -> dict | None:
    path = output_root / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail_log(path: Path, *, lines: int = 40) -> str:
    """Last lines of a run log, for the UI's failure box."""

    if not path.is_file():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])
