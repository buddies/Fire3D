"""Gradio front end: upload one RGB image, get a textured 3D scene.

The interface is intentionally thin. Every button press becomes:

    upload -> depth -> single_image scene -> `fire3d infer` -> GLB

so anything the WebUI produces can also be produced by hand from
`docs/webui.md`'s command line. Start-up provisioning lives in
`fire3d.webui.resources`, the scene contract in `fire3d.webui.scene_builder`,
and the pipeline invocation in `fire3d.webui.runner`.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from fire3d.webui.config import SCENE_ID_PREFIX, WebUIConfig
from fire3d.webui.depth import DepthEstimator, load_depth_estimator
from fire3d.webui.resources import (
    BootstrapReport,
    bootstrap,
    prune_scenes,
    write_bootstrap_report,
)
from fire3d.webui.runner import InferenceResult, run_inference, tail_log
from fire3d.webui.scene_builder import build_scene

LogFn = Callable[[str], None]
LOG_TAIL_LINES = 200

TITLE = "Fire3D · 单图生成 3D 场景"
DESCRIPTION = """
上传一张室内 RGB 照片，Fire3D 会一次性给出带纹理、可交互的 3D 场景（GLB）。

服务端流程：单目深度估计 → 组织化点云 → 冻结的 `fire3d_single_image_v1` 协议
（感知 + 重建）→ 场景 GLB。整个过程无需测试期优化，也不需要额外的位姿或深度输入。
"""


@dataclass
class WebUIState:
    """Server-side state shared by every request."""

    config: WebUIConfig
    report: BootstrapReport
    estimator: DepthEstimator | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    runs: int = 0

    def load_estimator(self, on_log: LogFn) -> DepthEstimator:
        if self.estimator is None:
            self.estimator = load_depth_estimator(self.config, on_log)
        return self.estimator


def _log_sink(lines: list[str], on_log: LogFn | None = None) -> LogFn:
    """Collect child-process output for the UI and echo it to the console."""

    def emit(message: str) -> None:
        lines.append(message)
        print(message, flush=True)
        if on_log is not None:
            on_log(message)

    return emit


def _log_text(lines: list[str]) -> str:
    return "\n".join(lines[-LOG_TAIL_LINES:])


def _describe_failure(result: InferenceResult) -> str:
    details = [result.to_markdown()]
    tail = tail_log(result.log_path, lines=25)
    if tail:
        details.append("")
        details.append("```text")
        details.append(tail)
        details.append("```")
    return "\n".join(details)


def generate(
    state: WebUIState,
    image_path: str | None,
    fov_degrees: float,
    max_side: int,
    render_preview: bool,
    gpu: str,
) -> Iterator[tuple]:
    """Scene build + inference, yielded stage by stage for live UI updates.

    Yields the same six outputs every time: status markdown, depth preview, GLB
    viewer, GLB download, render gallery, and log text.
    """

    lines: list[str] = []
    emit = _log_sink(lines)

    def bundle(status: str, preview: Any = None, gallery: Any = None, glb: Any = None):
        download = glb if isinstance(glb, str) else None
        return status, preview, glb, download, gallery, _log_text(lines)

    if not image_path:
        yield bundle("❌ 请先上传一张图片。")
        return

    config = state.config.replace(
        fov_degrees=float(fov_degrees),
        max_side=int(max_side),
        render_preview=bool(render_preview),
        gpu=str(gpu),
    )
    started = time.perf_counter()
    acquire = state.lock.acquire
    if not acquire(blocking=False):
        yield bundle("⏳ 已有一个任务在运行，当前请求排队等待 GPU 空闲 …")
        acquire(blocking=True)
    try:
        try:
            estimator = state.load_estimator(emit)
        except Exception as error:  # noqa: BLE001 - surfaced verbatim in the UI
            yield bundle(f"❌ 深度模型不可用：{error}")
            return

        try:
            summary = build_scene(config, estimator, image_path, log=emit)
        except Exception as error:  # noqa: BLE001 - surfaced verbatim in the UI
            yield bundle(f"❌ 场景构建失败：{error}")
            return
        preview = str(summary.layout.preview_path)
        yield bundle(
            "🖼️ 已构建单图场景，开始感知与重建 …\n\n" + summary.to_markdown(),
            preview=preview,
        )

        try:
            result = run_inference(config, summary.scene_id, on_log=emit)
        except Exception as error:  # noqa: BLE001 - surfaced verbatim in the UI
            yield bundle(f"❌ 推理无法启动：{error}", preview=preview)
            return

        elapsed = time.perf_counter() - started
        if not result.ok:
            yield bundle(
                f"❌ 重建失败（耗时 {elapsed:.0f} s）\n\n" + _describe_failure(result),
                preview=preview,
            )
            return

        glb = str(result.downloadable_glb)
        gallery = [str(path) for path in result.renders]
        state.runs += 1
        prune_scenes(
            config.scene_root,
            config.keep_scenes,
            prefix=SCENE_ID_PREFIX,
            on_log=emit,
        )
        status = "\n\n".join(
            [
                f"✅ 重建完成，耗时 {elapsed:.0f} s",
                summary.to_markdown(),
                result.to_markdown(),
                "_在预览中拖动可旋转查看，或下载 GLB 到任意三维查看器。_",
            ]
        )
        yield bundle(status, preview=preview, gallery=gallery or None, glb=glb)
    finally:
        state.lock.release()


def _status_markdown(config: WebUIConfig, report: BootstrapReport) -> str:
    header = "### 服务器资源"
    lines = [header, "", report.to_markdown()]
    if report.depth_model and report.depth_model != config.depth_model:
        lines.extend(
            [
                "",
                f"> 已回退到相对深度模型 `{report.depth_model}`，"
                "点云尺度由重建出的房间高度标定。",
            ]
        )
    lines.extend(
        [
            "",
            f"场景目录：`{config.scene_root}`　输出目录：`{config.output_root}`",
        ]
    )
    if not report.ok:
        lines.insert(
            0,
            "> ⚠️ 资源未就绪，点击生成会失败；请按下面的 ❌ 提示处理后重启服务。\n",
        )
    return "\n".join(lines)


def build_interface(state: WebUIState):
    """Construct the Gradio Blocks app. Importing gradio happens here."""

    import gradio as gr

    config = state.config
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)
        with gr.Accordion("服务器资源状态", open=not state.report.ok):
            gr.Markdown(_status_markdown(config, state.report))

        with gr.Row():
            with gr.Column(scale=4):
                image = gr.Image(
                    label="输入 RGB 图片",
                    type="filepath",
                    sources=["upload", "clipboard"],
                    height=340,
                )
                with gr.Accordion("高级选项", open=False):
                    fov = gr.Slider(
                        minimum=30.0,
                        maximum=110.0,
                        value=float(config.fov_degrees),
                        step=1.0,
                        label="水平视场角 (°)",
                        info="未知相机时的名义值；影响场景比例",
                    )
                    max_side = gr.Dropdown(
                        choices=[640, 960, 1280, 1600],
                        value=int(config.max_side),
                        label="输入长边上限（像素）",
                        info="越大越精细，显存与耗时也越高",
                    )
                    render_preview = gr.Checkbox(
                        value=bool(config.render_preview),
                        label="渲染静态预览图",
                        info="需要服务器安装 Blender 4.5.1",
                    )
                    gpu = gr.Textbox(value=str(config.gpu), label="GPU 编号")
                submit = gr.Button("生成 3D 模型", variant="primary")
                gr.Markdown(
                    "_首次运行需要加载感知、SS/Shape/PBR 流模型与 VAE 解码器，"
                    "比后续请求慢数十秒；整个流水线在单张 96 GB 级 GPU 上按分钟计。_"
                )
            with gr.Column(scale=6):
                status = gr.Markdown("上传图片后点击「生成 3D 模型」。")
                preview = gr.Image(label="深度预览", height=240, show_download_button=False)
                viewer = gr.Model3D(label="重建场景（GLB）", height=420)
                download = gr.File(label="下载 GLB", interactive=False)
                gallery = gr.Gallery(
                    label="渲染预览",
                    columns=2,
                    height=260,
                    object_fit="contain",
                )
                log = gr.Textbox(
                    label="运行日志",
                    lines=18,
                    max_lines=26,
                    autoscroll=True,
                    show_copy_button=True,
                )

        submit.click(
            fn=lambda *args: generate(state, *args),
            inputs=[image, fov, max_side, render_preview, gpu],
            outputs=[status, preview, viewer, download, gallery, log],
            concurrency_limit=config.concurrency,
        )
        image.change(
            fn=lambda: "图片已就绪，点击「生成 3D 模型」。",
            outputs=status,
        )
    return demo


def launch(
    config: WebUIConfig,
    *,
    on_log: LogFn | None = None,
    report: BootstrapReport | None = None,
) -> int:
    """Provision resources, then serve the Gradio app."""

    emit: LogFn = on_log or (lambda message: print(message, flush=True))
    if report is None:
        report = bootstrap(config, on_log=emit, download=config.bootstrap)
    if report.depth_model and report.depth_model != config.depth_model:
        config = config.replace(
            depth_model=report.depth_model,
            depth_model_dir=report.depth_model_dir,
        )
    write_bootstrap_report(report, config.output_root / "bootstrap.json")
    state = WebUIState(config=config, report=report)

    try:
        import gradio  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "gradio is required for the WebUI; run "
            "`python -m pip install -e '.[webui]'`"
        ) from error

    demo = build_interface(state)
    emit(
        f"[webui] 启动 Gradio：http://{config.host}:{config.port}"
        f"（GPU {config.gpu}，渲染预览 {'开' if config.render_preview else '关'}）"
    )
    demo.queue(default_concurrency_limit=config.concurrency).launch(
        server_name=config.host,
        server_port=config.port,
        share=config.share,
        auth=config.auth,
        show_error=True,
        # The GLB viewer and the download button hand Gradio absolute paths
        # inside these roots; naming them keeps the served set explicit.
        allowed_paths=[str(config.output_root), str(config.scene_root)],
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fire3d.webui.app",
        description="Gradio WebUI for single-image Fire3D reconstruction.",
    )
    parser.add_argument("--host", default=None, help="Bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port (default 7860)")
    parser.add_argument("--share", action="store_true", default=None)
    parser.add_argument("--auth", default=None, metavar="USER:PASSWORD")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--fov", type=float, default=None, help="Nominal horizontal FOV")
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--render-preview", action="store_true", default=None)
    parser.add_argument("--depth-model", default=None, help="Hugging Face depth model id")
    parser.add_argument("--keep-scenes", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--example-data",
        action="store_true",
        default=None,
        help="Also download the released single-image example scene",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        default=False,
        help="Verify resources but download nothing",
    )
    return parser


def config_from_namespace(args: argparse.Namespace) -> WebUIConfig:
    bootstrap_enabled = False if getattr(args, "no_bootstrap", False) else None
    return WebUIConfig.from_env(
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        share=getattr(args, "share", None),
        auth=getattr(args, "auth", None),
        gpu=getattr(args, "gpu", None),
        fov_degrees=getattr(args, "fov", None),
        max_side=getattr(args, "max_side", None),
        render_preview=getattr(args, "render_preview", None),
        depth_model=getattr(args, "depth_model", None),
        keep_scenes=getattr(args, "keep_scenes", None),
        concurrency=getattr(args, "concurrency", None),
        example_data=getattr(args, "example_data", None),
        bootstrap=bootstrap_enabled,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return launch(config_from_namespace(args))


if __name__ == "__main__":
    raise SystemExit(main())
