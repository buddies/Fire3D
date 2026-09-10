"""Tests for the WebUI's wrapper around the frozen `fire3d infer` interface.

The command line is part of the contract: it has to stay the released entry
point, and its data root has to be the parent of the scene root because the CLI
validates `<data_root>/single_image` before forwarding it as the protocol's
native input root.
"""

import json
import sys

from fire3d.webui.config import WebUIConfig
from fire3d.webui.runner import (
    SCENE_GLB,
    TRAINING_GLB,
    build_infer_command,
    collect_outputs,
    find_scene_glbs,
    scene_output_root,
    tail_log,
)


def make_config(tmp_path, **overrides) -> WebUIConfig:
    config = WebUIConfig.from_env(depth_model="plane", gpu="0")
    return config.replace(
        scene_root=tmp_path / "webui/single_image",
        output_root=tmp_path / "out",
        cache_root=tmp_path / "cache",
        **overrides,
    )


def test_infer_command_reuses_the_released_entry_point(tmp_path):
    config = make_config(tmp_path, render_preview=False)
    command = build_infer_command(config, "webui_demo")
    assert command[:5] == [sys.executable, "-m", "fire3d", "infer", "--dataset"]
    assert command[5] == "single_image"
    assert "--scene-id" in command and "webui_demo" in command
    assert "--skip-render" in command
    data_root = command[command.index("--data-root") + 1]
    assert data_root == str(config.scene_root.parent)
    assert str(config.scene_root).endswith("single_image")


def test_infer_command_renders_when_the_preview_is_enabled(tmp_path):
    config = make_config(tmp_path, render_preview=True)
    assert "--skip-render" not in build_infer_command(config, "webui_demo")
    assert config.skip_render is False


def test_scene_output_root_is_per_scene(tmp_path):
    config = make_config(tmp_path)
    assert scene_output_root(config, "a") != scene_output_root(config, "b")


def test_find_scene_glbs_prefers_the_driver_layout(tmp_path):
    output_root = tmp_path / "out"
    appearance = output_root / "reconstruction" / "webui_demo" / "appearance"
    appearance.mkdir(parents=True)
    world = appearance / SCENE_GLB
    world.write_bytes(b"glTF")
    training = appearance / TRAINING_GLB
    training.write_bytes(b"glTF")

    scene_glb, training_glb = find_scene_glbs(output_root, "webui_demo")
    assert scene_glb == world
    assert training_glb == training


def test_find_scene_glbs_falls_back_to_any_glb(tmp_path):
    deep = tmp_path / "out" / "reconstruction" / "webui_demo" / "other"
    deep.mkdir(parents=True)
    other = deep / "fallback.glb"
    other.write_bytes(b"glTF")

    scene_glb, training_glb = find_scene_glbs(tmp_path / "out", "webui_demo")
    assert scene_glb == other
    assert training_glb is None


def test_find_scene_glbs_ignores_empty_files(tmp_path):
    appearance = (
        tmp_path / "out" / "reconstruction" / "webui_demo" / "appearance"
    )
    appearance.mkdir(parents=True)
    (appearance / SCENE_GLB).write_bytes(b"")
    assert find_scene_glbs(tmp_path / "out", "webui_demo") == (None, None)


def test_collect_outputs_reports_success_and_artifacts(tmp_path):
    output_root = tmp_path / "out" / "webui_demo"
    appearance = output_root / "reconstruction" / "webui_demo" / "appearance"
    appearance.mkdir(parents=True)
    (appearance / SCENE_GLB).write_bytes(b"glTF")
    objects = output_root / "reconstruction" / "webui_demo" / "objects"
    objects.mkdir(parents=True)
    (objects / "pred_0001.ply").write_bytes(b"ply")
    renders = output_root / "renders"
    renders.mkdir(parents=True)
    (renders / "sheet.png").write_bytes(b"png")
    (output_root / "summary.json").write_text(
        json.dumps({"scenes": {"webui_demo": {"status": "complete"}}}),
        encoding="utf-8",
    )

    result = collect_outputs(
        "webui_demo",
        output_root,
        returncode=0,
        log_path=output_root / "logs" / "webui_infer.log",
    )
    assert result.ok
    assert result.scene_status() == "complete"
    assert result.downloadable_glb == appearance / SCENE_GLB
    assert len(result.object_meshes) == 1
    assert len(result.renders) == 1
    assert SCENE_GLB in result.to_markdown()


def test_collect_outputs_reports_failure_without_a_glb(tmp_path):
    output_root = tmp_path / "out" / "webui_demo"
    output_root.mkdir(parents=True)
    result = collect_outputs(
        "webui_demo",
        output_root,
        returncode=1,
        log_path=output_root / "logs" / "webui_infer.log",
    )
    assert not result.ok
    assert result.downloadable_glb is None
    assert result.scene_status() == "unknown"


def test_tail_log_returns_the_last_lines(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join(f"line {index}" for index in range(60)), encoding="utf-8")
    assert tail_log(log, lines=3) == "line 57\nline 58\nline 59"
    assert tail_log(tmp_path / "missing.log") == ""
