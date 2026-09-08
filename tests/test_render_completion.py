import json
from types import SimpleNamespace

from eval.unified_percept_recon import render_result_ready
from eval.unified_render import completed_requested_renders


def test_render_result_requires_scene_success(tmp_path):
    args = SimpleNamespace(output_root=tmp_path)
    index = tmp_path / "renders/render_index.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps({"scenes": {"scene": {"status": "failed"}}})
    )
    assert render_result_ready(args, "scene") is False

    index.write_text(
        json.dumps({"scenes": {"scene": {"status": "complete"}}})
    )
    assert render_result_ready(args, "scene") is True


def test_render_exit_count_ignores_unrequested_index_records():
    scenes = {
        "old": {"status": "complete"},
        "requested": {"status": "failed"},
    }
    assert completed_requested_renders(scenes, ["requested"]) == 0
