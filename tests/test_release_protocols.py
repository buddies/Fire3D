import json
from pathlib import Path

import pytest

from eval.unified_protocol import load_protocol, reproduction_geometry_args
from fire3d.runtime.model_loader import conditioning_overrides


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {
    "video": "fire3d_video_v1",
    "scannetpp": "fire3d_scannetpp_v1",
    "single_image": "fire3d_single_image_v1",
}


@pytest.mark.parametrize("name", PROTOCOLS.values())
def test_release_protocols_are_portable_and_complete(name):
    protocol, source = load_protocol(name)
    assert source == ROOT / "configs/inference" / f"{name}.json"
    serialized = json.dumps(protocol)
    assert "/home/" not in serialized
    assert '"ckpts/' not in serialized
    assert protocol["status"] == "release"
    assert protocol["evaluation"]["enabled"] is False

    args = reproduction_geometry_args(protocol, ROOT)
    command = " ".join(args)
    assert "/codes/ff_holoscene" not in command
    assert "/codes/dino" not in command
    assert "checkpoints/Fire3D/reconstruction" in command
    assert "checkpoints/Fire3D/external/trellis2" in command
    assert "third_party/dinov3" in command
    assert "third_party/anyup" in command


@pytest.mark.parametrize("name", PROTOCOLS.values())
def test_release_protocols_use_batchified_execution(name):
    protocol, _ = load_protocol(name)
    reconstruction = protocol["reconstruction"]
    flow = reconstruction["flow"]
    assert flow["ss_shape_object_batch_size"] == 16
    assert flow["pbr_object_batch_size"] == 16
    assert reconstruction["appearance_decode"] == {
        "object_chunk_size": 16,
        "persist_raw_decoded": False,
    }
    mesh = reconstruction["mesh_postprocess"]
    assert mesh["backend"] == "batch"
    assert mesh["object_batch_size"] == 8
    assert mesh["uv_cpu_workers"] == 32
    assert mesh["sparse_query_mode"] == "power2"


@pytest.mark.parametrize("name", PROTOCOLS.values())
def test_release_protocols_use_stable_model_aliases(name):
    protocol, _ = load_protocol(name)
    assert protocol["perception"]["checkpoint"] == "model.pt"
    reconstruction = protocol["reconstruction"]
    for stage in ("ss", "shape", "pbr"):
        assert reconstruction["flow"][stage]["checkpoint"] == (
            f"flows/{stage}/model.pt"
        )
        assert reconstruction["vae"][stage]["checkpoint"] == "decoder.pt"


@pytest.mark.parametrize("name", (PROTOCOLS["video"], PROTOCOLS["scannetpp"]))
def test_scene_dataset_protocols_enable_background_room_fit(name):
    protocol, _ = load_protocol(name)
    background = protocol["reconstruction"]["background"]
    assert background["room_box_prior"] is True
    assert background["canonical_transform"] == "room_box_isotropic_enclose"
    assert background["unit_box_prune"] is True
    assert background["canonical_margin"] == pytest.approx(0.01)


def test_dataset_specific_quality_settings_are_preserved():
    video, _ = load_protocol(PROTOCOLS["video"])
    assert video["reconstruction"]["flow"]["sample_method"] == "fps"
    assert video["reconstruction"]["flow"]["max_cond_len"] == 1024
    assert video["reconstruction"]["mesh_postprocess"]["topology_remesh_band"] == 1.0

    for name in (PROTOCOLS["scannetpp"], PROTOCOLS["single_image"]):
        protocol, _ = load_protocol(name)
        flow = protocol["reconstruction"]["flow"]
        assert flow["sample_method"] == "hybrid"
        assert flow["max_cond_len"] == 1024
        assert flow["pbr_sample_method"] == "random"
        assert flow["pbr_max_cond_len"] == 512
        assert protocol["reconstruction"]["mesh_postprocess"]["topology_remesh_band"] == 2.0

    scannetpp, _ = load_protocol(PROTOCOLS["scannetpp"])
    options = scannetpp["input_rgb"]["loader_options"]
    assert options == {
        "depth_confidence_threshold": 0.6,
        "depth_invalid_policy": "drop",
        "depth_subdir": "depth_pi3x_conf_chunk16",
        "max_frames": 300,
        "wall_alignment": True,
    }
    assert scannetpp["input_rgb"]["environment"]["FF_SCANNETPP_WHITELIST"] == "1"

    single, _ = load_protocol(PROTOCOLS["single_image"])
    assert single["input_rgb"]["loader_options"]["wall_alignment"] is False
    assert single["views"]["profile"] == "native-input-camera"


def test_flow_latent_statistics_resolve_inside_release_bundle(tmp_path):
    bundle = tmp_path / "checkpoints/Fire3D/reconstruction"
    args = type(
        "Args",
        (),
        {
            "max_cond_len": 1024,
            "dino_upsample": 4,
            "anyup_frame_batch_size": 16,
            "dino_repo_dir": tmp_path / "third_party/dinov3",
            "dino_model_path": tmp_path / "checkpoints/Fire3D/external/dino.pth",
            "anyup_repo_dir": tmp_path / "third_party/anyup",
            "ss_run_dir": bundle / "flows/ss",
        },
    )()
    config = {
        "model": {
            "dino": {},
            "ss_x2": {"latent_stats_path": "/home/author/ss.json"},
            "shape_x2": {"x2_latent_stats_path": "/home/author/shape.json"},
        }
    }

    resolved = conditioning_overrides(config, args)["model"]

    assert resolved["ss_x2"]["latent_stats_path"] == str(
        bundle / "stats/ss.json"
    )
    assert resolved["shape_x2"]["x2_latent_stats_path"] == str(
        bundle / "stats/shape.json"
    )
