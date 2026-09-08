from eval.reconstruction.lc64_benchmark_appearance import (
    align_shared_dino_config,
    dino_signature,
)


def test_pbr_uses_release_local_shared_dino_contract():
    pbr = {
        "dino": {
            "repo_dir": "/home/author/codes/dino/dinov3",
            "model_name": "dinov3_vitl16",
            "model_path": "/home/author/codes/dino/model.pth",
            "downsample": 16,
            "upsample": 4,
        }
    }
    shared = {
        "dino": {
            "repo_dir": "third_party/dinov3",
            "model_name": "dinov3_vitl16",
            "model_path": "checkpoints/Fire3D/external/dinov3.pth",
            "downsample": 16,
            "upsample": 4,
            "anyup_repo": "third_party/anyup",
            "anyup_source": "local",
        }
    }

    aligned = align_shared_dino_config(pbr, shared)

    assert dino_signature(aligned) == dino_signature(shared)
    assert aligned["dino"]["anyup_repo"] == "third_party/anyup"
    assert aligned["dino"]["anyup_source"] == "local"
