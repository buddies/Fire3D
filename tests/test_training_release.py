import json
import re
from pathlib import Path

import yaml

from baselines.fetch import parse_args as parse_baseline_fetch_args
from training.config import load_config


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIGS = tuple(sorted((ROOT / "configs/training").rglob("*.yaml")))
REQUIRED_SCENE_DATASETS = {
    "sage10k",
    "internscenes",
    "mansionworld",
    "ithor",
    "procthor",
    "scenesmith",
    "imaginarium",
}
REQUIRED_OBJECT_DATASETS = {
    "3d_future",
    "abo",
    "hssd",
    "objaverse_github",
    "objaverse_sketchfab",
}


def test_training_configs_resolve_only_explicit_environment_roots(monkeypatch, tmp_path):
    assert TRAINING_CONFIGS
    for path in TRAINING_CONFIGS:
        text = path.read_text(encoding="utf-8")
        for variable in set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text)):
            monkeypatch.setenv(variable, str(tmp_path / variable.lower()))
        resolved = json.dumps(load_config(path))
        assert "${" not in resolved
        assert "/home/" not in resolved


def test_public_release_sources_do_not_expose_private_paths_or_numbered_checkpoints():
    roots = (
        "README.md",
        "docs",
        "configs",
        "training",
        "baselines",
        "data_processing",
        "scripts",
        "fire3d",
        "eval",
        "benchmarks",
        "models",
        "modules",
        "utils",
    )
    private_path = re.compile(r"/(?:home|data)/(?:hongchi|hongchix)(?:/|\b)")
    numbered_checkpoint = re.compile(r"(?:step|iter)[_-]?\d{4,}", re.IGNORECASE)
    numbered_selector = re.compile(
        r"(?<![A-Za-z_])(?:step|iter)\s*[:=]\s*\d{4,}", re.IGNORECASE
    )
    failures = []
    for root_name in roots:
        root = ROOT / root_name
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if (
                "_upstream" in path.parts
                or not path.is_file()
                or path.suffix in {".png", ".jpg", ".pdf", ".pyc"}
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if (
                private_path.search(text)
                or numbered_checkpoint.search(text)
                or numbered_selector.search(text)
            ):
                failures.append(path.relative_to(ROOT).as_posix())
    assert failures == []


def test_baseline_registry_is_pinned_and_all_adapters_exist():
    registry = json.loads((ROOT / "baselines/registry.json").read_text())
    assert registry["schema"] == "fire3d.baseline_registry.v1"
    for method in registry["methods"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", method["revision"])
        assert method["repository"].startswith("https://github.com/")
        assert all((ROOT / adapter).is_file() for adapter in method["adapters"])


def test_baseline_fetch_without_names_selects_all_in_main():
    args = parse_baseline_fetch_args([])
    assert args.methods == []
    assert args.checkout_root == ROOT / "baselines/_upstream"


def test_data_processing_registry_covers_release_sources_and_stages():
    registry = json.loads((ROOT / "data_processing/registry.json").read_text())
    assert set(registry["scene_datasets"]) == REQUIRED_SCENE_DATASETS
    assert set(registry["object_datasets"]) == REQUIRED_OBJECT_DATASETS
    assert {"object_render", "ovoxel_manifest", "shape_encode", "pbr_encode"} <= set(
        registry["shared_stages"]
    )
    for section in ("scene_datasets", "object_datasets", "shared_stages"):
        for record in registry[section].values():
            paths = record.values() if isinstance(record, dict) else (record,)
            assert all((ROOT / path).is_file() for path in paths)


def test_flow_training_configs_use_stable_vae_aliases():
    for relative in ("flow_matching/shape.yaml", "flow_matching/pbr.yaml"):
        config = yaml.safe_load((ROOT / "configs/training" / relative).read_text())
        vae_sections = {
            key: value
            for key, value in config["model"].items()
            if key in {"shape_x2", "pbr_x2"}
        }
        assert vae_sections
        assert all(section.get("step") is None for section in vae_sections.values())


def test_huggingface_model_download_count_uses_root_query_file():
    card = (ROOT / "docs/huggingface_model_card.md").read_text(encoding="utf-8")
    config = json.loads((ROOT / "docs/huggingface_model_config.json").read_text())
    front_matter = card.split("---", 2)[1]
    assert "library_name:" not in front_matter
    assert config["schema"] == "fire3d_model_bundle_v1"
    assert config["architectures"] == ["Fire3DSceneReconstructionPipeline"]
    assert "root `config.json`" in card
    assert "https://huggingface.co/docs/hub/models-download-stats" in card


def test_hcvae_encoder_and_decoder_public_aliases_are_documented():
    for path in (ROOT / "README.md", ROOT / "docs/huggingface_model_card.md"):
        text = path.read_text(encoding="utf-8")
        assert "reconstruction/vae/shape/ckpts/encoder.pt" in text or "{encoder,decoder}.pt" in text
        assert "reconstruction/vae/pbr/ckpts/encoder.pt" in text or "{encoder,decoder}.pt" in text
