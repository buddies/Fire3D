from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_release_wording_is_reused_verbatim():
    overview = (ROOT / "docs/release_overview.md").read_text().strip()
    documents = [
        ROOT / "README.md",
        ROOT / "docs/huggingface_model_card.md",
        ROOT / "docs/huggingface_dataset_card.md",
    ]
    assert all(overview in path.read_text() for path in documents)


def test_release_does_not_track_runtime_artifacts_or_research_workspaces():
    forbidden = {
        "ckpts",
        "checkpoints",
        "deprecated",
        "distill",
        "preprocess",
        "results",
        "syn2real",
        "temp_scripts",
        "trainers",
    }
    candidates = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    present = {path.split("/", 1)[0] for path in candidates}
    assert not (forbidden & present)


def test_repository_contains_no_accidental_large_files():
    large = []
    candidates = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for relative in candidates:
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            large.append(path.relative_to(ROOT).as_posix())
    assert large == []


def test_third_party_components_ship_license_notices():
    required = [
        ROOT / "licenses/DINOV3_LICENSE.md",
        ROOT / "licenses/TRELLIS2_LICENSE",
        ROOT / "third_party/anyup/LICENSE",
        ROOT / "trellis2_x2/CuMesh/LICENSE",
        ROOT / "baselines/shaper/LICENSE",
        ROOT / "baselines/boxer_trellis2/LICENSE",
    ]
    assert all(path.is_file() for path in required)


def test_public_runtime_does_not_import_research_preprocess_tree():
    runtime_files = [
        ROOT / "fire3d/runtime/model_loader.py",
        ROOT / "eval/reconstruction/lc64_shape_pbr_decode.py",
        ROOT / "benchmarks/scene_reconstruction/run_lc64_geometry.py",
    ]
    assert all("from preprocess" not in path.read_text() for path in runtime_files)


def test_vendored_trellis_decoder_modules_import(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "trellis2_x2"))
    from trellis2 import models

    assert models.SparseStructureEncoder is not None
    assert models.SparseStructureDecoder is not None
    assert models.SparseUnetVaeDecoder is not None
    assert models.FlexiDualGridVaeDecoder is not None


def test_installer_handles_strict_shell_cuda_activation():
    installer = (ROOT / "scripts/install.sh").read_text()
    activate_offset = installer.index('source "$VENV_DIR/bin/activate"')
    for variable in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        initialization = f'export {variable}="${{{variable}:-}}"'
        assert installer.index(initialization) < activate_offset
    # The toolkit is resolved before the multi-gigabyte downloads that need it,
    # so a missing nvcc fails immediately instead of an hour into the install.
    assert installer.index("nvcc was not found") < installer.index("torch==2.7.1")


def test_installer_bounds_cuda_source_builds():
    """Memory-limited containers must not compile CUDA with unbounded parallelism.

    `pip install flash-attn` has no PyPI wheel, and its ninja build defaults to
    one nvcc per core at several GB each -- that is what OOM-kills a k8s pod
    (exit 137). The installer therefore caps the job count, prefers the
    project's prebuilt wheel, and offers a backend that compiles nothing.
    """

    installer = (ROOT / "scripts/install.sh").read_text()
    assert 'export MAX_JOBS="${FIRE3D_BUILD_JOBS:-' in installer
    assert 'export NVCC_THREADS="${FIRE3D_NVCC_THREADS:-' in installer
    # Prebuilt wheel first: the name encodes the torch series, the C++11 ABI and
    # the interpreter tag, all of which have to match the installed torch.
    assert "cxx11abi" in installer
    assert "FIRE3D_ATTENTION_BACKEND" in installer
    assert "ATTN_BACKEND=xformers" in installer
    # Both the cap and the wheel attempt must precede any source build.
    source_build = installer.index('--no-build-isolation "flash-attn==')
    assert installer.index("export MAX_JOBS=") < source_build
    assert installer.index("cxx11abi") < source_build
    # The unbounded single-command form must not come back.
    assert "flash-attn==2.7.3 --no-build-isolation" not in installer
