import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from subprocess import DEVNULL, call

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import o_voxel
import torch
import trimesh
from tqdm import tqdm

torch.set_grad_enabled(False)


REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "trellis2_x2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.objects import dataset_choices, load_adapter

BLENDER_SCRIPT_DIR = REPO_ROOT / "data_processing" / "blender"
BLENDER_LINK = "https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = REPO_ROOT / "blender"
BLENDER_DIR = BLENDER_INSTALLATION_PATH / "blender-4.5.1-linux-x64"
BLENDER_PATH = BLENDER_DIR / "blender"
OBJECT_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_ROOT", REPO_ROOT / "data/training_objects")
)
POSTPROCESS_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_POSTPROCESS_ROOT", OBJECT_ROOT / "postprocess")
)
LOG_ROOT = Path(
    os.environ.get("FIRE3D_PROCESSING_LOG_ROOT", REPO_ROOT / "outputs/data_processing/logs")
)
TEMP_DIR = POSTPROCESS_ROOT / "temp_glbs"


def _install_blender():
    if BLENDER_PATH.exists():
        return

    BLENDER_INSTALLATION_PATH.mkdir(parents=True, exist_ok=True)
    archive_path = BLENDER_INSTALLATION_PATH / Path(BLENDER_LINK).name
    if not archive_path.exists():
        print(f"Downloading Blender to {archive_path}", flush=True)
        urllib.request.urlretrieve(BLENDER_LINK, archive_path)

    print(f"Extracting Blender to {BLENDER_INSTALLATION_PATH}", flush=True)
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(BLENDER_INSTALLATION_PATH)

    if not BLENDER_PATH.exists():
        raise FileNotFoundError(f"Failed to install Blender at {BLENDER_PATH}")


def _load_dataset_module(dataset_name):
    return load_adapter(dataset_name)


def _object_id_from_path(model_path):
    model_path = Path(model_path)
    if model_path.stem in {"raw_model", "model"}:
        return model_path.parent.name
    return model_path.stem


def _save_glb(mesh_or_scene, save_path):
    rot_x_90 = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh_or_scene_copy = mesh_or_scene.copy()
    mesh_or_scene_copy.apply_transform(rot_x_90)
    trimesh.exchange.export.export_mesh(mesh_or_scene_copy, save_path)


def _dump_mesh(file_path, output_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_name = os.path.splitext(os.path.basename(output_path))[0]
        temp_path = os.path.join(tmp_dir, f"{output_name}.pkl")
        args = [
            str(BLENDER_PATH),
            "-b",
            "-P",
            str(BLENDER_SCRIPT_DIR / "dump_mesh.py"),
            "--",
            "--object",
            os.path.expanduser(file_path),
            "--output_path",
            os.path.expanduser(temp_path),
        ]
        if file_path.endswith(".blend"):
            args.insert(1, file_path)

        # call(args)
        call(args, stdout=DEVNULL, stderr=DEVNULL)

        if os.path.exists(temp_path):
            shutil.move(temp_path, output_path)
            return True
        return False


def _dual_grid_mesh(dump, output_path, res=512):
    try:
        need_process = False

        if os.path.exists(output_path):
            try:
                o_voxel.io.read_vxz_info(output_path)
            except Exception as e:
                print(f"Error reading {output_path}: {e}")
                need_process = True
        else:
            need_process = True

        if need_process:
            start = 0
            vertices = []
            faces = []
            for obj in dump["objects"]:
                if obj["vertices"].size == 0 or obj["faces"].size == 0:
                    continue
                vertices.append(obj["vertices"])
                faces.append(obj["faces"] + start)
                start += len(obj["vertices"])
            vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
            faces = torch.from_numpy(np.concatenate(faces, axis=0)).long()
            vertices_min = vertices.min(dim=0)[0]
            vertices_max = vertices.max(dim=0)[0]
            center = (vertices_min + vertices_max) / 2
            scale = 0.99999 / (vertices_max - vertices_min).max()
            vertices = (vertices - center) * scale
            assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), "vertices out of range"
            data = {"vertices": vertices, "faces": faces}

            voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
                **data,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                face_weight=1.0,
                boundary_weight=0.2,
                regularization_weight=1e-2,
                timing=False,
            )
            dual_vertices = dual_vertices * res - voxel_indices
            assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3), "dual_vertices out of range"
            dual_vertices = torch.clamp(dual_vertices, 0, 1)
            dual_vertices = (dual_vertices * 255).type(torch.uint8)
            intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

            o_voxel.io.write_vxz(
                output_path,
                voxel_indices,
                {"vertices": dual_vertices, "intersected": intersected},
            )
            del voxel_indices, dual_vertices, intersected, data

        return True
    except Exception as e:
        print(f"Error voxelizing {output_path}: {e}")
        return False


def _dump_pbr(file_path, output_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_name = os.path.splitext(os.path.basename(output_path))[0]
        temp_path = os.path.join(tmp_dir, f"{output_name}.pkl")
        args = [
            str(BLENDER_PATH),
            "-b",
            "-P",
            str(BLENDER_SCRIPT_DIR / "dump_pbr.py"),
            "--",
            "--object",
            os.path.expanduser(file_path),
            "--output_path",
            os.path.expanduser(temp_path),
        ]
        if file_path.endswith(".blend"):
            args.insert(1, file_path)

        # call(args)
        call(args, stdout=DEVNULL, stderr=DEVNULL)

        if os.path.exists(temp_path):
            shutil.move(temp_path, output_path)
            return True
        return False


def _pbr_voxelize(dump, output_path, res=512):
    try:
        need_process = False

        if os.path.exists(output_path):
            try:
                o_voxel.io.read_vxz_info(output_path)
            except Exception as e:
                print(f"Error reading {output_path}: {e}")
                need_process = True
        else:
            need_process = True

        if need_process:
            for mat in dump["materials"]:
                if mat["alphaTexture"] is not None and mat["alphaMode"] == "OPAQUE":
                    mat["alphaMode"] = "BLEND"
            dump["materials"].append(
                {
                    "baseColorFactor": [0.8, 0.8, 0.8],
                    "alphaFactor": 1.0,
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.5,
                    "alphaMode": "OPAQUE",
                    "alphaCutoff": 0.5,
                    "baseColorTexture": None,
                    "alphaTexture": None,
                    "metallicTexture": None,
                    "roughnessTexture": None,
                }
            )
            dump["objects"] = [
                obj for obj in dump["objects"]
                if obj["vertices"].size != 0 and obj["faces"].size != 0
            ]
            vertices = torch.from_numpy(np.concatenate([obj["vertices"] for obj in dump["objects"]], axis=0)).float()
            vertices_min = vertices.min(dim=0)[0]
            vertices_max = vertices.max(dim=0)[0]
            center = (vertices_min + vertices_max) / 2
            scale = 0.99999 / (vertices_max - vertices_min).max()
            for obj in dump["objects"]:
                obj["vertices"] = (torch.from_numpy(obj["vertices"]).float() - center) * scale
                obj["vertices"] = obj["vertices"].numpy()
                obj["mat_ids"][obj["mat_ids"] == -1] = len(dump["materials"]) - 1
                assert np.all(obj["mat_ids"] >= 0), "invalid mat_ids"
                assert np.all(obj["vertices"] >= -0.5) and np.all(obj["vertices"] <= 0.5), "vertices out of range"

            coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
                dump,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                mip_level_offset=0,
                verbose=False,
                timing=False,
            )
            del attr["normal"]
            del attr["emissive"]
            o_voxel.io.write_vxz(output_path, coord, attr)
            del coord, attr

        return True
    except Exception as e:
        print(f"Error voxelizing {output_path}: {e}")
        return False


def _encode(model_path, mesh, object_id, shape_voxel_path, pbr_voxel_path, log_path, res=512):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    path_hash = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:16]
    temp_save_name = f"{object_id}_{path_hash}"
    export_glb_path = TEMP_DIR / f"{temp_save_name}.glb"
    export_pbr_path = TEMP_DIR / f"{temp_save_name}_pbr.pkl"

    try:
        _save_glb(mesh, export_glb_path)

        if not _dump_pbr(file_path=str(export_glb_path), output_path=str(export_pbr_path)):
            print(f"Failed to dump PBR for {object_id}")
            return False

        with open(export_pbr_path, "rb") as f:
            dump = pickle.load(f)

        if not _dual_grid_mesh(dump=dump, output_path=str(shape_voxel_path), res=res):
            print(f"Failed to voxelize shape for {object_id}")
            return False
        if not _valid_vxz(shape_voxel_path):
            print(f"Shape vxz invalid after voxelize for {object_id}")
            return False

        if not _pbr_voxelize(dump=dump, output_path=str(pbr_voxel_path), res=res):
            print(f"Failed to voxelize PBR for {object_id}")
            return False
        if not _valid_vxz(pbr_voxel_path):
            print(f"PBR vxz invalid after voxelize for {object_id}")
            return False

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(
                {
                    "object_id": object_id,
                    "model_path": str(model_path),
                    "shape_voxel": str(shape_voxel_path),
                    "pbr_voxel": str(pbr_voxel_path),
                },
                f,
            )
        return True
    finally:
        for path in (export_glb_path, export_pbr_path):
            if path.exists():
                path.unlink()


_VALIDATE_VXZ_SNIPPET = (
    "import sys; "
    "import o_voxel; "
    "o_voxel.io.read_vxz_info(sys.argv[1])"
)


def _valid_vxz(path):
    # Run the check in an isolated subprocess so a native crash inside
    # o_voxel.io.read_vxz_info (e.g. on a truncated/corrupt vxz) only kills
    # the child rather than the worker process.
    path = Path(path)
    if not path.exists():
        print(f"vXZ file does not exist: {path}")
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", _VALIDATE_VXZ_SNIPPET, str(path)],
            stdout=DEVNULL,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except Exception as e:
        print(f"Error launching vXZ check for {path}: {e}")
        return False
    if result.returncode == 0:
        return True
    err = result.stderr.decode("utf-8", errors="replace").strip() if result.stderr else f"exit {result.returncode}"
    print(f"Error reading vXZ file: {path}; {err}")
    return False


def _process_one_object(dataset, model_path, shape_voxel_save_dir, pbr_voxel_save_dir, log_save_dir, res=512, overwrite=False):
    try:
        object_id = _object_id_from_path(model_path)
        shape_voxel_path = shape_voxel_save_dir / f"{object_id}.vxz"
        pbr_voxel_path = pbr_voxel_save_dir / f"{object_id}.vxz"
        log_path = log_save_dir / f"{object_id}.json"

        mesh = dataset.load_model(model_path)
        success = _encode(
            model_path=model_path,
            mesh=mesh,
            object_id=object_id,
            shape_voxel_path=shape_voxel_path,
            pbr_voxel_path=pbr_voxel_path,
            log_path=log_path,
            res=res,
        )
        return object_id, success
    except Exception as e:
        print(f"Error processing {model_path}: {e}")
        return None, False


# Per-worker state. Initialized once per process so we don't re-import the
# (heavy) dataset module or re-set torch thread caps for every task.
_WORKER_DATASET = None


def _worker_init(dataset_name, torch_num_threads):
    global _WORKER_DATASET
    torch.set_num_threads(max(1, torch_num_threads))
    torch.set_num_interop_threads(max(1, torch_num_threads))
    _, _WORKER_DATASET = _load_dataset_module(dataset_name)


def _worker_process(model_path, shape_voxel_save_dir, pbr_voxel_save_dir, log_save_dir, res):
    return _process_one_object(
        _WORKER_DATASET,
        model_path,
        shape_voxel_save_dir,
        pbr_voxel_save_dir,
        log_save_dir,
        res,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=dataset_choices(),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_random_10", action="store_true", help="Only process 10 randomly sampled objects.")
    parser.add_argument("--install_only", action="store_true", help="Install/check Blender and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate objects even when both output vxz files already exist.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of object-processing worker processes.")
    parser.add_argument("--torch_num_threads", type=int, default=1, help="PyTorch CPU threads per object worker. Keep at 1 when num_workers is high to avoid oversubscription.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_num_threads))
    torch.set_num_interop_threads(max(1, args.torch_num_threads))

    print("Checking blender...", flush=True)
    _install_blender()
    if args.install_only:
        print(f"Blender ready at {BLENDER_PATH}", flush=True)
        return

    module_name, dataset = _load_dataset_module(args.dataset_name)
    output_dataset_name = module_name
    shape_voxel_save_dir = POSTPROCESS_ROOT / "shape_ovoxels" / output_dataset_name
    pbr_voxel_save_dir = POSTPROCESS_ROOT / "pbr_ovoxels" / output_dataset_name
    log_save_dir = LOG_ROOT / output_dataset_name
    shape_voxel_save_dir.mkdir(parents=True, exist_ok=True)
    pbr_voxel_save_dir.mkdir(parents=True, exist_ok=True)
    log_save_dir.mkdir(parents=True, exist_ok=True)

    model_paths = dataset.list_all_model_paths()
    if args.debug_random_10:
        rng = random.Random(args.seed)
        model_paths = rng.sample(model_paths, min(10, len(model_paths)))

    if not args.overwrite:
        processed_ids = {p.stem for p in log_save_dir.glob("*.json")}
        if processed_ids:
            total_before = len(model_paths)
            model_paths = [p for p in model_paths if _object_id_from_path(p) not in processed_ids]
            print(f"Skipping {total_before - len(model_paths)} already-logged objects", flush=True)

    print(f"Processing {len(model_paths)} {output_dataset_name} objects", flush=True)
    success_count = 0
    num_workers = max(1, args.num_workers)
    if num_workers == 1:
        for model_path in tqdm(model_paths):
            _, success = _process_one_object(
                dataset=dataset,
                model_path=model_path,
                shape_voxel_save_dir=shape_voxel_save_dir,
                pbr_voxel_save_dir=pbr_voxel_save_dir,
                log_save_dir=log_save_dir,
                res=args.resolution,
            )
            success_count += int(success)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(args.dataset_name, args.torch_num_threads),
        ) as executor:
            model_path_iter = iter(model_paths)
            pending = set()

            def submit_next():
                try:
                    model_path = next(model_path_iter)
                except StopIteration:
                    return False
                pending.add(executor.submit(
                    _worker_process,
                    model_path,
                    shape_voxel_save_dir,
                    pbr_voxel_save_dir,
                    log_save_dir,
                    args.resolution,
                ))
                return True

            for _ in range(min(num_workers * 2, len(model_paths))):
                submit_next()

            with tqdm(total=len(model_paths)) as pbar:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            _, success = future.result()
                            success_count += int(success)
                        except Exception as e:
                            print(f"Error processing object: {e}", flush=True)
                        pbar.update(1)
                        submit_next()

    print(f"Finished {success_count}/{len(model_paths)} objects", flush=True)


if __name__ == "__main__":
    main()
