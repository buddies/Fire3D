import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _preparse_gpu_id():
    for i, a in enumerate(sys.argv):
        if a == "--gpu_id" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return -1
        if a.startswith("--gpu_id="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return -1
    return -1


_GPU_ID = _preparse_gpu_id()
if _GPU_ID >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU_ID)
    os.environ["EGL_DEVICE_ID"] = "0"

import numpy as np
import pyrender
import trimesh
from PIL import Image
from tqdm import tqdm

from data_processing.objects import dataset_choices, load_adapter


OBJECT_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_ROOT", REPO_ROOT / "data/training_objects")
)
RENDER_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_RENDER_ROOT", OBJECT_ROOT / "render")
)
RENDER_FLAGS = pyrender.constants.RenderFlags.SKIP_CULL_FACES


def load_dataset_module(dataset_name):
    return load_adapter(dataset_name)


def object_id_from_path(model_path):
    model_path = Path(model_path)
    if model_path.stem in {"raw_model", "model"}:
        return model_path.parent.name
    return model_path.stem


def _normalize_texture_image(image):
    if image is None:
        return None
    mode = getattr(image, "mode", None)
    if mode in ("RGB", "RGBA"):
        return image
    if mode in ("L", "I", "F", "P", "1"):
        return image.convert("RGB")
    return image.convert("RGBA")


_TEXTURE_ATTRS = (
    "image",
    "baseColorTexture",
    "emissiveTexture",
    "normalTexture",
    "occlusionTexture",
    "metallicRoughnessTexture",
)


def _has_valid_uvs(geom):
    visual = getattr(geom, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is None:
        return False
    uv = np.asarray(uv)
    if uv.ndim != 2 or uv.shape[1] < 2:
        return False
    if len(uv) != len(geom.vertices):
        return False
    return True


def sanitize_visual_textures(geom):
    visual = getattr(geom, "visual", None)
    material = getattr(visual, "material", None)
    if material is None:
        return
    has_uvs = _has_valid_uvs(geom)
    for attr in _TEXTURE_ATTRS:
        img = getattr(material, attr, None)
        if img is None:
            continue
        if not has_uvs:
            setattr(material, attr, None)
            continue
        fixed = _normalize_texture_image(img)
        if fixed is not img:
            setattr(material, attr, fixed)


def make_double_sided_geometry(geom):
    sanitize_visual_textures(geom)
    faces = np.asarray(geom.faces)
    double_sided_faces = np.concatenate([faces, faces[:, ::-1]], axis=0)
    double_sided = trimesh.Trimesh(
        vertices=np.asarray(geom.vertices).copy(),
        faces=double_sided_faces,
        visual=geom.visual.copy(),
        process=False,
        maintain_order=True,
    )
    if hasattr(double_sided.visual, "face_materials"):
        face_materials = np.asarray(geom.visual.face_materials)
        if face_materials.ndim == 0 or len(face_materials) != len(faces):
            return double_sided
        double_sided.visual.face_materials = np.concatenate(
            [face_materials, face_materials],
            axis=0,
        )
    return double_sided


def make_pyrender_mesh(geom):
    mesh = pyrender.Mesh.from_trimesh(make_double_sided_geometry(geom), smooth=False)
    for primitive in mesh.primitives:
        if primitive.material is not None and hasattr(primitive.material, "doubleSided"):
            primitive.material.doubleSided = True
    return mesh


def add_to_scene(scene, mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Scene):
        try:
            graph_items = list(mesh_or_scene.graph.geometry_nodes.items())
        except Exception:
            graph_items = []

        if graph_items:
            for geom_name, node_names in graph_items:
                geom = mesh_or_scene.geometry.get(geom_name)
                if not isinstance(geom, trimesh.Trimesh):
                    continue
                if len(geom.vertices) == 0 or len(geom.faces) == 0:
                    continue
                mesh = make_pyrender_mesh(geom)
                for node_name in node_names:
                    transform, _ = mesh_or_scene.graph.get(node_name)
                    scene.add(mesh, pose=np.asarray(transform, dtype=np.float64))
        else:
            for geom in mesh_or_scene.geometry.values():
                if not isinstance(geom, trimesh.Trimesh):
                    continue
                if len(geom.vertices) == 0 or len(geom.faces) == 0:
                    continue
                scene.add(make_pyrender_mesh(geom))
        return

    if not isinstance(mesh_or_scene, trimesh.Trimesh):
        return
    if len(mesh_or_scene.vertices) > 0 and len(mesh_or_scene.faces) > 0:
        scene.add(make_pyrender_mesh(mesh_or_scene))


def look_at_pose(eye, lookat, world_up=None):
    eye = np.asarray(eye, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = lookat - eye
    forward /= np.linalg.norm(forward) + 1e-12
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack([right, up, -forward])
    pose[:3, 3] = eye
    return pose, up


def sample_camera(rng, resolution):
    fov_deg = float(rng.uniform(45.0, 80.0))
    radius = float(rng.uniform(0.75, 2.0))
    azimuth_deg = float(rng.uniform(0.0, 360.0))
    elevation_deg = float(rng.uniform(-60.0, 60.0))

    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    eye = radius * np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    lookat = np.zeros(3, dtype=np.float64)
    pose, up = look_at_pose(eye, lookat)

    fov = math.radians(fov_deg)
    focal = 0.5 * resolution / math.tan(0.5 * fov)
    intrinsics = {
        "fx": focal,
        "fy": focal,
        "cx": resolution / 2.0,
        "cy": resolution / 2.0,
        "matrix": [
            [focal, 0.0, resolution / 2.0],
            [0.0, focal, resolution / 2.0],
            [0.0, 0.0, 1.0],
        ],
    }
    camera = {
        "fov": fov_deg,
        "radius": radius,
        "azimuth": azimuth_deg,
        "elevation": elevation_deg,
        "extrinsics": {
            "eye": eye.tolist(),
            "lookat": lookat.tolist(),
            "up": up.tolist(),
        },
        "intrinsics": intrinsics,
    }
    return pose, camera


def render_one_object(mesh_or_scene, save_dir, resolution, num_frames, rng, overwrite=False):
    frames_dir = save_dir / "frames"
    depth_dir = save_dir / "depth"
    mask_dir = save_dir / "mask"
    cameras_path = save_dir / "cameras.json"

    if cameras_path.exists() and not overwrite:
        return False

    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],
        ambient_light=[0.4, 0.4, 0.4],
    )
    add_to_scene(scene, mesh_or_scene)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(resolution, resolution)

    camera_records = []
    color = None
    depth = None
    mask = None
    try:
        for frame_idx in range(num_frames):
            pose, camera_record = sample_camera(rng, resolution)
            yfov = math.radians(camera_record["fov"])
            camera = pyrender.PerspectiveCamera(
                yfov=yfov,
                aspectRatio=1.0,
                znear=0.01,
                zfar=1000.0,
            )
            camera_node = scene.add(camera, pose=pose)
            scene.set_pose(light_node, pose)

            color, depth = renderer.render(scene, flags=RENDER_FLAGS)
            scene.remove_node(camera_node)

            frame_name = f"{frame_idx:03d}"
            Image.fromarray(color[..., :3].astype(np.uint8)).save(
                frames_dir / f"{frame_name}.jpg",
                quality=95,
            )
            depth = np.asarray(depth, dtype=np.float32)
            mask = depth > 0.0
            np.savez_compressed(depth_dir / f"{frame_name}.npz", depth=depth.astype(np.float16))
            np.savez_compressed(mask_dir / f"{frame_name}.npz", mask=mask)

            camera_record["file"] = f"{frame_name}.jpg"
            camera_records.append(camera_record)
    finally:
        try:
            renderer.delete()
        except Exception:
            pass
        for node in list(scene.mesh_nodes):
            try:
                scene.remove_node(node)
            except Exception:
                pass
        scene.clear()
        del renderer
        del scene
        del color
        del depth
        del mask
        gc.collect()

    with open(cameras_path, "w") as f:
        json.dump(
            {
                "height": resolution,
                "width": resolution,
                "frames": camera_records,
            },
            f,
            indent=2,
        )

    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=dataset_choices(),
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--debug_random_10",
        action="store_true",
        help="Only render 10 randomly sampled objects.",
    )
    parser.add_argument("--rank", type=int, default=0,
                        help="This worker's shard index (0-indexed). Ignored when --paths_file is given.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of shards across all workers. Ignored when --paths_file is given.")
    parser.add_argument("--gpu_id", type=int, default=-1,
                        help="GPU index to pin via EGL_DEVICE_ID (parsed early, before pyrender import).")
    parser.add_argument("--paths_file", type=str, default="",
                        help="JSON file with a list of model paths to process. When given, the dataset's "
                             "list_all_model_paths is not used and rank/world_size sharding is bypassed.")
    return parser.parse_args()


def main():
    args = parse_args()
    module_name, dataset = load_dataset_module(args.dataset_name)
    output_dataset_name = module_name

    rng = np.random.default_rng(args.seed)
    if args.paths_file:
        with open(args.paths_file, "r") as f:
            model_paths = [Path(p) for p in json.load(f)]
        print(
            f"[gpu={args.gpu_id}] paths_file={args.paths_file} -> {len(model_paths)} objects",
            flush=True,
        )
    else:
        model_paths = [Path(p) for p in dataset.list_all_model_paths()]
        if args.debug_random_10:
            indices = rng.choice(len(model_paths), size=min(10, len(model_paths)), replace=False)
            model_paths = [model_paths[i] for i in indices]

        if args.world_size > 1:
            if not (0 <= args.rank < args.world_size):
                raise ValueError(f"rank {args.rank} out of range for world_size {args.world_size}")
            total = len(model_paths)
            model_paths = model_paths[args.rank::args.world_size]
            print(
                f"[rank {args.rank}/{args.world_size} gpu={args.gpu_id}] "
                f"sharded {total} -> {len(model_paths)} objects",
                flush=True,
            )

    print(f"Rendering {len(model_paths)} objects from {output_dataset_name}", flush=True)
    for idx, model_path in enumerate(tqdm(model_paths, unit="obj")):
        object_name = object_id_from_path(model_path)
        save_dir = RENDER_ROOT / output_dataset_name / object_name
        mesh_or_scene = None
        try:
            mesh_or_scene = dataset.load_model(model_path)
            render_one_object(
                mesh_or_scene=mesh_or_scene,
                save_dir=save_dir,
                resolution=args.resolution,
                num_frames=args.num_frames,
                rng=rng,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"Error rendering {model_path}: {e}", flush=True)
        finally:
            if mesh_or_scene is not None:
                if isinstance(mesh_or_scene, trimesh.Scene):
                    try:
                        mesh_or_scene.geometry.clear()
                    except Exception:
                        pass
                del mesh_or_scene
            if (idx + 1) % 10 == 0:
                gc.collect()


if __name__ == "__main__":
    main()
