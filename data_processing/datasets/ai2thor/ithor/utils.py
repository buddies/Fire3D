import os
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.path import Path
from tqdm import tqdm
import copy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
AI2THOR_ROOT = os.environ.get(
    "FIRE3D_AI2THOR_ROOT",
    os.path.join(REPO_ROOT, "data", "training_scenes", "ai2thor-hab"),
)
DATASET_DIR = os.environ.get(
    "FIRE3D_AI2THOR_HAB_ROOT",
    os.path.join(AI2THOR_ROOT, "ai2thor-hab"),
)
ASSETS_DIR = os.path.join(DATASET_DIR, "assets")
CONFIG_DIR = os.path.join(DATASET_DIR, "configs")

CONFIG_SCENE_DIR = os.path.join(CONFIG_DIR, "scenes")
ASSETS_OBJECT_DIR = os.path.join(ASSETS_DIR, "objects")
ASSETS_STAGE_DIR = os.path.join(ASSETS_DIR, "stages")

PROCTHOR_CONFIG_SCENE_DIR = os.path.join(CONFIG_SCENE_DIR, "ProcTHOR")
ITHOR_CONFIG_SCENE_DIR = os.path.join(CONFIG_SCENE_DIR, "iTHOR")

EXPORT_DIR = os.environ.get("FIRE3D_AI2THOR_EXPORT_ROOT", os.path.join(AI2THOR_ROOT, "exports"))


def configure_paths(dataset_dir, export_dir=None):
    """Configure ai2thor-hab paths before loading scene assets."""

    global DATASET_DIR, ASSETS_DIR, CONFIG_DIR, CONFIG_SCENE_DIR
    global ASSETS_OBJECT_DIR, ASSETS_STAGE_DIR, PROCTHOR_CONFIG_SCENE_DIR
    global ITHOR_CONFIG_SCENE_DIR, EXPORT_DIR
    DATASET_DIR = os.path.abspath(os.path.expanduser(str(dataset_dir)))
    ASSETS_DIR = os.path.join(DATASET_DIR, "assets")
    CONFIG_DIR = os.path.join(DATASET_DIR, "configs")
    CONFIG_SCENE_DIR = os.path.join(CONFIG_DIR, "scenes")
    ASSETS_OBJECT_DIR = os.path.join(ASSETS_DIR, "objects")
    ASSETS_STAGE_DIR = os.path.join(ASSETS_DIR, "stages")
    PROCTHOR_CONFIG_SCENE_DIR = os.path.join(CONFIG_SCENE_DIR, "ProcTHOR")
    ITHOR_CONFIG_SCENE_DIR = os.path.join(CONFIG_SCENE_DIR, "iTHOR")
    if export_dir is not None:
        EXPORT_DIR = os.path.abspath(os.path.expanduser(str(export_dir)))

def save_glb(mesh_or_scene, save_path):
    rot_x_90 = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
    mesh_or_scene_copy = mesh_or_scene.copy()
    mesh_or_scene_copy.apply_transform(rot_x_90)
    trimesh.exchange.export.export_mesh(mesh_or_scene_copy, save_path)


def compose_transform(translation, rotation, non_uniform_scale):
    angles = trimesh.transformations.euler_from_quaternion(rotation, "sxyz")

    transform_matrix = trimesh.transformations.compose_matrix(
        scale=non_uniform_scale,
        angles=angles,
        translate=translation,
    )

    return transform_matrix

def add_object_to_scene(
    scene,
    object_or_path,
    parent_node_name,
    transform_matrix=None,
    return_internal_geometries=False,
    return_bbox=False
):
    if isinstance(object_or_path, str):
        object = trimesh.load(object_or_path)
    else:
        object = object_or_path
    # if transform_matrix is not None:
        # object.apply_transform(transform_matrix)
    # scene.add_geometry(object)

    if transform_matrix is None:
        transform_matrix = np.eye(4)

    if return_internal_geometries:
        internal_geometries = {}

    if return_bbox:
        bbox = object.copy().apply_transform(transform_matrix).bounds.tolist()
    else:
        bbox = None

    scene.graph.update(frame_to=parent_node_name, matrix=transform_matrix)

    if isinstance(object, trimesh.Scene):
        for geom_name, mesh_part in object.geometry.items():
            nodes_for_this_geometry = object.graph.geometry_nodes.get(geom_name, [])

            for i, node_name_in_subscene in enumerate(nodes_for_this_geometry):
                internal_transform, _ = object.graph.get(node_name_in_subscene)
                scene.add_geometry(
                    mesh_part,
                    geom_name=f"{parent_node_name}_{geom_name}_{i}",
                    transform=internal_transform,
                    parent_node_name=parent_node_name
                )


            if return_internal_geometries:
                internal_geom = trimesh.Scene()
                for i, node_name_in_subscene in enumerate(nodes_for_this_geometry):
                    internal_transform, _ = object.graph.get(node_name_in_subscene)
                    internal_geom.add_geometry(
                        mesh_part,
                        geom_name=f"{parent_node_name}_{geom_name}_{i}",
                        transform=internal_transform,
                        # parent_node_name=parent_node_name
                    )
                internal_geom = internal_geom.apply_transform(transform_matrix)
                internal_geometries[f"{parent_node_name}_{geom_name}"] = internal_geom
    else: #
        scene.add_geometry(
            object,
            geom_name=parent_node_name + "_geom",
            parent_node_name=parent_node_name
        )

    if return_bbox and return_internal_geometries:
        return scene, bbox, internal_geometries
    elif return_bbox:
        return scene, bbox
    elif return_internal_geometries:
        return scene, internal_geometries
    else:
        return scene


def export_scene(scene_name, return_supp=False):

    scene_json_path = os.path.join(CONFIG_SCENE_DIR, scene_name+".scene_instance.json")
    stage_glb_path = os.path.join(ASSETS_STAGE_DIR, scene_name+".glb")

    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)

    rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

    scene = trimesh.Scene()
    scene = add_object_to_scene(scene, stage_glb_path, "layout", rot_x_90)


    object_instances = scene_data["object_instances"]


    for index, object_instance in enumerate(object_instances):
        object_name = object_instance["template_name"]
        if "objects/Doorway_" in object_name:
            object_name = object_name.replace("_open", "")
        object_glb_path = os.path.join(ASSETS_DIR, object_name+".glb")
        if not os.path.exists(object_glb_path):
            print(f"Object {object_name} not found")
            continue

        translation = object_instance["translation"]
        rotation = object_instance["rotation"]
        non_uniform_scale = object_instance["non_uniform_scale"]
        motion_type = object_instance["motion_type"]

        transform_matrix = compose_transform(translation, rotation, non_uniform_scale)
        object_node_name = f"object_{index}_{object_name.replace('/', '_')}"
        scene = add_object_to_scene(scene, object_glb_path, object_node_name, rot_x_90 @ transform_matrix)
    if return_supp:
        return scene, {
            "object_instances": object_instances,
        }
    return scene


def normalize_mesh(mesh: trimesh.Trimesh):
    """
    Preprocess the input mesh.
    """
    vertices = mesh.vertices
    vertices_min = vertices.min(axis=0)
    vertices_max = vertices.max(axis=0)
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()
    # vertices = (vertices - center) * scale

    mesh_transform = (center, scale)
    # assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
    return mesh_transform

def export_scene_with_canonical_meshes(scene_name):

    scene_json_path = os.path.join(CONFIG_SCENE_DIR, scene_name+".scene_instance.json")
    stage_glb_path = os.path.join(ASSETS_STAGE_DIR, scene_name+".glb")

    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)

    scene_geoms_dict = {}

    bg = trimesh.Scene()
    rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    bg = add_object_to_scene(bg, stage_glb_path, "layout", rot_x_90)

    scene_geoms_dict["bg"] = {
        "mesh": bg.copy().apply_transform(np.linalg.inv(rot_x_90)),
        "transform": rot_x_90,
        "scale": np.eye(4)
    }

    object_instances = scene_data["object_instances"]

    for index, object_instance in enumerate(copy.deepcopy(object_instances)):
        object_name = object_instance["template_name"]
        if "objects/Doorway_" in object_name:
            object_name = object_name.replace("_open", "")
        object_glb_path = os.path.join(ASSETS_DIR, object_name+".glb")
        if not os.path.exists(object_glb_path):
            print(f"Object {object_name} not found")
            continue

        translation = object_instance["translation"]
        rotation = object_instance["rotation"]
        non_uniform_scale = object_instance["non_uniform_scale"]
        motion_type = object_instance["motion_type"]

        # transform_matrix = compose_transform(translation, rotation, non_uniform_scale)
        scale_matrix = np.eye(4)
        scale_matrix[np.diag_indices(3)] = non_uniform_scale
        translate_matrix = np.eye(4)
        translate_matrix[:3, 3] = translation
        angles = trimesh.transformations.euler_from_quaternion(rotation, axes='sxyz')
        rotation_matrix = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
        transform_matrix = rot_x_90 @ translate_matrix @ rotation_matrix @ scale_matrix

        object_node_name = f"object_{index}_{object_name.replace('/', '_')}"

        object = trimesh.load(object_glb_path)

        scene_geoms_dict[object_node_name] = {
            "mesh": object.copy(),
            "transform": transform_matrix,
            "scale": scale_matrix
        }

    for mesh_name, mesh_info in scene_geoms_dict.items():
        mesh = mesh_info["mesh"]
        transform = mesh_info["transform"]
        scale_matrix = mesh_info["scale"]

        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(scale_matrix)

        orientation_matrix = np.array([
            [-1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

        mesh_copy.apply_transform(orientation_matrix)

        center, scale = normalize_mesh(trimesh.util.concatenate(mesh_copy.dump()))
        normalization_center_matrix = np.eye(4)
        normalization_center_matrix[:3, 3] = -center
        normalization_scale_matrix = np.eye(4)
        normalization_scale_matrix[np.diag_indices(3)] = scale
        mesh_copy.apply_transform(normalization_center_matrix)
        mesh_copy.apply_transform(normalization_scale_matrix)

        scene_geoms_dict[mesh_name]["canonical_mesh"] = mesh_copy

        total_transform = transform @ np.linalg.inv(normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix @ scale_matrix)
        scene_geoms_dict[mesh_name]["total_transform"] = total_transform
        scene_geoms_dict[mesh_name]["transformed_mesh"] = mesh_copy.copy().apply_transform(total_transform)

    return scene_geoms_dict


def export_all_canonical_objects():
    all_object_paths = sorted([os.path.join(ASSETS_OBJECT_DIR, object_name) for object_name in os.listdir(ASSETS_OBJECT_DIR)])
    print(f"Found {len(all_object_paths)} objects")

    canonical_objects_dict = {}

    for object_path in tqdm(all_object_paths, desc="Exporting canonical objects"):
        object_name = os.path.basename(object_path)
        mesh = trimesh.load(object_path)

        orientation_matrix = np.array([
            [-1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])
        mesh.apply_transform(orientation_matrix)

        center, scale = normalize_mesh(trimesh.util.concatenate(mesh.dump()))
        normalization_center_matrix = np.eye(4)
        normalization_center_matrix[:3, 3] = -center
        normalization_scale_matrix = np.eye(4)
        normalization_scale_matrix[np.diag_indices(3)] = scale
        mesh.apply_transform(normalization_center_matrix)
        mesh.apply_transform(normalization_scale_matrix)

        canonical_objects_dict[object_name] = {
            "mesh": mesh,
            "transform": np.linalg.inv(normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix)
        }

    return canonical_objects_dict


def export_all_object_names():
    object_names = sorted([os.path.splitext(os.path.basename(object_name))[0] for object_name in os.listdir(ASSETS_OBJECT_DIR)])
    print(f"Found {len(object_names)} objects")

    return object_names

def get_object_path_from_name(object_name):
    object_path = os.path.join(ASSETS_OBJECT_DIR, object_name+".glb")
    return object_path

def get_canonical_object_from_name(object_name):
    object_path = get_object_path_from_name(object_name)
    print("loading object from path: ", object_path)
    mesh = trimesh.load(object_path)
    print("loaded object from path: ", object_path)

    orientation_matrix = np.array([
        [-1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1]
    ])
    mesh.apply_transform(orientation_matrix)

    center, scale = normalize_mesh(trimesh.util.concatenate(mesh.dump()))
    normalization_center_matrix = np.eye(4)
    normalization_center_matrix[:3, 3] = -center
    normalization_scale_matrix = np.eye(4)
    normalization_scale_matrix[np.diag_indices(3)] = scale
    mesh.apply_transform(normalization_center_matrix)
    mesh.apply_transform(normalization_scale_matrix)

    return {
        "mesh": mesh,
        "transform": np.linalg.inv(normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix)
    }

def get_all_ithor_scenes():

    scene_names = [
        os.path.join("iTHOR", scene_instance_json_name.replace(".scene_instance.json", "")) \
        for scene_instance_json_name in sorted(os.listdir(ITHOR_CONFIG_SCENE_DIR)) \
        if scene_instance_json_name.endswith(".scene_instance.json")
    ]

    print(f"Found {len(scene_names)} scenes in iTHOR")

    return scene_names
