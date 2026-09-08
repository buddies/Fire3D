import os
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.path import Path
from tqdm import tqdm
import copy
try:
    from .room_regions import find_room_regions
except ImportError:
    from room_regions import find_room_regions

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

EXPORT_DIR = os.environ.get("FIRE3D_AI2THOR_EXPORT_ROOT", os.path.join(AI2THOR_ROOT, "exports"))


def configure_paths(dataset_dir, export_dir=None):
    """Configure ai2thor-hab paths before loading scene assets."""

    global DATASET_DIR, ASSETS_DIR, CONFIG_DIR, CONFIG_SCENE_DIR
    global ASSETS_OBJECT_DIR, ASSETS_STAGE_DIR, PROCTHOR_CONFIG_SCENE_DIR, EXPORT_DIR
    DATASET_DIR = os.path.abspath(os.path.expanduser(str(dataset_dir)))
    ASSETS_DIR = os.path.join(DATASET_DIR, "assets")
    CONFIG_DIR = os.path.join(DATASET_DIR, "configs")
    CONFIG_SCENE_DIR = os.path.join(CONFIG_DIR, "scenes")
    ASSETS_OBJECT_DIR = os.path.join(ASSETS_DIR, "objects")
    ASSETS_STAGE_DIR = os.path.join(ASSETS_DIR, "stages")
    PROCTHOR_CONFIG_SCENE_DIR = os.path.join(CONFIG_SCENE_DIR, "ProcTHOR")
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

def classify_layout_geoms(layout_geometries):
    """
    Classify layout geometries into walls, floors, and ceilings by bbox extent.
    Returns three dicts: wall_segments (geom_name -> list of ((x1,y1),(x2,y2))),
    floor_segments (geom_name -> (min_x, min_y, max_x, max_y)), ceiling_segments (same).
    Floor vs ceiling are distinguished by z: all floors share a similar low z, all ceilings a similar high z.
    """
    def get_bounds(geom):
        return geom.bounds

    wall_segments = {}
    floor_candidates = []  # (geom_name, bounds, z_center) for thin-along-z geoms

    for geom_name, geometry in layout_geometries.items():
        bounds = get_bounds(geometry)
        if bounds is None:
            continue
        min_pt = np.asarray(bounds[0])
        max_pt = np.asarray(bounds[1])
        extents = max_pt - min_pt
        thinnest_axis = np.argmin(extents)
        if thinnest_axis == 2:
            # Floor or ceiling (thin along z); classify by z later
            z_center = (min_pt[2] + max_pt[2]) / 2
            floor_candidates.append((geom_name, (min_pt, max_pt), z_center))
            continue
        # Wall: thin along x or y
        line = None
        if thinnest_axis == 0:
            x_center = (min_pt[0] + max_pt[0]) / 2
            line = ((float(x_center), float(min_pt[1])), (float(x_center), float(max_pt[1])))
        else:
            y_center = (min_pt[1] + max_pt[1]) / 2
            line = ((float(min_pt[0]), float(y_center)), (float(max_pt[0]), float(y_center)))
        wall_segments.setdefault(geom_name, []).append(line)

    # Classify floor vs ceiling by z: lower z cluster = floor, higher = ceiling
    floor_segments = {}
    ceiling_segments = {}
    if floor_candidates:
        z_centers = np.array([z for (_, _, z) in floor_candidates])
        z_mid = (z_centers.min() + z_centers.max()) / 2
        for geom_name, (min_pt, max_pt), z_center in floor_candidates:
            rect = (float(min_pt[0]), float(min_pt[1]), float(max_pt[0]), float(max_pt[1]))
            if z_center <= z_mid:
                floor_segments[geom_name] = rect
            else:
                ceiling_segments[geom_name] = rect

    return wall_segments, floor_segments, ceiling_segments

def visualize_layout(wall_segments, room_regions, vertices, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Draw room regions first (filled, semi-transparent) so walls render on top
    for i, poly in enumerate(room_regions):
        if len(poly) < 3:
            continue
        xy = np.array(poly)
        color = plt.cm.tab20(i % 20)
        ax.fill(xy[:, 0], xy[:, 1], facecolor=color, edgecolor="none", alpha=0.4)
    for (x1, y1), (x2, y2) in wall_segments:
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=1)
    # Draw vertices (wall endpoints and intersections)
    if vertices:
        vx = [p[0] for p in vertices]
        vy = [p[1] for p in vertices]
        ax.scatter(vx, vy, c="red", s=8, zorder=5, label="vertices")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Layout (walls + room regions + vertices, top-down)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# When testing bbox center against room polygon: positive radius enlarges the polygon
# so points slightly outside the boundary still count as inside (same units as scene, e.g. meters).


def bbox_in_room(bbox, room_region):
    """
    Check whether an object's 3D bbox (from trimesh .bounds.tolist()) lies in the room.
    bbox is [[min_x, min_y, min_z], [max_x, max_y, max_z]]. We use the x-y footprint and
    test if the center of the xy bbox is inside the room_region polygon. A positive
    radius enlarges the polygon so points near the outer bound still count.
    """
    BBOX_IN_ROOM_BOUNDARY_TOL = 0.05
    if not bbox or not room_region or len(room_region) < 3:
        return False
    min_pt, max_pt = bbox[0], bbox[1]
    min_x, min_y = min_pt[0], min_pt[1]
    max_x, max_y = max_pt[0], max_pt[1]
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    return Path(np.asarray(room_region)).contains_point((cx, cy), radius=BBOX_IN_ROOM_BOUNDARY_TOL) or Path(np.asarray(room_region)).contains_point((cx, cy), radius=-BBOX_IN_ROOM_BOUNDARY_TOL)

def decompose_scene_to_rooms(room_regions, room_geoms, layout_geometries, object_instances, bbox_dict):
    rooms = []
    rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    for room_idx, room_geom in room_geoms.items():
        room_layout = trimesh.Scene()
        room_region = room_regions[room_idx]

        for wall_geom_name in room_geom["walls"]:
            wall_geom = layout_geometries[wall_geom_name]
            room_layout = add_object_to_scene(room_layout, wall_geom, wall_geom_name)

        for floor_geom_name in room_geom["floors"]:
            floor_geom = layout_geometries[floor_geom_name]
            room_layout = add_object_to_scene(room_layout, floor_geom, floor_geom_name)

        for ceiling_geom_name in room_geom["ceilings"]:
            ceiling_geom = layout_geometries[ceiling_geom_name]
            room_layout = add_object_to_scene(room_layout, ceiling_geom, ceiling_geom_name)

        for index, object_instance in enumerate(object_instances):
            object_name = object_instance["template_name"]
            if "objects/Doorway_" in object_name:
                object_name = object_name.replace("_open", "")
            object_node_name = f"object_{index}_{object_name.replace('/', '_')}"

            bbox = bbox_dict[object_node_name]

            if bbox_in_room(bbox, room_region):

                object_glb_path = os.path.join(ASSETS_DIR, object_name+".glb")
                if not os.path.exists(object_glb_path):
                    print(f"Object {object_name} not found")
                    continue

                translation = object_instance["translation"]
                rotation = object_instance["rotation"]
                non_uniform_scale = object_instance["non_uniform_scale"]
                motion_type = object_instance["motion_type"]

                transform_matrix = compose_transform(translation, rotation, non_uniform_scale)
                room_layout = add_object_to_scene(room_layout, object_glb_path, object_node_name, rot_x_90 @ transform_matrix)

        rooms.append(room_layout)
    return rooms





def export_scene(scene_name, return_supp=False):

    scene_json_path = os.path.join(CONFIG_SCENE_DIR, scene_name+".scene_instance.json")
    stage_glb_path = os.path.join(ASSETS_STAGE_DIR, scene_name+".glb")

    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)

    rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

    scene = trimesh.Scene()
    layout_rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0])
    scene, layout_geometries = add_object_to_scene(scene, stage_glb_path, "layout", rot_x_90 @ layout_rotation_matrix, return_internal_geometries=True)


    object_instances = scene_data["object_instances"]

    bbox_dict = {}

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
        scene, bbox = add_object_to_scene(scene, object_glb_path, object_node_name, rot_x_90 @ transform_matrix, return_bbox=True)
        bbox_dict[object_node_name] = bbox
    if return_supp:
        return scene, {
            "bbox_dict": bbox_dict,
            "layout_geometries": layout_geometries,
            "object_instances": object_instances,
        }
    return scene

def export_rooms(scene_name):
    scene, supp = export_scene(scene_name, return_supp=True)
    layout_geometries = supp["layout_geometries"]
    object_instances = supp["object_instances"]
    bbox_dict = supp["bbox_dict"]

    wall_segments_dict, floor_segments_dict, ceiling_segments_dict = classify_layout_geoms(layout_geometries)
    wall_segments = [line for lines in wall_segments_dict.values() for line in lines]

    if not wall_segments:
        print("No walls found")
        assert False, "No walls found"
    print(f"Found {len(wall_segments)} wall segments")

    room_regions, vertices, room_geoms = find_room_regions(
        wall_segments_dict, floor_segments_dict, ceiling_segments_dict
    )

    if len(room_regions) == 0:
        print("No room regions found")
        assert False, "No room regions found"
    print(f"Found {len(room_regions)} room regions, {len(vertices)} vertices")

    # layout_visualize_path = os.path.join(EXPORT_DIR, scene_save_name+".layout.png")
    # visualize_layout(wall_segments, room_regions, vertices, layout_visualize_path)

    rooms = decompose_scene_to_rooms(room_regions, room_geoms, layout_geometries, object_instances, bbox_dict)


    return rooms

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

def export_rooms_with_canonical_meshes(scene_name):

    scene_json_path = os.path.join(CONFIG_SCENE_DIR, scene_name+".scene_instance.json")
    stage_glb_path = os.path.join(ASSETS_STAGE_DIR, scene_name+".glb")

    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)

    rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

    scene = trimesh.Scene()

    layout_rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0])
    layout_transform_matrix = rot_x_90 @ layout_rotation_matrix
    scene, layout_geometries = add_object_to_scene(scene, stage_glb_path, "layout", layout_transform_matrix, return_internal_geometries=True)
    wall_segments_dict, floor_segments_dict, ceiling_segments_dict = classify_layout_geoms(layout_geometries)
    room_regions, vertices, room_geoms = find_room_regions(
        wall_segments_dict, floor_segments_dict, ceiling_segments_dict
    )

    room_geoms_dict = {}

    for room_idx, room_geom in room_geoms.items():
        room_geoms_dict[room_idx] = {}

        room_layout = trimesh.Scene()
        for wall_geom_name in room_geom["walls"]:
            wall_geom = layout_geometries[wall_geom_name].copy()
            room_layout = add_object_to_scene(room_layout, wall_geom, wall_geom_name)
        for floor_geom_name in room_geom["floors"]:
            floor_geom = layout_geometries[floor_geom_name].copy()
            room_layout = add_object_to_scene(room_layout, floor_geom, floor_geom_name)
        for ceiling_geom_name in room_geom["ceilings"]:
            ceiling_geom = layout_geometries[ceiling_geom_name].copy()
            room_layout = add_object_to_scene(room_layout, ceiling_geom, ceiling_geom_name)

        # Keep backgrounds in the render/scene frame used by the transform PKLs.
        room_geoms_dict[room_idx]["bg"] = {
            "mesh": room_layout.copy(),
            "transform": np.eye(4),
            "scale": np.eye(4)
        }

    object_instances = scene_data["object_instances"]

    bbox_dict = {}
    mesh_info_dict = {}

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

        bbox = object.copy().apply_transform(transform_matrix).bounds.tolist()

        bbox_dict[object_node_name] = bbox

        mesh_info_dict[object_node_name] = {
            "mesh": object.copy(),
            "transform": transform_matrix,
            "scale": scale_matrix
        }

    for room_idx, room_geom in room_geoms.items():
        room_region = room_regions[room_idx]
        for index, object_instance in enumerate(copy.deepcopy(object_instances)):
            object_name = object_instance["template_name"]
            if "objects/Doorway_" in object_name:
                object_name = object_name.replace("_open", "")
            object_node_name = f"object_{index}_{object_name.replace('/', '_')}"

            bbox = bbox_dict[object_node_name]

            if bbox_in_room(bbox, room_region):
                room_geoms_dict[room_idx][object_node_name] = copy.deepcopy(mesh_info_dict[object_node_name])

    for room_idx, room_geom in room_geoms.items():
        room_i_geoms_dict = room_geoms_dict[room_idx]
        for mesh_name, mesh_info in room_i_geoms_dict.items():
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
            if mesh_name == "bg":
                orientation_matrix = np.eye(4)

            mesh_copy.apply_transform(orientation_matrix)

            center, scale = normalize_mesh(trimesh.util.concatenate(mesh_copy.dump()))
            normalization_center_matrix = np.eye(4)
            normalization_center_matrix[:3, 3] = -center
            normalization_scale_matrix = np.eye(4)
            normalization_scale_matrix[np.diag_indices(3)] = scale
            mesh_copy.apply_transform(normalization_center_matrix)
            mesh_copy.apply_transform(normalization_scale_matrix)

            room_i_geoms_dict[mesh_name]["canonical_mesh"] = mesh_copy

            total_transform = transform @ np.linalg.inv(normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix @ scale_matrix)
            room_i_geoms_dict[mesh_name]["total_transform"] = total_transform
            room_i_geoms_dict[mesh_name]["transformed_mesh"] = mesh_copy.copy().apply_transform(total_transform)


        room_geoms_dict[room_idx] = room_i_geoms_dict

    return room_regions, room_geoms_dict


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

def get_all_procthor_scenes():

    scene_names = [
        os.path.join("ProcTHOR", idx, scene_instance_json_name.replace(".scene_instance.json", "")) \
        for idx in sorted(os.listdir(PROCTHOR_CONFIG_SCENE_DIR)) \
        for scene_instance_json_name in sorted(os.listdir(os.path.join(PROCTHOR_CONFIG_SCENE_DIR, idx))) \
        if scene_instance_json_name.endswith(".scene_instance.json")
    ]

    print(f"Found {len(scene_names)} scenes in ProcTHOR")

    return scene_names

def export_rooms_layout_from_scene(scene_name):

    scene_json_path = os.path.join(CONFIG_SCENE_DIR, scene_name+".scene_instance.json")
    stage_glb_path = os.path.join(ASSETS_STAGE_DIR, scene_name+".glb")

    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)

    rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

    scene = trimesh.Scene()

    layout_rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0])
    layout_transform_matrix = rot_x_90 @ layout_rotation_matrix
    scene, layout_geometries = add_object_to_scene(scene, stage_glb_path, "layout", layout_transform_matrix, return_internal_geometries=True)
    wall_segments_dict, floor_segments_dict, ceiling_segments_dict = classify_layout_geoms(layout_geometries)
    room_regions, vertices, room_geoms = find_room_regions(
        wall_segments_dict, floor_segments_dict, ceiling_segments_dict
    )

    room_geoms_dict = {}

    for room_idx, room_geom in room_geoms.items():
        room_geoms_dict[room_idx] = {}

        room_layout = trimesh.Scene()
        for wall_geom_name in room_geom["walls"]:
            wall_geom = layout_geometries[wall_geom_name]
            room_layout = add_object_to_scene(room_layout, wall_geom, wall_geom_name)
        for floor_geom_name in room_geom["floors"]:
            floor_geom = layout_geometries[floor_geom_name]
            room_layout = add_object_to_scene(room_layout, floor_geom, floor_geom_name)
        for ceiling_geom_name in room_geom["ceilings"]:
            ceiling_geom = layout_geometries[ceiling_geom_name]
            room_layout = add_object_to_scene(room_layout, ceiling_geom, ceiling_geom_name)

        # Keep backgrounds in the render/scene frame used by the transform PKLs.
        room_geoms_dict[room_idx]["bg"] = {
            "mesh": room_layout,
            "transform": np.eye(4),
            "scale": np.eye(4)
        }

    # object_instances = scene_data["object_instances"]

    # bbox_dict = {}
    # mesh_info_dict = {}

    # for index, object_instance in enumerate(object_instances):
    #     object_name = object_instance["template_name"]
    #     if "objects/Doorway_" in object_name:
    #         object_name = object_name.replace("_open", "")
    #     object_glb_path = os.path.join(ASSETS_DIR, object_name+".glb")
    #     if not os.path.exists(object_glb_path):
    #         print(f"Object {object_name} not found")
    #         continue

    #     translation = object_instance["translation"]
    #     rotation = object_instance["rotation"]
    #     non_uniform_scale = object_instance["non_uniform_scale"]
    #     motion_type = object_instance["motion_type"]

    #     # transform_matrix = compose_transform(translation, rotation, non_uniform_scale)
    #     scale_matrix = np.eye(4)
    #     scale_matrix[np.diag_indices(3)] = non_uniform_scale
    #     translate_matrix = np.eye(4)
    #     translate_matrix[:3, 3] = translation
    #     angles = trimesh.transformations.euler_from_quaternion(rotation, axes='sxyz')
    #     rotation_matrix = trimesh.transformations.euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
    #     transform_matrix = rot_x_90 @ translate_matrix @ rotation_matrix @ scale_matrix

    #     object_node_name = f"object_{index}_{object_name.replace('/', '_')}"

    #     object = trimesh.load(object_glb_path)

    #     bbox = object.copy().apply_transform(transform_matrix).bounds.tolist()

    #     bbox_dict[object_node_name] = bbox

    #     mesh_info_dict[object_node_name] = {
    #         "mesh": object,
    #         "transform": transform_matrix,
    #         "scale": scale_matrix
    #     }

    # for room_idx, room_geom in room_geoms.items():
    #     room_region = room_regions[room_idx]
    #     for index, object_instance in enumerate(object_instances):
    #         object_name = object_instance["template_name"]
    #         if "objects/Doorway_" in object_name:
    #             object_name = object_name.replace("_open", "")
    #         object_node_name = f"object_{index}_{object_name.replace('/', '_')}"

    #         bbox = bbox_dict[object_node_name]

    #         if bbox_in_room(bbox, room_region):
    #             room_geoms_dict[room_idx][object_node_name] = mesh_info_dict[object_node_name]

    for room_idx, room_geom in room_geoms.items():
        room_i_geoms_dict = room_geoms_dict[room_idx]
        for mesh_name, mesh_info in room_i_geoms_dict.items():
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
            if mesh_name == "bg":
                orientation_matrix = np.eye(4)

            mesh_copy.apply_transform(orientation_matrix)

            center, scale = normalize_mesh(trimesh.util.concatenate(mesh_copy.dump()))
            normalization_center_matrix = np.eye(4)
            normalization_center_matrix[:3, 3] = -center
            normalization_scale_matrix = np.eye(4)
            normalization_scale_matrix[np.diag_indices(3)] = scale
            mesh_copy.apply_transform(normalization_center_matrix)
            mesh_copy.apply_transform(normalization_scale_matrix)

            room_i_geoms_dict[mesh_name]["canonical_mesh"] = mesh_copy

            total_transform = transform @ np.linalg.inv(normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix @ scale_matrix)
            room_i_geoms_dict[mesh_name]["total_transform"] = total_transform
            room_i_geoms_dict[mesh_name]["transformed_mesh"] = mesh_copy.copy().apply_transform(total_transform)


        room_geoms_dict[room_idx] = room_i_geoms_dict

    return room_regions, room_geoms_dict
