import os
import argparse
import trimesh
import networkx as nx
from easydict import EasyDict as edict
import json
import gc
import copy
import numpy as np
from subprocess import DEVNULL, call
import tempfile
import pickle
from typing import Optional
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = os.environ.get(
    "FIRE3D_IMAGINARIUM_ROOT",
    str(REPO_ROOT / "data/training_scenes/Imaginarium"),
)
BLENDER_PATH = os.environ.get(
    "FIRE3D_BLENDER",
    str(REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender"),
)

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        raise FileNotFoundError(
            f"Blender is missing at {BLENDER_PATH}; run scripts/install_blender.sh "
            "or set FIRE3D_BLENDER"
        )


def save_glb(mesh_or_scene, save_path):
    rot_x_90 = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
    mesh_or_scene_copy = mesh_or_scene.copy()
    mesh_or_scene_copy.apply_transform(rot_x_90)
    trimesh.exchange.export.export_mesh(mesh_or_scene_copy, save_path)


def _to_glb(file_path, output_dir):
    to_glb_script_path = os.path.join(os.path.dirname(__file__), 'to_glb.py')
    file_path = os.path.expanduser(file_path)

    # Note: for `.blend` we must pass it to `-b` so Blender opens it before running `-P`.
    args = [BLENDER_PATH, '-b']
    if file_path.endswith('.blend'):
        args.append(file_path)

    args += [
        '-P', to_glb_script_path,
        '--',
        '--object', file_path,
        '--output_folder', output_dir,
    ]

    call(args)
    # call(args, stdout=DEVNULL, stderr=DEVNULL)


def robust_subscene(scene, node_name):
    """
    Build a subscene rooted at `node_name` without using trimesh.Scene.subscene.
    Keeps only ancestors + node + descendants to preserve transforms.
    """
    scene_work = copy.deepcopy(scene)
    graph = scene_work.graph.to_networkx()
    if node_name not in graph.nodes:
        return trimesh.Scene()

    ancestor_nodes = set(nx.ancestors(graph, node_name))
    descendant_nodes = set(nx.descendants(graph, node_name))
    keep_nodes = ancestor_nodes | {node_name} | descendant_nodes
    nodes_to_delete = [n for n in graph.nodes if n not in keep_nodes]

    for del_node_name in nodes_to_delete:
        if del_node_name not in scene_work.graph.nodes:
            continue
        cur_graph = scene_work.graph.to_networkx()
        if del_node_name not in cur_graph.nodes:
            continue
        del_descendants = sorted(nx.descendants(cur_graph, del_node_name))
        del_subtree_nodes = [del_node_name] + del_descendants
        del_subtree_graph = cur_graph.subgraph(del_subtree_nodes)
        try:
            del_order = list(reversed(list(nx.topological_sort(del_subtree_graph))))
        except Exception:
            del_order = list(reversed(del_subtree_nodes))

        for target_node_name in del_order:
            if target_node_name not in scene_work.graph.nodes:
                continue
            del_geom_name = None
            try:
                del_geom_name = scene_work.graph.transforms.node_data[target_node_name].get('geometry')
            except Exception:
                del_geom_name = None

            scene_work.graph.transforms.remove_node(target_node_name)

            if del_geom_name and del_geom_name in scene_work.geometry:
                still_referenced = False
                for nd in getattr(scene_work.graph.transforms, "node_data", {}).values():
                    if nd.get("geometry") == del_geom_name:
                        still_referenced = True
                        break
                if not still_referenced:
                    del scene_work.geometry[del_geom_name]

    # Flatten transform chain so `world -> node_name` is direct and
    # intermediate ancestor axis nodes are removed.
    base_frame = scene_work.graph.base_frame
    if node_name in scene_work.graph.nodes and node_name != base_frame:
        world_to_node, _ = scene_work.graph.get(frame_to=node_name, frame_from=base_frame)
        node_geom = scene_work.graph.transforms.node_data.get(node_name, {}).get("geometry")
        update_kwargs = {"matrix": world_to_node}
        if node_geom:
            update_kwargs["geometry"] = node_geom
        scene_work.graph.update(frame_to=node_name, frame_from=base_frame, **update_kwargs)

        removable_ancestors = [a for a in ancestor_nodes if a != base_frame]
        if removable_ancestors:
            try:
                anc_graph = scene_work.graph.to_networkx().subgraph(removable_ancestors)
                remove_order = list(reversed(list(nx.topological_sort(anc_graph))))
            except Exception:
                remove_order = list(reversed(removable_ancestors))

            for anc in remove_order:
                if anc in scene_work.graph.nodes:
                    scene_work.graph.transforms.remove_node(anc)

    # Prune any now-unreferenced geometry after transform flattening.
    referenced_geoms = set()
    for nd in getattr(scene_work.graph.transforms, "node_data", {}).values():
        g = nd.get("geometry")
        if g:
            referenced_geoms.add(g)
    for g in list(scene_work.geometry.keys()):
        if g not in referenced_geoms:
            del scene_work.geometry[g]

    return scene_work

def prune_descendant_nodes(node_mesh, node_name, object_names):
    """
    Keep `node_name` and prune its descendants from `node_mesh`.
    Mirrors existing deletion behavior.
    """
    node_names_to_delete = []
    node_mesh_graph = node_mesh.graph.to_networkx()
    if node_name in node_mesh_graph.nodes:
        descendants_node_names = sorted(nx.descendants(node_mesh_graph, node_name))
        # Delete the entire descendant subtree (including auto-suffixed instance
        # nodes), but keep the root `node_name` itself.
        try:
            subtree_nodes = [node_name] + descendants_node_names
            subtree = node_mesh_graph.subgraph(subtree_nodes)
            topo = list(nx.topological_sort(subtree))
            node_names_to_delete = [n for n in reversed(topo) if n != node_name]
        except Exception:
            node_names_to_delete = list(reversed(descendants_node_names))

    # Remove descendant nodes. For each target node, recursively delete
    # its full subtree in leaf-to-root order.
    for del_node_name in node_names_to_delete:
        if del_node_name not in object_names:
            continue
        if del_node_name not in node_mesh.graph.nodes:
            continue

        # Rebuild graph each loop because prior deletions mutate the scene graph.
        cur_graph = node_mesh.graph.to_networkx()
        if del_node_name not in cur_graph.nodes:
            continue
        del_descendants = sorted(nx.descendants(cur_graph, del_node_name))
        del_subtree_nodes = [del_node_name] + del_descendants
        del_subtree_graph = cur_graph.subgraph(del_subtree_nodes)
        try:
            del_order = list(reversed(list(nx.topological_sort(del_subtree_graph))))
        except Exception:
            del_order = list(reversed(del_subtree_nodes))

        for target_node_name in del_order:
            if target_node_name not in node_mesh.graph.nodes:
                continue

            # Capture referenced geometry before removing the node.
            del_geom_name = None
            try:
                del_geom_name = node_mesh.graph.transforms.node_data[target_node_name].get('geometry')
            except Exception:
                del_geom_name = None

            node_mesh.graph.transforms.remove_node(target_node_name)

            # If that geometry is now unreferenced by any remaining node, prune it.
            if del_geom_name and del_geom_name in node_mesh.geometry:
                still_referenced = False
                for nd in getattr(node_mesh.graph.transforms, "node_data", {}).values():
                    if nd.get("geometry") == del_geom_name:
                        still_referenced = True
                        break
                if not still_referenced:
                    del node_mesh.geometry[del_geom_name]


def _fit_scene_to_bounds(template_scene, target_min, target_max, eps=1e-8):
    """
    Scale and translate a trimesh.Scene so its world AABB matches [target_min, target_max].
    Keeps scene geometry/materials/textures intact.
    """
    fitted = template_scene.copy()
    bounds = fitted.bounds
    if bounds is None:
        return None

    target_min = np.asarray(target_min, dtype=float)
    target_max = np.asarray(target_max, dtype=float)

    src_min, src_max = bounds
    src_size = src_max - src_min
    dst_size = target_max - target_min

    # Cube templates are axis-aligned here, so one affine map is enough:
    # translate source min to origin -> scale per axis -> translate to target min.
    scale = np.ones(3, dtype=float)
    for axis in range(3):
        if abs(src_size[axis]) >= eps:
            scale[axis] = dst_size[axis] / src_size[axis]

    t_to_src_min = np.eye(4, dtype=float)
    t_to_src_min[:3, 3] = -src_min
    s_mat = np.eye(4, dtype=float)
    s_mat[0, 0], s_mat[1, 1], s_mat[2, 2] = scale[0], scale[1], scale[2]
    t_to_dst_min = np.eye(4, dtype=float)
    t_to_dst_min[:3, 3] = target_min
    fitted.apply_transform(t_to_dst_min @ s_mat @ t_to_src_min)

    # Final snap translation: remove tiny FP drift without extra scaling.
    fitted_bounds = fitted.bounds
    if fitted_bounds is not None:
        t_snap = np.eye(4, dtype=float)
        t_snap[:3, 3] = target_min - fitted_bounds[0]
        fitted.apply_transform(t_snap)

    # Post-check: non-degenerate axes must match the requested target bounds.
    checked_bounds = fitted.bounds
    if checked_bounds is not None:
        checked_min, checked_max = checked_bounds
        mismatch_axes = []
        for axis in range(3):
            if abs(src_size[axis]) >= eps:
                if (
                    abs(checked_min[axis] - target_min[axis]) > 1e-6
                    or abs(checked_max[axis] - target_max[axis]) > 1e-6
                ):
                    mismatch_axes.append(axis)
        if mismatch_axes:
            raise RuntimeError(
                f"Failed to fit scene bounds on axes {mismatch_axes}: "
                f"got min={checked_min}, max={checked_max}, "
                f"target min={target_min}, target max={target_max}"
            )

    return fitted


def _resolve_prefixed_scene_name(base: str, available_names) -> Optional[str]:
    """
    Match `base` or `base.<suffix>` where '.' is literal. Prefer exact `base`;
    otherwise pick the shortest full name (then lexicographic) among dot-suffix matches.
    """
    names = set(available_names)
    if base in names:
        return base
    prefix = base + "."
    candidates = [n for n in names if n.startswith(prefix)]
    if not candidates:
        return None
    return min(candidates, key=lambda n: (len(n), n))


def build_and_save_bg(mesh_info_dict, decompose_output_dir=None, save_outputs=True):
    """
    Build room background meshes (floor/ceiling/4 walls) from decomposed meshes.
    Save to decompose_output_dir/bg.
    """
    bg_output_dir = None
    if save_outputs:
        if decompose_output_dir is None:
            raise ValueError("decompose_output_dir is required when save_outputs=True")
        bg_output_dir = os.path.join(decompose_output_dir, "bg")
        os.makedirs(bg_output_dir, exist_ok=True)

    available = mesh_info_dict.keys()
    floor_name = _resolve_prefixed_scene_name("Floor", available)
    ceiling_name = "Ceiling"
    wall_logical = ["Wall1", "Wall2", "Wall3", "Wall4"]
    wall_key_by_logical = {
        w: _resolve_prefixed_scene_name(w, available) for w in wall_logical
    }
    wall_names = {k for k in wall_key_by_logical.values() if k is not None}
    excluded_names = set(wall_names)
    excluded_names.add(ceiling_name)
    if floor_name is not None:
        excluded_names.add(floor_name)

    def get_aabb(name):
        entry = mesh_info_dict.get(name)
        if entry is None:
            return None
        aabb = entry.get("aabb")
        if aabb is None:
            return None
        return np.asarray(aabb, dtype=float)

    if floor_name is None:
        print("[bg] Skip background generation: Floor / Floor.* not found.")
        return None

    floor_aabb = get_aabb(floor_name)
    if floor_aabb is None:
        print("[bg] Skip background generation: Floor AABB missing.")
        return None

    room_floor_y = float(floor_aabb[1, 1])

    ceiling_aabb = get_aabb(ceiling_name)
    if ceiling_aabb is not None:
        room_ceiling_y = float(ceiling_aabb[0, 1])
    else:
        max_z = None
        for name, info in mesh_info_dict.items():
            if name in excluded_names:
                continue
            aabb = info.get("aabb")
            if aabb is None:
                continue
            cur_max_z = float(aabb[1][2])
            max_z = cur_max_z if max_z is None else max(max_z, cur_max_z)
        if max_z is None:
            max_z = float(floor_aabb[1, 2])
        room_ceiling_y = 1.25 * max_z

    x_min, x_max, z_min, z_max = None, None, None, None
    for name, info in mesh_info_dict.items():
        if name in excluded_names:
            continue
        aabb = info.get("aabb")
        if aabb is None:
            continue
        aabb_np = np.asarray(aabb, dtype=float)
        cur_x_min, cur_z_min = aabb_np[0, 0], aabb_np[0, 2]
        cur_x_max, cur_z_max = aabb_np[1, 0], aabb_np[1, 2]
        x_min = cur_x_min if x_min is None else min(x_min, cur_x_min)
        x_max = cur_x_max if x_max is None else max(x_max, cur_x_max)
        z_min = cur_z_min if z_min is None else min(z_min, cur_z_min)
        z_max = cur_z_max if z_max is None else max(z_max, cur_z_max)

    if x_min is None:
        x_min, x_max = float(floor_aabb[0, 0]), float(floor_aabb[1, 0])
    if z_min is None:
        z_min, z_max = float(floor_aabb[0, 2]), float(floor_aabb[1, 2])

    floor_scene = mesh_info_dict[floor_name]["mesh"]
    if floor_scene is None or floor_scene.bounds is None:
        print("[bg] Skip background generation: Floor mesh empty.")
        return None

    floor_thickness = max(float(floor_scene.bounds[1, 1] - floor_scene.bounds[0, 1]), 1e-3)
    floor_target_min = np.array([x_min, room_floor_y - floor_thickness, z_min], dtype=float)
    floor_target_max = np.array([x_max, room_floor_y, z_max], dtype=float)
    floor_bg = _fit_scene_to_bounds(floor_scene, floor_target_min, floor_target_max)
    if floor_bg is None:
        print("[bg] Skip background generation: failed to fit floor.")
        return None
    if save_outputs:
        floor_bg.export(os.path.join(bg_output_dir, "Floor.glb"))

    if ceiling_name in mesh_info_dict and ceiling_aabb is not None:
        ceiling_scene = mesh_info_dict[ceiling_name]["mesh"].copy()
    else:
        ceiling_scene = floor_scene.copy()
    ceiling_bg = None
    if ceiling_scene is not None and ceiling_scene.bounds is not None:
        ceiling_thickness = max(float(ceiling_scene.bounds[1, 1] - ceiling_scene.bounds[0, 1]), 1e-3)
        ceiling_target_min = np.array([x_min, room_ceiling_y, z_min], dtype=float)
        ceiling_target_max = np.array([x_max, room_ceiling_y + ceiling_thickness, z_max], dtype=float)
        ceiling_bg = _fit_scene_to_bounds(ceiling_scene, ceiling_target_min, ceiling_target_max)
        if ceiling_bg is not None and save_outputs:
            ceiling_bg.export(os.path.join(bg_output_dir, "Ceiling.glb"))

    wall_sources = {}
    wall_thin_sizes = []
    for logical_wall in wall_logical:
        actual_wall = wall_key_by_logical.get(logical_wall)
        if actual_wall is None:
            continue
        wall_entry = mesh_info_dict.get(actual_wall)
        if wall_entry is None:
            continue
        wall_scene = wall_entry.get("mesh")
        if wall_scene is None or wall_scene.bounds is None:
            continue
        wall_sources[logical_wall] = wall_scene
        wall_size = wall_scene.bounds[1] - wall_scene.bounds[0]
        wall_thin_sizes.append(float(max(min(wall_size[0], wall_size[2]), 1e-3)))

    if len(wall_sources) == 0:
        print("[bg] Skip wall generation: no wall template found.")
        return None

    wall_thickness = float(np.median(wall_thin_sizes))

    wall_targets = [
        # Place walls outside the room bounds so wall thickness doesn't penetrate inward.
        (np.array([x_min, room_floor_y, z_min - wall_thickness], dtype=float),
         np.array([x_max, room_ceiling_y, z_min], dtype=float)),
        (np.array([x_min, room_floor_y, z_max], dtype=float),
         np.array([x_max, room_ceiling_y, z_max + wall_thickness], dtype=float)),
        (np.array([x_min - wall_thickness, room_floor_y, z_min], dtype=float),
         np.array([x_min, room_ceiling_y, z_max], dtype=float)),
        (np.array([x_max, room_floor_y, z_min], dtype=float),
         np.array([x_max + wall_thickness, room_ceiling_y, z_max], dtype=float)),
    ]

    wall_fallback_name = sorted(wall_sources.keys())[0]

    def _extract_texture_material(scene):
        for geom in scene.geometry.values():
            visual = getattr(geom, "visual", None)
            if visual is None:
                continue
            material = getattr(visual, "material", None)
            if material is not None:
                return copy.deepcopy(material)
        return None

    def _make_wall_box(target_min, target_max, material):
        target_min = np.asarray(target_min, dtype=float)
        target_max = np.asarray(target_max, dtype=float)
        extents = np.maximum(target_max - target_min, 1e-8)
        center = 0.5 * (target_min + target_max)

        wall_box = trimesh.creation.box(extents=extents)
        t = np.eye(4, dtype=float)
        t[:3, 3] = center
        wall_box.apply_transform(t)

        # Simple UVs for wall texture mapping on the long wall plane.
        vertices = wall_box.vertices
        thin_axis = 0 if extents[0] <= extents[2] else 2
        if thin_axis == 0:
            u = (vertices[:, 2] - target_min[2]) / max(extents[2], 1e-8)
        else:
            u = (vertices[:, 0] - target_min[0]) / max(extents[0], 1e-8)
        v = (vertices[:, 1] - target_min[1]) / max(extents[1], 1e-8)
        uv = np.column_stack((u, v))

        if material is not None:
            wall_box.visual = trimesh.visual.texture.TextureVisuals(
                uv=uv,
                material=material
            )
        return wall_box

    walls_bg = {}
    for wall_idx, (target_min, target_max) in enumerate(wall_targets, start=1):
        src_name = f"Wall{wall_idx}" if f"Wall{wall_idx}" in wall_sources else wall_fallback_name
        src_scene = wall_sources[src_name]
        src_material = _extract_texture_material(src_scene)
        wall_bg = _make_wall_box(target_min, target_max, src_material)
        if wall_bg is not None:
            walls_bg[f"Wall{wall_idx}"] = wall_bg
            if save_outputs:
                wall_bg.export(os.path.join(bg_output_dir, f"Wall{wall_idx}.glb"))

    bg_scene = trimesh.Scene()
    bg_scene.add_geometry(floor_bg)
    if ceiling_bg is not None:
        bg_scene.add_geometry(ceiling_bg)
    for wall_name in ["Wall1", "Wall2", "Wall3", "Wall4"]:
        if wall_name in walls_bg:
            bg_scene.add_geometry(walls_bg[wall_name])
    if save_outputs:
        bg_scene.export(os.path.join(bg_output_dir, "bg.glb"))

    if save_outputs:
        print(f"[bg] Saved room background meshes to {bg_output_dir}")
    return {
        "floor": floor_bg,
        "ceiling": ceiling_bg,
        "walls": walls_bg,
        "bg_scene": bg_scene,
    }


def get_mesh_info_dict(file_path, meta_path):
    """
    Build and return mesh_info_dict from input scene/meta without exporting meshes.
    A temporary directory is used for intermediate GLB conversion and removed automatically.
    """
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    with tempfile.TemporaryDirectory(prefix="decompose_scene_glb_") as tmp_dir:
        _to_glb(file_path, tmp_dir)
        glb_model_path = os.path.join(tmp_dir, "model.glb")

        with open(meta_path, 'r') as f:
            meta = json.load(f)
        object_names = list(meta['objects'].keys())

        scene_mesh = trimesh.load(glb_model_path, force='scene')
        mesh_info_dict = {}

        print("object_names: ", object_names)
        for node_name in object_names:
            if meta["objects"][node_name]["type"] != "MESH":
                continue

            print(f"Processing {node_name}...")
            scene_mesh_iteration = copy.deepcopy(scene_mesh)
            node_mesh = robust_subscene(scene_mesh_iteration, node_name)
            prune_descendant_nodes(node_mesh, node_name, object_names)

            transform = None
            try:
                transform, _ = node_mesh.graph.get(
                    frame_to=node_name,
                    frame_from=node_mesh.graph.base_frame
                )
            except Exception:
                transform = None

            aabb = None
            try:
                bounds = node_mesh.bounds
                if bounds is not None:
                    aabb = bounds.tolist()
            except Exception:
                aabb = None

            mesh_info_dict[node_name] = {
                "mesh": node_mesh,
                "transform": transform,
                "aabb": aabb,
                "canonical_mesh": None,
            }

            if len(node_mesh.geometry) == 0:
                print(f"node_name: {node_name} has no geometry")
                continue

            canonical_mesh = node_mesh.copy()
            node_geom = canonical_mesh.graph.transforms.node_data.get(node_name, {}).get("geometry")
            update_kwargs = {"matrix": np.eye(4)}
            if node_geom:
                update_kwargs["geometry"] = node_geom
            canonical_mesh.graph.update(
                frame_to=node_name,
                frame_from=canonical_mesh.graph.base_frame,
                **update_kwargs
            )
            mesh_info_dict[node_name]["canonical_mesh"] = canonical_mesh

        bg_result = build_and_save_bg(mesh_info_dict, save_outputs=False)
        if bg_result is not None:
            for old_bg_name in ["Floor", "Ceiling", "Wall1", "Wall2", "Wall3", "Wall4"]:
                if old_bg_name in mesh_info_dict:
                    del mesh_info_dict[old_bg_name]

            bg_scene = bg_result["bg_scene"]
            bg_bounds = bg_scene.bounds
            mesh_info_dict["bg"] = {
                "mesh": bg_scene,
                "transform": np.eye(4),
                "canonical_mesh": bg_scene.copy(),
                "aabb": bg_bounds.tolist() if bg_bounds is not None else None,
            }

        return mesh_info_dict

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


def get_canonical_mesh_info_dict(file_path, meta_path):

    mesh_info_dict = get_mesh_info_dict(file_path, meta_path)
    canonical_mesh_info_dict = {}
    for mesh_name in mesh_info_dict.keys():
        original_mesh = mesh_info_dict[mesh_name]["canonical_mesh"] # not canonical here
        added_transform = mesh_info_dict[mesh_name]["transform"]

        scale, _, _, _, _ = trimesh.transformations.decompose_matrix(added_transform)
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale[0]
        scale_matrix[1, 1] = scale[1]
        scale_matrix[2, 2] = scale[2]

        original_mesh = original_mesh.apply_transform(scale_matrix)

        # rotate
        orientation_matrix = np.array([
            [-1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

        original_mesh.apply_transform(orientation_matrix)

        if isinstance(original_mesh, trimesh.Scene):
            center, scale = normalize_mesh(trimesh.util.concatenate(original_mesh.dump()))
        else:
            center, scale = normalize_mesh(original_mesh)

        normalization_center_matrix = np.eye(4)
        normalization_center_matrix[:3, 3] = -center
        normalization_scale_matrix = np.eye(4)
        normalization_scale_matrix[np.diag_indices(3)] = scale
        original_mesh.apply_transform(normalization_center_matrix)
        original_mesh.apply_transform(normalization_scale_matrix)

        canonical_mesh = original_mesh.copy()

        rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

        added_transform = rot_x_90 @ added_transform
        canonical_transform = normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix @ scale_matrix
        total_transform = added_transform @ np.linalg.inv(canonical_transform)

        # decompose the total_transform
        scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(total_transform)

        print(f"mesh_name: {mesh_name}; scale: {scale}; angles: {angles}; trans: {trans}")

        scale = float(scale.reshape(3)[0])
        angles = [float(angle) for angle in angles]
        trans = [float(trans) for trans in trans]

        final_mesh = canonical_mesh.copy().apply_transform(total_transform)

        canonical_mesh_info_dict[mesh_name] = {
            "mesh": final_mesh,
            "transform": total_transform,
            "canonical_mesh": canonical_mesh,
            "scale": scale,
            "angles": angles,
            "trans": trans,
        }

    return canonical_mesh_info_dict

def export_canonical_meshes(file_path, meta_path, debug=False):


    scene_name = file_path.split("/")[-1].split(".")[0]
    print(f"Exporting scene: {scene_name}")

    scene_save_dir = os.path.join(DATASET_ROOT, "scenes")
    transforms_save_dir = os.path.join(DATASET_ROOT, "transforms")
    os.makedirs(scene_save_dir, exist_ok=True)
    os.makedirs(transforms_save_dir, exist_ok=True)
    if debug:
        export_dir = os.path.join(DATASET_ROOT, "exports", scene_name, "from_canonical")
        os.makedirs(export_dir, exist_ok=True)



    canonical_mesh_info_dict = get_canonical_mesh_info_dict(file_path, meta_path)

    objects_save_dir = os.path.join(scene_save_dir, scene_name)
    transforms_save_path = os.path.join(transforms_save_dir, f"{scene_name}.pkl")

    os.makedirs(objects_save_dir, exist_ok=True)

    all_room_element_names = list(canonical_mesh_info_dict.keys())
    all_room_object_names = [element_name for element_name in all_room_element_names if element_name != "bg"]
    all_room_object_names = sorted(all_room_object_names)
    all_room_element_names = ["bg"] + all_room_object_names

    object_transform_dict = {}

    for mesh_idx, mesh_name in enumerate(all_room_element_names):
        mesh_dict = canonical_mesh_info_dict[mesh_name]
        mesh_canonical = mesh_dict["canonical_mesh"]
        scale = mesh_dict["scale"]
        angles = mesh_dict["angles"]
        trans = mesh_dict["trans"]

        if mesh_idx == 0:
            latent_name = f"layout_{scene_name}"
        else:
            latent_name = f"object_{mesh_idx:04d}"

        save_glb(mesh_canonical, os.path.join(objects_save_dir, f"{latent_name}.glb"))

        object_transform_dict[latent_name] = {
            "scale": scale,
            "angles": angles,
            "trans": trans,
            "latent": latent_name,
            "mesh_name": mesh_name,
        }

        print(f"Exported transform for {latent_name} ({mesh_name}): scale={scale}, angles={angles}, trans={trans}")

        if debug:
            total_transform = trimesh.transformations.compose_matrix(
                scale=[scale, scale, scale],
                shear=None,
                angles=[angles[0], angles[1], angles[2]],
                translate=[trans[0], trans[1], trans[2]],
                perspective=None
            )

            save_glb(mesh_canonical.copy().apply_transform(total_transform), os.path.join(export_dir, f"{latent_name}.glb"))

    with open(transforms_save_path, "wb") as f:
        pickle.dump(object_transform_dict, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to the 3D blend model file to be rendered.')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='Path to the metadata file.')
    parser.add_argument('--dataset-root', type=Path, default=Path(DATASET_ROOT))
    parser.add_argument('--blender-path', type=Path, default=Path(BLENDER_PATH))
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False)

    opt = parser.parse_args()
    opt = edict(vars(opt))
    DATASET_ROOT = str(opt.dataset_root.expanduser().resolve())
    BLENDER_PATH = str(opt.blender_path.expanduser().resolve())

    export_canonical_meshes(opt.file_path, opt.meta_path, debug=opt.debug)
