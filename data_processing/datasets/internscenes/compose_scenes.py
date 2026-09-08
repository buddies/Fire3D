import argparse
import os
import json
import trimesh
import numpy as np
from trimesh.transformations import rotation_matrix

import threading
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = os.environ.get(
    "FIRE3D_INTERNSCENES_ROOT",
    str(REPO_ROOT / "data/training_scenes/InternScenes"),
)

ASSET_LIBRARY_FOLDER = os.path.join(BASE_DIR, "asset_library")
SCENE_SAVE_DIR = os.path.join(BASE_DIR, "composed_scenes")
SCENE_INFO_DIR = os.path.join(BASE_DIR, "Layout_info")


def configure_paths(
    root: str | Path,
    *,
    asset_library: str | Path | None = None,
    scene_dir: str | Path | None = None,
    layout_dir: str | Path | None = None,
) -> None:
    """Configure source/output roots before constructing ``SceneComposer``."""

    global BASE_DIR, ASSET_LIBRARY_FOLDER, SCENE_SAVE_DIR, SCENE_INFO_DIR
    BASE_DIR = str(Path(root).expanduser().resolve())
    ASSET_LIBRARY_FOLDER = str(
        Path(asset_library).expanduser().resolve()
        if asset_library is not None
        else Path(BASE_DIR) / "asset_library"
    )
    SCENE_SAVE_DIR = str(
        Path(scene_dir).expanduser().resolve()
        if scene_dir is not None
        else Path(BASE_DIR) / "composed_scenes"
    )
    SCENE_INFO_DIR = str(
        Path(layout_dir).expanduser().resolve()
        if layout_dir is not None
        else Path(BASE_DIR) / "Layout_info"
    )


def smooth_geometry(data, iterations=10, lamb=0.5):
    """
    Smoothes a trimesh Mesh or Scene without breaking UV/topology.

    Args:
        data: trimesh.Trimesh or trimesh.Scene object.
        iterations: Number of smoothing passes.
        lamb: Smoothing displacement factor (0.0 to 1.0).

    Returns:
        The modified Mesh or Scene.
    }
    """
    # Check if the input is a Scene (GLB files usually load as scenes)
    if isinstance(data, trimesh.Scene):
        # Scenes store geometry in a dictionary: {name: Mesh object}
        for name, geometry in data.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                _apply_smoothing(geometry, iterations, lamb)
    elif isinstance(data, trimesh.Trimesh):
        _apply_smoothing(data, iterations, lamb)
    else:
        raise ValueError("Input must be a trimesh.Trimesh or trimesh.Scene object.")

    return data

def _apply_smoothing(mesh, iterations, lamb):
    """Internal helper to apply Laplacian smoothing and fix normals."""
    # 1. Smooth the actual vertex positions (Topology stays identical)
    # trimesh.smoothing.filter_laplacian(mesh, iterations=iterations, lamb=lamb)

    # 2. Clear out old normals so they are recalculated for the new shape
    # This prevents the "ghosting" look of old lighting on a new shape
    mesh.vertex_normals = None
    mesh.face_normals = None

    # 3. Force a recompute of normals
    # fix_normals() ensures consistent winding and shading
    mesh.fix_normals()

def fix_flipped_faces_by_visibility(mesh, whole_mesh):
    # 1. Ensure we are using the Embree engine if available
    # trimesh.ray.ray_pyembree is the wrapper for Intel's Embree
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(whole_mesh, scale_to_box=True)

    # 2. Setup Ray Origins and Directions
    # We use face centers and move slightly along the current normal
    # to avoid "self-intersection" at distance 0.
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    # Small epsilon offset
    epsilon = 1e-4
    ray_origins = face_centers + (face_normals * epsilon)
    ray_directions = face_normals

    # 3. Perform the Intersect Test
    # 'intersects_any' is highly optimized in Embree for boolean visibility
    is_blocked = intersector.intersects_any(ray_origins, ray_directions)
    is_inside = intersector.contains_points(ray_origins)
    is_blocked = is_blocked | is_inside

    # 4. Flip the 'Blocked' Faces
    # If the ray is blocked, it means the normal is pointing 'into' the object
    mesh.faces[is_blocked] = np.fliplr(mesh.faces[is_blocked])

    # 5. Final Cleanup
    # mesh.remove_duplicate_faces() # Critical for Cycles
    # mesh.fix_normals()

    return mesh

def fix_by_occupancy(mesh):
    eps = 1e-3
    # Move a tiny bit along the normal
    test_points = mesh.triangles_center + (mesh.face_normals * eps)

    # Check if these points are 'inside' the mesh
    # This requires the mesh to be somewhat watertight
    is_inside = mesh.contains(test_points)

    # If the point in front of the normal is INSIDE, the normal is pointing WRONG
    mesh.faces[is_inside] = np.fliplr(mesh.faces[is_inside])
    # mesh.fix_normals()
    return mesh

def merge_vertices_averaged(verts, faces, uvs, decimal_precision=3):
    # 1. Find unique positions
    verts_rounded = np.round(verts, decimals=decimal_precision)
    unique_verts, inverse_indices = np.unique(verts_rounded, axis=0, return_inverse=True)

    # 2. Compute Averaged UVs using vectorized accumulation
    # Initialize array for sums and counts
    num_unique = len(unique_verts)
    uv_sums = np.zeros((num_unique, 2), dtype=uvs.dtype)
    counts = np.zeros(num_unique, dtype=int)

    # Use np.add.at to handle duplicate indices (Python's += would fail here)
    np.add.at(uv_sums, inverse_indices, uvs)
    np.add.at(counts, inverse_indices, 1)

    new_uvs = uv_sums / counts[:, np.newaxis]

    # 3. Re-map faces
    new_faces = inverse_indices[faces]

    # 4. Filter degenerate faces
    mask = (new_faces[:, 0] != new_faces[:, 1]) & \
           (new_faces[:, 1] != new_faces[:, 2]) & \
           (new_faces[:, 0] != new_faces[:, 2])

    return unique_verts, new_faces[mask], new_uvs

class AssetMeshLoader():
    def __init__(self):

        # other assets:
        self.asset_dir = ASSET_LIBRARY_FOLDER
        self.obja_uid_2_rotation = json.load(open(os.path.join(ASSET_LIBRARY_FOLDER, "uid_2_angle.json")))
        self.pm_uid_2_origin_cate = json.load(open(os.path.join(ASSET_LIBRARY_FOLDER, "uid_2_origin_cate.json")))

    def get_mesh_path(self, uid):
        mesh_path = None

        if uid.startswith("objaverse/"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("objaverse_old/"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("partnet_mobility"):
            mesh_path = os.path.join(self.asset_dir, uid,  "whole.glb")
        elif uid.startswith("3D-FUTURE-model"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("hssd-models"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("gen_assets"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("gr100"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        else:
            raise ValueError(f"Invalid uid: {uid}")

        return mesh_path

    def load_init_mesh(self, uid, use_texture=False):
        mesh_path = self.get_mesh_path(uid)

        if use_texture:
            mesh = trimesh.load(mesh_path)
        else:
            mesh = trimesh.load(mesh_path, force="mesh")

        # --- AUTO-FIX LOGIC ---
        if isinstance(mesh, trimesh.Scene):
            whole_mesh = trimesh.util.concatenate(mesh.dump())
            # Iterate through every mesh in the scene
            for name, geometry in mesh.geometry.items():
                if isinstance(geometry, trimesh.Trimesh):
                    geometry.fix_normals(multibody=True)
        elif isinstance(mesh, trimesh.Trimesh):
            geometry.fix_normals(multibody=True)

        return mesh

    def load_init_rotation(self, uid):
        transform = None

        if uid.startswith("objaverse/"):
            rot_radius = self.obja_uid_2_rotation[uid.split("objaverse/")[-1]] / 180.0 * np.pi
            transform = rotation_matrix(rot_radius, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("objaverse_old/"):
            transform = rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("partnet_mobility"):

            transform =  rotation_matrix(np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

            # if partnet_mobility is ["pen", "remote", "phone"] category, need to rotate 180 degree around Z axis,
            # then rotate 90 degree around Y axis
            pm_cate = self.pm_uid_2_origin_cate[uid]
            if pm_cate in ["Pen", "Remote", "Phone"]:
                rotation_matrix_1 = rotation_matrix(np.pi, [0, 0, 1])
                rotation_matrix_2 = rotation_matrix(np.pi / 2, [0, 1, 0])
                transform = rotation_matrix_2 @ rotation_matrix_1 @ transform

        elif uid.startswith("3D-FUTURE-model"):
            transform =  rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("hssd-models"):
            transform = rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("gen_assets"):
            transform = rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("gr100"):
            transform = rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        else:
            raise ValueError(f"Invalid uid: {uid}")

        return transform

    def load_canonical_mesh(self, uid, use_texture=False):
        '''
        canonical : means the object is facing the X-axis direction in trimesh, and the Z-axis is up
        '''
        mesh = self.load_init_mesh(uid, use_texture)
        mesh = smooth_geometry(mesh)
        transform = self.load_init_rotation(uid)

        mesh_origen_centroid =  mesh.bounding_box.centroid
        mesh.apply_translation(- mesh_origen_centroid)
        mesh.apply_transform(transform)

        return mesh


class SceneComposer():
    def __init__(self, asset_mesh_loader = None):
        self.asset_mesh_loader = asset_mesh_loader if asset_mesh_loader is not None else AssetMeshLoader()

        # file paths
        self.scene_files_dir = SCENE_SAVE_DIR
        self.scene_info_dir = SCENE_INFO_DIR

    def get_scale_transform_from_rules(self, mesh_size, instance_info, bbox_data_key = "bbox"):
        '''
        introduce some special rules
        calculate scale based on mesh_size & instance_info's bbox, return a 4x4 transform matrix
        '''
        if instance_info["category"] not in ["carpet", "clothes"]:
            target_size = np.array(instance_info[bbox_data_key][3:6])
            scale = target_size / mesh_size
            scale_matrix = np.diag([scale[0], scale[1], scale[2], 1])
            return scale_matrix

        elif instance_info["category"] == "carpet":
            # add the judgment of carpet wrong orientation causing excessive stretching
            target_size = np.array(instance_info[bbox_data_key][3:6])
            scale_factors = target_size / mesh_size

            if target_size[2]/target_size[0] > 150 or target_size[2]/target_size[1] > 150:
                # 地毯错误朝向导致过度拉伸
                if target_size[2]/target_size[0] > target_size[2]/target_size[1]:
                    rotation_matrix = trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 1, 0]) # 绕 Y 轴转90度
                    target_size = np.array([target_size[2], target_size[0], target_size[1]])
                    scale_factors = target_size / mesh_size
                    scale_matrix = np.diag([scale_factors[0], scale_factors[1], scale_factors[2]/100.0, 1])
                else:
                    rotation_matrix = trimesh.transformations.rotation_matrix(0.5 * np.pi, [1, 0, 0]) # 绕 X 轴转90度
                    target_size = np.array([target_size[0], target_size[2], target_size[1]])
                    scale_factors = target_size / mesh_size
                    scale_matrix = np.diag([scale_factors[0], scale_factors[1], scale_factors[2]/100.0, 1])

                return scale_matrix @ rotation_matrix

            else:
                # carpet normal orientation
                scale_matrix = np.diag([scale_factors[0], scale_factors[1], scale_factors[2]/100.0, 1])

                return scale_matrix

        elif instance_info["category"] == "clothes":
            target_size = np.array(instance_info[bbox_data_key][3:6])
            scale = target_size / mesh_size
            min_scale = min(scale)
            scale_matrix = np.diag([min_scale, min_scale, min_scale, 1])

        return scale_matrix

    def _instance_mesh_and_transform(self, instance, bbox_data_key, use_texture):
        """
        Same logic as process_single_instance: load canonical mesh and compute 4x4 world transform.
        Returns (mesh_or_scene, transform) or None. Used by get_flat_mesh_dict for render_scenes.
        """
        if instance.get("model_uid") == '':
            print(f"model_uid is empty. No retrieval result. (cate: {instance.get('category')})")
            return None
        uid = instance["model_uid"]
        mesh = self.asset_mesh_loader.load_canonical_mesh(uid, use_texture=use_texture)
        box_data = instance[bbox_data_key]
        transform_final = np.eye(4)
        mesh_size = mesh.bounding_box.extents
        scale_transform_matrix = self.get_scale_transform_from_rules(mesh_size, instance, bbox_data_key=bbox_data_key)
        transform_final = scale_transform_matrix @ transform_final
        euler_angles = np.array(box_data[6:9])
        rot_mat = trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
        transform_final = rot_mat @ transform_final
        center = np.array(box_data[0:3])
        transform_final[:3, 3] = center
        # rot_minus_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        # transform_final = rot_minus_x @ transform_final
        return mesh, transform_final

    def _geom_to_single_mesh(self, geom):
        """Convert geometry (Trimesh or Scene) to a single trimesh.Trimesh."""
        if isinstance(geom, trimesh.Trimesh):
            return geom
        if isinstance(geom, trimesh.Scene):
            meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                return None
            if len(meshes) == 1:
                return meshes[0]
            return trimesh.util.concatenate(meshes)
        return None

    def get_flat_mesh_dict(self, scene_name, use_texture=True, add_floor=True, add_wall=True, add_ceiling=True, bbox_data_key="bbox"):
        """
        Build mesh_info_dict with keys "0","1",... and "floor","wall","ceiling".
        Each value is {"mesh": trimesh.Trimesh} in world coordinates.
        Uses the same loading and transform logic as compose_scene_from_instance_infos (no Scene graph).
        For use by render_scenes to avoid scene graph traversal.
        """
        input_path = os.path.join(self.scene_info_dir, scene_name, "layout.json")
        instance_infos = json.load(open(input_path))
        mesh_info_dict = {}
        for index, instance in enumerate(instance_infos):
            result = self._instance_mesh_and_transform(instance, bbox_data_key, use_texture)
            if result is None:
                continue
            mesh_or_scene, transform = result


            mesh_scene = trimesh.Scene()
            parent_node_name = str(index)

            mesh_scene.graph.update(frame_to=parent_node_name, matrix=transform)

            if isinstance(mesh_or_scene, trimesh.Scene):
                for geom_name, mesh_part in mesh_or_scene.geometry.items():
                    nodes_for_this_geometry = mesh_or_scene.graph.geometry_nodes.get(geom_name, [])

                    for i, node_name_in_subscene in enumerate(nodes_for_this_geometry):
                        internal_transform, _ = mesh_or_scene.graph.get(node_name_in_subscene)
                        mesh_scene.add_geometry(
                            mesh_part,
                            geom_name=f"{parent_node_name}_{geom_name}_{i}",
                            transform=internal_transform,
                            parent_node_name=parent_node_name
                        )
            else: #
                mesh_scene.add_geometry(
                    mesh_or_scene,
                    geom_name=parent_node_name + "_geom",
                    parent_node_name=parent_node_name
                )

            key = f"{index:04d}"
            mesh_info_dict[key] = {"mesh": mesh_scene}
        structure_dir = os.path.join(self.scene_info_dir, scene_name, "StructureMesh")
        if add_floor:
            try:
                floor = trimesh.load(os.path.join(structure_dir, "floor.glb"))

                rot_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
                floor.apply_transform(rot_x)

                if floor is not None:
                    mesh_info_dict["floor"] = {"mesh": floor}
            except Exception as e:
                print(f"Error adding floor: {e}")
        if add_wall:
            try:
                wall = trimesh.load(os.path.join(structure_dir, "wall.glb"))

                rot_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
                wall.apply_transform(rot_x)

                if wall is not None:
                    mesh_info_dict["wall"] = {"mesh": wall}
            except Exception as e:
                print(f"Error adding wall: {e}")
        if add_ceiling:
            try:
                ceiling = trimesh.load(os.path.join(structure_dir, "ceiling.glb"))

                rot_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
                ceiling.apply_transform(rot_x)

                if ceiling is not None:
                    mesh_info_dict["ceiling"] = {"mesh": ceiling}
            except Exception as e:
                print(f"Error adding ceiling: {e}")
        if not use_texture:
            from trimesh.visual.texture import TextureVisuals
            empty_visual = TextureVisuals()
            for info in mesh_info_dict.values():
                if "mesh" in info and hasattr(info["mesh"], "visual"):
                    info["mesh"].visual = empty_visual
        return mesh_info_dict

    def compose_scene_from_instance_infos(self, instance_infos, output_glb_path, use_texture, bbox_data_key = "bbox"):
        # init scene
        scene = trimesh.scene.Scene()
        lock = threading.Lock()

        def process_single_instance(instance):
            """
            function to process single instance, for multi-threading
            """
            if instance["model_uid"] == '':
                print(f"model_uid is empty. No retrieval reslut. (cate: {instance['category']})")
                return None

            uid = instance["model_uid"]
            mesh = self.asset_mesh_loader.load_canonical_mesh(uid, use_texture = use_texture)

            # get geometry name
            geometry_name = instance["category"] + "@" + instance["model_uid"]

            # transform
            transform_final = np.eye(4)
            box_data = instance[bbox_data_key]

            # scale
            mesh_size = mesh.bounding_box.extents
            scale_transform_matrix = self.get_scale_transform_from_rules(mesh_size, instance,   bbox_data_key = bbox_data_key)
            transform_final = scale_transform_matrix @ transform_final

            # rotation
            euler_angles = np.array(box_data[6:9])
            rotation_matrix = trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
            transform_final = rotation_matrix @ transform_final

            # translation
            center = np.array(box_data[0:3])
            transform_final[:3, 3] = center

            # rotate -90 degree around X axis, to make the scene Y-up, so that the scene has the correct upward orientation
            rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
            transform_final = rotation_matrix @ transform_final

            total_transform = transform_final

            return mesh, geometry_name, total_transform

        def process_instance(index):
            instance = instance_infos[index]
            result = process_single_instance(instance)

            if result is not None:
                mesh_or_scene, parent_node_name, transform = result
                parent_node_name = str(index) + '_' + parent_node_name
                with lock:
                    scene.graph.update(frame_to=parent_node_name, matrix=transform)

                    if isinstance(mesh_or_scene, trimesh.Scene):
                        for geom_name, mesh_part in mesh_or_scene.geometry.items():
                            nodes_for_this_geometry = mesh_or_scene.graph.geometry_nodes.get(geom_name, [])

                            for i, node_name_in_subscene in enumerate(nodes_for_this_geometry):
                                internal_transform, _ = mesh_or_scene.graph.get(node_name_in_subscene)
                                scene.add_geometry(
                                    mesh_part,
                                    geom_name=f"{parent_node_name}_{geom_name}_{i}",
                                    transform=internal_transform,
                                    parent_node_name=parent_node_name
                                )
                    else: #
                        scene.add_geometry(
                            mesh_or_scene,
                            geom_name=parent_node_name + "_geom",
                            parent_node_name=parent_node_name
                        )
                print(f"process_instance {index} done")

        # use thread pool to process all instances
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for index in range(len(instance_infos)):
                futures.append(executor.submit(process_instance, index))

            # wait for all tasks to complete and handle exceptions
            for future in futures: #tqdm(futures, desc="exporting scene..."):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing instance: {str(e)}")

        # remove texture from scene glb
        if not use_texture:
            from trimesh.visual.texture import TextureVisuals
            empty_visual = TextureVisuals()
            for geometry in scene.geometry.values():
                geometry.visual = empty_visual

        if output_glb_path != None:
            trimesh.exchange.export.export_mesh(scene, output_glb_path)
        return scene


    def compose_one_scene(self, scene_name, use_texture = True, add_floor = True, add_wall = True, add_ceiling = True, output_glb_path = None, _skip_export = False):
        """
        Load layout.json and StructureMesh (floor, wall, ceiling) using the same paths and trimesh.load as in this module.
        When _skip_export=True (used by render_scenes), the scene is not written to disk; the composed trimesh_scene is still returned.
        Otherwise exports to output_glb_path, or to scene_files_dir/<scene_name>/glb_scene.glb if output_glb_path is None.
        Returns trimesh_scene in all cases.
        """
        input_instance_infos_path = os.path.join(self.scene_info_dir, scene_name, "layout.json")
        if output_glb_path is None:
            output_glb_path = os.path.join(self.scene_files_dir, scene_name, "glb_scene.glb")
        do_export = not _skip_export
        if do_export:
            os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
        instance_infos = json.load(open(input_instance_infos_path))
        trimesh_scene = self.compose_scene_from_instance_infos(instance_infos, None, use_texture, bbox_data_key = "bbox")

        if add_floor:
            try:
                floor = trimesh.load(os.path.join(self.scene_info_dir, scene_name, "StructureMesh", "floor.glb"))
                trimesh_scene.add_geometry(floor, geom_name=f"floor")
            except Exception as e:
                print(f"Error adding floor: {e}")

        if add_wall:
            try:
                wall = trimesh.load(os.path.join(self.scene_info_dir, scene_name, "StructureMesh", "wall.glb"))
                trimesh_scene.add_geometry(wall, geom_name=f"wall")
            except Exception as e:
                print(f"Error adding wall: {e}")

        if add_ceiling:
            try:
                ceiling = trimesh.load(os.path.join(self.scene_info_dir, scene_name, "StructureMesh", "ceiling.glb"))
                trimesh_scene.add_geometry(ceiling, geom_name=f"ceiling")
            except Exception as e:
                print(f"Error adding ceiling: {e}")

        # remove texture from scene glb
        if not use_texture:
            from trimesh.visual.texture import TextureVisuals
            empty_visual = TextureVisuals()
            for geometry in trimesh_scene.geometry.values():
                geometry.visual = empty_visual

        if do_export:
            trimesh.exchange.export.export_mesh(trimesh_scene, output_glb_path)
            print(f"Composed glb scene has been saved to {output_glb_path}")

        return trimesh_scene



def parse_args():
    parser = argparse.ArgumentParser(description="Compose an InternScenes scene as GLB.")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path(BASE_DIR))
    parser.add_argument("--asset-library", type=Path)
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--layout-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--texture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--floor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wall", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ceiling", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_paths(
        args.dataset_root,
        asset_library=args.asset_library,
        scene_dir=args.scene_dir,
        layout_dir=args.layout_dir,
    )
    SceneComposer().compose_one_scene(
        args.scene_name,
        use_texture=args.texture,
        add_floor=args.floor,
        add_wall=args.wall,
        add_ceiling=args.ceiling,
        output_glb_path=str(args.output) if args.output else None,
    )
