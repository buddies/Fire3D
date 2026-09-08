import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import pickle
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
SAGE_KIT = REPO_ROOT / "data_processing/datasets/sage10k"
if str(SAGE_KIT) not in sys.path:
    sys.path.insert(0, str(SAGE_KIT))

from tex_utils_local import export_layout_to_mesh_dict_list_v2  # noqa: E402
from utils import dict_to_floor_plan  # noqa: E402

DEFAULT_SAGE_ROOT = Path(
    os.environ.get(
        "FIRE3D_SAGE10K_ROOT",
        REPO_ROOT / "data/training_scenes/SAGE-10k",
    )
)


def preprocess_mesh(mesh: trimesh.Trimesh):
    """
    Preprocess the input mesh.
    """
    vertices = mesh.vertices
    vertices_min = vertices.min(axis=0)
    vertices_max = vertices.max(axis=0)
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()
    vertices = (vertices - center) * scale

    mesh_transform = (center, scale)
    assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
    return mesh_transform

def _encode(
    input_mesh_file_path,
):

    mesh = trimesh.load_mesh(input_mesh_file_path, process=False)
    mesh_transform = preprocess_mesh(mesh)

    mesh_center, mesh_scale = mesh_transform

    normalized_transform = {
        'center': mesh_center,
        'scale': mesh_scale,
    }

    object_id = os.path.splitext(os.path.basename(input_mesh_file_path))[0]

    return {
        object_id: normalized_transform
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='Directory to the data',
                        default=str(DEFAULT_SAGE_ROOT / "scenes"))
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save the latents',
                        default=str(DEFAULT_SAGE_ROOT / "transforms"))
    parser.add_argument('--scene_id', type=str,
                        help='Scene ID', required=True)
    opt = parser.parse_args()
    os.makedirs(opt.save_dir, exist_ok=True)

    # process objects
    scene_id = opt.scene_id

    scene_dir = os.path.join(opt.data_dir, scene_id)
    objects_dir = os.path.join(scene_dir, 'objects')
    mesh_file_paths = [os.path.join(objects_dir, f) for f in os.listdir(objects_dir) if f.endswith('.ply')]

    layout_id = scene_id[-len("layout_xxxxxxxx"):]
    layout_json_path = os.path.join(scene_dir, f'{layout_id}.json')
    with open(layout_json_path, 'r') as f:
        layout_data = json.load(f)
    layout = dict_to_floor_plan(layout_data)
    mesh_dict_list = export_layout_to_mesh_dict_list_v2(layout, scene_dir)

    bg_mesh_list = []
    for mesh_id in mesh_dict_list:
        if mesh_id.startswith('floor_') or mesh_id.startswith('wall_') or mesh_id.startswith('window_') or mesh_id.startswith('door_'):
            bg_mesh_list.append(mesh_dict_list[mesh_id]["mesh"])
    bg_mesh = trimesh.util.concatenate(bg_mesh_list)

    # create a temp file to save the bg mesh
    temp_bg_mesh_dir = os.path.join(opt.data_dir, '../temp_bg_mesh')
    os.makedirs(temp_bg_mesh_dir, exist_ok=True)
    temp_bg_mesh_path = os.path.join(temp_bg_mesh_dir, f'{layout_id}_bg.ply')
    bg_mesh.export(temp_bg_mesh_path)
    bg_id = f'{layout_id}_bg'

    mesh_file_paths = [temp_bg_mesh_path] + mesh_file_paths
    normalized_transform_dict = {}

    for mesh_file_path in tqdm(mesh_file_paths, desc='Encoding objects'):
        mesh_normalized_transform = _encode(mesh_file_path)
        normalized_transform_dict.update(mesh_normalized_transform)

    # remove the temp bg mesh
    if os.path.exists(temp_bg_mesh_path):
        os.remove(temp_bg_mesh_path)

    object_transform_dict = {}
    object_transform_dict[bg_id] = {
        "scale": 1 / float(normalized_transform_dict[bg_id]['scale']),
        "angles": [0, 0, 0],
        "trans": normalized_transform_dict[bg_id]['center'].reshape(3).tolist(),
        "latent": bg_id
    }
    print(f"Background transform: scale={object_transform_dict[bg_id]['scale']}, angles={object_transform_dict[bg_id]['angles']}, trans={object_transform_dict[bg_id]['trans']}")

    for room in layout.rooms:
        for obj in room.objects:
            rx_rad = np.radians(obj.rotation.x)
            ry_rad = np.radians(obj.rotation.y)
            rz_rad = np.radians(obj.rotation.z)

            # Create rotation matrices for each axis
            # Rotation order: X -> Y -> Z (Euler XYZ)
            rotation_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
            rotation_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
            rotation_z = trimesh.transformations.rotation_matrix(rz_rad, [0, 0, 1])

            # Combine rotations (order matters: Z * Y * X for XYZ Euler)
            combined_rotation = rotation_z @ rotation_y @ rotation_x

            # Create translation matrix
            translation = trimesh.transformations.translation_matrix([
                obj.position.x,
                obj.position.y,
                obj.position.z
            ])


            source_id = obj.source_id
            assert source_id in normalized_transform_dict, f"Object {source_id} not found in normalized transform dict"

            normalized_transform = normalized_transform_dict[source_id]
            normalized_center = normalized_transform['center'].reshape(3)
            normalized_scale = float(normalized_transform['scale'])

            normalized_translation = trimesh.transformations.translation_matrix([
                -normalized_center[0],
                -normalized_center[1],
                -normalized_center[2]
            ])

            normalized_scaling = trimesh.transformations.scale_matrix(normalized_scale)

            # Combine rotation and translation (translation after rotation)
            final_transform = translation @ combined_rotation @ np.linalg.inv(normalized_translation) @ np.linalg.inv(normalized_scaling)

            # decompose the final transform into translation, rotation, and scaling
            scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(final_transform)

            scale = float(scale.reshape(3)[0])
            angles = [float(angle) for angle in angles]
            trans = [float(trans) for trans in trans]

            object_transform_dict[obj.id] = {
                "scale": scale,
                "angles": angles,
                "trans": trans,
                "latent": source_id
            }

            print(f"Object {obj.id} transform: scale={scale}, angles={angles}, trans={trans}")

    # save the transforms
    save_transform_path = os.path.join(opt.save_dir, f'{scene_id}.pkl')
    with open(save_transform_path, 'wb') as f:
        pickle.dump(object_transform_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved transforms to {save_transform_path}")
