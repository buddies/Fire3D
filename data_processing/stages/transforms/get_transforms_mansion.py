import os
from concurrent.futures import ProcessPoolExecutor
import pickle
import trimesh
from tqdm import tqdm
import time
from utils import (
    get_all_scene_paths,
    get_room_geoms,
)

if __name__ == '__main__':
    data_dir = "/path/to/scene_datasets/MansionWorld"
    transforms_save_dir = os.path.join(data_dir, 'transforms')
    os.makedirs(transforms_save_dir, exist_ok=True)
    num_processes = 64
    scene_paths = get_all_scene_paths()

    def process_scene(scene_path):
        scene_name = "_".join(os.path.splitext(scene_path)[0].split("/")[-2:])

        scene_geoms, room_polygons_dict = get_room_geoms(scene_path)

        for room_id, room_geoms in scene_geoms.items():

            object_transform_dict = {}
            # print(f"Processing room {room_id} of {scene_name}")

            all_room_elements = list(room_geoms.keys())
            # print(f"All room elements: {all_room_elements}")
            all_room_objects = [e for e in all_room_elements if e != "bg"]
            # print(f"All room objects: {all_room_objects}")
            all_room_objects = sorted(all_room_objects)
            # print(f"Sorted all room objects: {all_room_objects}")
            all_room_elements = ["bg"] + all_room_objects
            # print(f"Sorted all room elements: {all_room_elements}")

            room_save_name = f"{scene_name}_room_{room_id}"
            room_transform_save_path = os.path.join(transforms_save_dir, f'{room_save_name}.pkl')

            if os.path.exists(room_transform_save_path):
                print(f"Transforms already exist for {room_save_name}, skipping")
                continue


            for mesh_idx, mesh_name in enumerate(all_room_elements):

                mesh_geom = room_geoms[mesh_name]

                final_transform = mesh_geom["transform"]
                if mesh_idx == 0:
                    latent_name = f"layout_{room_save_name}"
                    asset_id = None
                else:
                    latent_name = f"object_{mesh_idx:04d}"
                    asset_id = mesh_geom["asset_id"]

                # decompose the final transform into translation, rotation, and scaling
                scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(final_transform)

                scale = float(scale.reshape(3)[0])
                angles = [float(angle) for angle in angles]
                trans = [float(trans) for trans in trans]

                object_transform_dict[latent_name] = {
                    "scale": scale,
                    "angles": angles,
                    "trans": trans,
                    "latent": latent_name,
                    "asset_id": asset_id
                }

                # print(f"Object {mesh_name} ({asset_id}|{latent_name}) transform: scale={scale}, angles={angles}, trans={trans}")



            # save the transforms
            with open(room_transform_save_path, 'wb') as f:
                pickle.dump(object_transform_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            # print(f"Saved transforms to {room_transform_save_path}")

        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{time_stamp}] Processed scene {scene_name}")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        list(tqdm(executor.map(process_scene, scene_paths), total=len(scene_paths), desc="Processing scenes"))

    # for scene_path in tqdm(scene_paths, desc="Processing scenes"):
    #     process_scene(scene_path)
