import os
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATASET_ROOT = os.environ.get(
    "FIRE3D_SCENESMITH_ROOT",
    os.path.join(REPO_ROOT, "data", "training_scenes", "Scenesmith"),
)
def get_all_scene_paths():
    data_sub_dir_name_list = [
        "House",
        "Room",
        "NoAgentMemory",
        "NoAssetValidation",
        "NoCritic",
        "NoObserveScene",
        "NoSpecializedTools",
        "NotGenerated"
    ]

    scene_paths = []
    for data_sub_dir_name in data_sub_dir_name_list:
        data_sub_dir_path = os.path.join(DATASET_ROOT, data_sub_dir_name)

        sub_dir_scene_names = sorted(os.listdir(data_sub_dir_path))
        print(f"Sub dir {data_sub_dir_name} has {len(sub_dir_scene_names)} scenes")

        scene_paths.extend([os.path.join(data_sub_dir_name, scene_name) for scene_name in sub_dir_scene_names])
    return scene_paths

if __name__ == "__main__":
    scene_paths = get_all_scene_paths()
    print(f"Found {len(scene_paths)} scenes")
    scene_paths_save_path = os.path.join(DATASET_ROOT, "scene_paths.json")
    with open(scene_paths_save_path, "w") as f:
        json.dump(scene_paths, f, indent=4)
