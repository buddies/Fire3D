import torch
import os
import copy
import random
import numpy as np
import multiprocessing as mp
from contextlib import contextmanager
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import json
from scipy import ndimage
from PIL import Image
from utils.project import project_depth_to_points
from utils.read_objects import read_objects_to_tokens_wo_latents_with_instance_ids
from utils.transforms import (
    _euler_to_rotation_matrix_sxyz,
    point_augment,
    point_normalize,
    transform_6d_from_transform_np,
)
from utils.constants import MAX_SCENE_OBJECTS, POS_MIN, POS_MAX, SCALE_MIN, SCALE_MAX
from utils.read_frames import (
    read_rgbs,
    read_depths,
    read_masks_v2,
    read_cameras,
    read_prune_masks,
)
from utils.project import (
    project_depth_to_world_patch_geometry_with_instance_mask,
    downsample_all,
    downsample_valid_masks,
)
from utils.load_real_world_dataset import load_data as load_real_world_data
from utils.load_s3d_dataset import load_data as load_s3d_data
from training.datasets.split_manifest_utils import DEFAULT_SPLIT_DIR, load_scene_split


DEFAULT_SPLIT_PREFIX = "obj_gen_feat_with_objects"

class VoxelizationDataset(Dataset):
    @property
    def enable_noise(self):
        return bool(self._enable_noise.value)

    @enable_noise.setter
    def enable_noise(self, value):
        self._enable_noise.value = bool(value)

    @property
    def enable_instance_augment(self):
        return bool(self._enable_instance_augment.value)

    @enable_instance_augment.setter
    def enable_instance_augment(self, value):
        self._enable_instance_augment.value = bool(value)

    @property
    def augmentation_intensity(self):
        return float(self._augmentation_intensity.value)

    @augmentation_intensity.setter
    def augmentation_intensity(self, value):
        self._augmentation_intensity.value = float(np.clip(value, 0.0, 1.0))

    def __init__(
        self,
        data_dir_info, # dict
        scene_scale=POS_MAX,
        split='train',
        train_split_ratio=0.9,
        train_val_split=True,
        augment=False,
        num_max_scenes=None, # only use the first x scenes for training for overfitting
        depth_noise_std=0.0,
        camera_translation_noise_std=0.0,
        camera_rotation_noise_std=0.0,
        frame_subsample=False,
        sample_full_frame_prob=0.1,
        single_frame_sampling=None,
        all_augmentation_probability=1.0,
        split_manifest_dir=DEFAULT_SPLIT_DIR,
        split_manifest_prefix=DEFAULT_SPLIT_PREFIX,
        use_split_manifests=True,
        strict_split_manifests=True,
        instance_augment=None,
        scene_yaw_augmentation=None,
    ):
        self.data_dir_info = data_dir_info
        self.split = split

        self.scene_scale = scene_scale
        self.augment = augment
        self.depth_noise_std = float(depth_noise_std)
        self.camera_translation_noise_std = float(camera_translation_noise_std)
        self.camera_rotation_noise_std = float(camera_rotation_noise_std)
        self.frame_subsample = bool(frame_subsample)
        self.sample_full_frame_prob = sample_full_frame_prob
        single_frame_sampling = dict(single_frame_sampling or {})
        self.single_frame_sampling_enabled = bool(
            single_frame_sampling.get("enabled", False)
        )
        self.single_frame_sampling_strategy = str(
            single_frame_sampling.get(
                "strategy", "foreground_instance_weighted"
            )
        )
        self.single_frame_sampling_instance_count_power = float(
            single_frame_sampling.get("instance_count_power", 2.0)
        )
        if self.single_frame_sampling_strategy != "foreground_instance_weighted":
            raise ValueError(
                "single_frame_sampling.strategy must be "
                "'foreground_instance_weighted'"
            )
        if self.single_frame_sampling_instance_count_power <= 0.0:
            raise ValueError(
                "single_frame_sampling.instance_count_power must be positive"
            )
        self.all_augmentation_probability = float(all_augmentation_probability)
        self.split_manifest_dir = split_manifest_dir
        self.split_manifest_prefix = split_manifest_prefix
        self.use_split_manifests = bool(use_split_manifests)
        self.strict_split_manifests = bool(strict_split_manifests)

        scene_yaw_augmentation = dict(scene_yaw_augmentation or {})
        self.scene_yaw_enabled = bool(
            scene_yaw_augmentation.get("enabled", False)
        )
        self.scene_yaw_sampling = str(
            scene_yaw_augmentation.get("sampling", "continuous")
        ).lower()
        self.scene_yaw_degrees = tuple(
            float(value)
            for value in scene_yaw_augmentation.get(
                "yaw_degrees", [-180.0, 180.0]
            )
        )
        if self.scene_yaw_sampling == "continuous":
            if len(self.scene_yaw_degrees) != 2:
                raise ValueError(
                    "continuous scene yaw requires yaw_degrees=[min, max]"
                )
            if self.scene_yaw_degrees[0] > self.scene_yaw_degrees[1]:
                raise ValueError(
                    "continuous scene yaw requires min <= max"
                )
        elif self.scene_yaw_sampling == "discrete":
            if not self.scene_yaw_degrees:
                raise ValueError(
                    "discrete scene yaw requires at least one yaw_degrees choice"
                )
        else:
            raise ValueError(
                "scene_yaw_augmentation.sampling must be continuous or discrete"
            )
        self.continuous_scene_yaw_enabled = (
            self.scene_yaw_enabled and self.scene_yaw_sampling == "continuous"
        )
        self.continuous_scene_yaw_degrees = self.scene_yaw_degrees

        instance_augment = dict(instance_augment or {})
        self.instance_augment_enabled = bool(instance_augment.get("enabled", False))
        self.instance_augment_probability = float(instance_augment.get("probability", 0.0))
        self.instance_augment_object_ratio = float(instance_augment.get("object_ratio", 0.4))
        operation_probs = dict(instance_augment.get("operation_probs", {}))
        self.instance_augment_operation_names = []
        self.instance_augment_operation_probs = []
        for operation_name in ("partial_delete", "full_delete", "rotate", "transform", "copy_paste"):
            probability = float(operation_probs.get(operation_name, 0.0))
            if probability > 0:
                self.instance_augment_operation_names.append(operation_name)
                self.instance_augment_operation_probs.append(probability)
        if self.instance_augment_operation_probs:
            probs = np.asarray(self.instance_augment_operation_probs, dtype=np.float64)
            self.instance_augment_operation_probs = probs / probs.sum()
        self.instance_augment_yaw_degrees = tuple(
            float(value) for value in instance_augment.get("yaw_degrees", [-180.0, 180.0])
        )
        self.instance_augment_delete_ratio = tuple(
            float(value) for value in instance_augment.get("partial_delete_ratio", [0.2, 0.5])
        )
        self.instance_augment_min_source_points = int(instance_augment.get("min_source_points", 128))
        self.instance_augment_min_remaining_points = int(instance_augment.get("min_remaining_points", 64))
        transform_config = dict(instance_augment.get("transform", {}))
        self.instance_augment_translation_candidates = int(transform_config.get("translation_candidates", 16))
        self.instance_augment_yaw_candidates = int(transform_config.get("yaw_candidates_per_translation", 8))
        self.instance_augment_scale_range = tuple(float(v) for v in transform_config.get("scale_range", [0.85, 1.15]))
        self.instance_augment_min_translation_size_ratio = float(transform_config.get("min_translation_xy_size_ratio", 0.5))
        self.instance_augment_room_grid_size = float(transform_config.get("room_grid_size", 0.05))
        self.instance_augment_room_gap_close = float(transform_config.get("room_gap_close", 0.20))
        self.instance_augment_room_boundary_margin = float(transform_config.get("room_boundary_margin", 0.05))
        self.instance_augment_collision_voxel_size = float(transform_config.get("collision_voxel_size", 0.05))
        self.instance_augment_max_collision_ratio = float(transform_config.get("max_collision_ratio", 0.01))
        self.instance_augment_max_copies_per_scene = int(transform_config.get("max_copies_per_scene", 8))
        copy_paste_config = dict(instance_augment.get("copy_paste", {}))
        self.instance_augment_copy_count_min = int(copy_paste_config.get("copies_per_object_min", 1))
        self.instance_augment_copy_count_max = int(copy_paste_config.get("copies_per_object_max", 1))
        copy_count_values = np.arange(
            self.instance_augment_copy_count_min,
            self.instance_augment_copy_count_max + 1,
            dtype=np.int64,
        )
        copy_count_probs = copy_paste_config.get("copies_per_object_probs")
        if copy_count_probs is None:
            self.instance_augment_copy_count_probs = np.full(
                copy_count_values.shape[0], 1.0 / copy_count_values.shape[0], dtype=np.float64
            )
        else:
            probs = np.asarray(copy_count_probs, dtype=np.float64)
            if probs.shape != copy_count_values.shape or np.any(probs < 0) or probs.sum() <= 0:
                raise ValueError(
                    "copy_paste.copies_per_object_probs must be nonnegative and match the configured integer range"
                )
            self.instance_augment_copy_count_probs = probs / probs.sum()
        external_config = dict(instance_augment.get("external_object_paste", {}))
        self.external_object_paste_enabled = bool(external_config.get("enabled", False))
        self.external_object_paste_count = int(external_config.get("objects_per_scene", 8))
        self.external_object_paste_attempts_per_object = int(external_config.get("attempts_per_object", 8))
        default_object_root = os.environ.get(
            "FIRE3D_OBJECT_ROOT", os.path.join("data", "training_objects")
        )
        default_manifest_root = os.environ.get(
            "FIRE3D_MANIFEST_ROOT", DEFAULT_SPLIT_DIR
        )
        self.external_object_render_root = str(
            external_config.get("render_root", os.path.join(default_object_root, "render"))
        )
        self.external_object_train_manifest = str(
            external_config.get(
                "train_manifest",
                os.path.join(
                    default_manifest_root,
                    "obj_gen_feat_with_objects_object_train.json",
                ),
            )
        )
        self.external_object_scale_jitter = tuple(
            float(v) for v in external_config.get("scene_scale_jitter", [0.8, 1.2])
        )
        self.external_object_scale_strategy = str(
            external_config.get("scale_strategy", "volume_match")
        )
        self.external_object_volume_quantiles = tuple(
            float(v) for v in external_config.get("volume_extent_quantiles", [0.02, 0.98])
        )
        self.external_object_volume_scale_clamp_ratio = tuple(
            float(v) for v in external_config.get("volume_scale_clamp_ratio", [0.5, 2.0])
        )
        self._external_object_names = None
        if len(self.instance_augment_yaw_degrees) != 2:
            raise ValueError("instance_augment.yaw_degrees must contain [min, max]")
        if len(self.instance_augment_delete_ratio) != 2:
            raise ValueError("instance_augment.partial_delete_ratio must contain [min, max]")
        if not 0.0 <= self.instance_augment_probability <= 1.0:
            raise ValueError("instance_augment.probability must be in [0, 1]")
        if not 0.0 <= self.all_augmentation_probability <= 1.0:
            raise ValueError("all_augmentation_probability must be in [0, 1]")
        if not 0.0 < self.instance_augment_object_ratio <= 1.0:
            raise ValueError("instance_augment.object_ratio must be in (0, 1]")
        if not 0.0 <= self.instance_augment_delete_ratio[0] <= self.instance_augment_delete_ratio[1] < 1.0:
            raise ValueError("instance_augment.partial_delete_ratio must satisfy 0 <= min <= max < 1")
        if len(self.instance_augment_scale_range) != 2 or not 0.0 < self.instance_augment_scale_range[0] <= self.instance_augment_scale_range[1]:
            raise ValueError("instance_augment.transform.scale_range must satisfy 0 < min <= max")
        if self.instance_augment_translation_candidates < 1 or self.instance_augment_yaw_candidates < 1:
            raise ValueError("instance_augment transform candidate counts must be positive")
        if self.instance_augment_copy_count_min < 1 or self.instance_augment_copy_count_max < self.instance_augment_copy_count_min:
            raise ValueError("copy_paste copies_per_object range must satisfy 1 <= min <= max")
        if self.external_object_paste_count < 0 or self.external_object_paste_attempts_per_object < 1:
            raise ValueError("external object count must be nonnegative and attempts_per_object positive")
        if len(self.external_object_scale_jitter) != 2 or not 0.0 < self.external_object_scale_jitter[0] <= self.external_object_scale_jitter[1]:
            raise ValueError("external_object_paste.scene_scale_jitter must satisfy 0 < min <= max")
        if self.external_object_scale_strategy not in ("volume_match", "scene_scale"):
            raise ValueError("external_object_paste.scale_strategy must be volume_match or scene_scale")
        if len(self.external_object_volume_quantiles) != 2 or not 0.0 <= self.external_object_volume_quantiles[0] < self.external_object_volume_quantiles[1] <= 1.0:
            raise ValueError("external object volume quantiles must satisfy 0 <= low < high <= 1")
        if len(self.external_object_volume_scale_clamp_ratio) != 2 or not 0.0 < self.external_object_volume_scale_clamp_ratio[0] <= self.external_object_volume_scale_clamp_ratio[1]:
            raise ValueError("external object volume scale clamp ratio must be positive and ordered")

        # Share the noise toggle across DataLoader worker processes so the
        # trainer can flip it on after warmup even with persistent workers.
        self._enable_noise = mp.Value("b", False)
        self._enable_instance_augment = mp.Value("b", False)
        self._augmentation_intensity = mp.Value("d", 0.0)

        """
        data_dir_info

        dataset_name: {
            data_dir: str,
            use_weight: int, (>=1)
        }

        """

        if not isinstance(self.data_dir_info, dict) or len(self.data_dir_info) == 0:
            raise ValueError("data_dir_info must be a non-empty dict")

        dataset_names = sorted(
            dataset_name
            for dataset_name, dataset_cfg in self.data_dir_info.items()
            if dataset_cfg.get("enabled", True)
        )
        if not dataset_names:
            raise ValueError("data_dir_info must contain at least one enabled dataset")
        for dataset_name in dataset_names:
            dataset_cfg = self.data_dir_info[dataset_name]
            if "data_dir" not in dataset_cfg or "use_weight" not in dataset_cfg:
                raise ValueError(
                    f"Dataset '{dataset_name}' must contain keys: 'data_dir' and 'use_weight'"
                )
            use_weight = int(dataset_cfg["use_weight"])
            if use_weight < 1:
                raise ValueError(
                    f"Dataset '{dataset_name}' has invalid use_weight={use_weight}. Must be integer >= 1"
                )

        split_data_by_dataset = {}
        scene_split = None
        if self.use_split_manifests:
            scene_split = load_scene_split(
                self.split,
                split_dir=self.split_manifest_dir,
                prefix=self.split_manifest_prefix,
            )
            manifest_split = scene_split.get("split")
            if manifest_split is not None and manifest_split != self.split:
                raise ValueError(
                    f"Scene manifest split is '{manifest_split}', requested '{self.split}'"
                )

        for dataset_name in dataset_names:
            dataset_cfg = self.data_dir_info[dataset_name]
            use_real = dataset_name.startswith("REAL-")
            num_videos_per_scene = dataset_cfg["num_videos_per_scene"]
            manifest_dataset_name = dataset_name.removeprefix("REAL-")
            require_prune_masks = bool(dataset_cfg.get("require_prune_masks", False))

            dataset_scene_split = scene_split
            dataset_split_manifest_dir = dataset_cfg.get(
                "split_manifest_dir", self.split_manifest_dir
            )
            dataset_split_manifest_prefix = dataset_cfg.get(
                "split_manifest_prefix", self.split_manifest_prefix
            )
            if self.use_split_manifests and (
                dataset_split_manifest_dir != self.split_manifest_dir
                or dataset_split_manifest_prefix != self.split_manifest_prefix
            ):
                dataset_scene_split = load_scene_split(
                    self.split,
                    split_dir=dataset_split_manifest_dir,
                    prefix=dataset_split_manifest_prefix,
                )
                manifest_split = dataset_scene_split.get("split")
                if manifest_split is not None and manifest_split != self.split:
                    raise ValueError(
                        f"Scene manifest split is '{manifest_split}', requested '{self.split}'"
                    )

            data_dir = self.data_dir_info[dataset_name]["data_dir"]
            realistic_dir = os.path.join(data_dir, 'realistic')
            transforms_dir = os.path.join(data_dir, 'transforms')

            unused_scene_ids_path = os.path.join("preprocess/unused_data", f"{dataset_name.replace('REAL-', '')}.json")
            if os.path.exists(unused_scene_ids_path):
                with open(unused_scene_ids_path, 'r') as f:
                    unused_scene_ids = set(json.load(f))
                print(f"[{dataset_name}] Loaded {len(unused_scene_ids)} unused scene IDs from {unused_scene_ids_path}")
            else:
                unused_scene_ids = set()
                print(f"[{dataset_name}] No unused scene IDs file found at {unused_scene_ids_path}; using all scenes")

            sample_weights_path = os.path.join("preprocess/sample_weights", f"{dataset_name.replace('REAL-', '')}.json")
            if os.path.exists(sample_weights_path):
                with open(sample_weights_path, 'r') as f:
                    sample_weights = json.load(f)
                print(f"[{dataset_name}] Loaded sample weights from {sample_weights_path}")
            else:
                sample_weights = None

            if use_real and not os.path.isdir(realistic_dir):
                raise ValueError(f"Dataset '{dataset_name}' realistic_dir not found: {realistic_dir}")
            if not os.path.isdir(transforms_dir):
                raise ValueError(f"Dataset '{dataset_name}' transforms_dir not found: {transforms_dir}")


            renders_dir = os.path.join(data_dir, 'renders')

            if not os.path.isdir(renders_dir):
                raise ValueError(f"Dataset '{dataset_name}' renders_dir not found: {renders_dir}")

            manifest_video_ids = {}
            if self.use_split_manifests:
                manifest_scenes = [
                    item for item in dataset_scene_split.get("scenes", [])
                    if item.get("dataset_name") == manifest_dataset_name
                ]
                if not manifest_scenes and self.strict_split_manifests:
                    raise ValueError(
                        f"Scene manifest has no entries for dataset '{manifest_dataset_name}'"
                    )
                scene_ids = [item["scene_id"] for item in manifest_scenes]
                manifest_video_ids = {
                    item["scene_id"]: [
                        int(video_id) for video_id in item.get("video_ids", [])
                        if 0 <= int(video_id) < num_videos_per_scene
                    ]
                    for item in manifest_scenes
                }
                print(
                    f"[{dataset_name}] Loaded {len(scene_ids)} ordered scene IDs from "
                    f"{dataset_split_manifest_dir}/{dataset_split_manifest_prefix}_scene_{self.split}.json"
                )
            else:
                scene_ids = sorted(os.listdir(renders_dir))

            # debug: only use the first x scenes
            if num_max_scenes is not None:
                scene_ids = scene_ids[:num_max_scenes]
                print(
                    f"[{dataset_name}] Using only the first {num_max_scenes} scenes for overfitting"
                )

            data_list_group_by_scene = []
            unused_indices = []
            missing_manifest_sequences = []
            for scene_idx, scene_id in enumerate(scene_ids):

                scene_data_list = []
                video_ids = (
                    manifest_video_ids.get(scene_id, [])
                    if self.use_split_manifests
                    else range(num_videos_per_scene)
                )
                for video_i in video_ids:

                    sample_weight_scene = int(sample_weights[scene_id]) if sample_weights is not None and scene_id in sample_weights else 1
                    sample_weight_scene = max(sample_weight_scene, 1)


                    data_dict = {
                        'dataset_name': dataset_name,
                        'scene_id': scene_id,
                        'video_id': video_i,
                        "data_name": f"{dataset_name}_{scene_id}_{video_i}",
                        'camera_path': os.path.join(renders_dir, scene_id, f'{video_i}.json'),
                        'frames_dir': os.path.join(renders_dir, scene_id, f'{video_i}_frames'),
                        'masks_dir': os.path.join(renders_dir, scene_id, f'{video_i}_masks'),
                        'depth_dir': os.path.join(renders_dir, scene_id, f'{video_i}_depth'),
                        'prune_masks_dir': os.path.join(renders_dir, scene_id, f'{video_i}_prune_masks'),
                        'realistic_frames_dir': os.path.join(realistic_dir, scene_id, f'{video_i}_frames'),
                        'transforms_path': os.path.join(transforms_dir, f'{scene_id}.pkl'),
                        'use_real': use_real,
                        'sample_weight_scene': sample_weight_scene
                    }
                    required_paths = [
                        data_dict['camera_path'],
                        data_dict['realistic_frames_dir'] if use_real else data_dict['frames_dir'],
                        data_dict['masks_dir'],
                        data_dict['depth_dir'],
                        data_dict['transforms_path'],
                    ]
                    if require_prune_masks:
                        required_paths.append(data_dict['prune_masks_dir'])
                    if all(os.path.exists(path) for path in required_paths):
                        scene_data_list.append(data_dict)
                    elif self.use_split_manifests and scene_id not in unused_scene_ids:
                        missing_manifest_sequences.append(data_dict['data_name'])


                if len(scene_data_list) > 0:
                    data_list_group_by_scene.append(scene_data_list)
                    if scene_id in unused_scene_ids:
                        unused_indices.append(len(data_list_group_by_scene) - 1)

            if missing_manifest_sequences and self.strict_split_manifests:
                preview = ", ".join(missing_manifest_sequences[:5])
                raise ValueError(
                    f"Scene manifest references {len(missing_manifest_sequences)} missing "
                    f"{dataset_name} sequences. First missing: {preview}"
                )

            print(f"[{dataset_name}] Total number of scenes: {len(data_list_group_by_scene)}")

            total_num_scenes = len(data_list_group_by_scene)
            if total_num_scenes == 0:
                split_data_by_dataset[dataset_name] = []
                continue

            if self.use_split_manifests:
                scene_indices = np.arange(total_num_scenes, dtype=np.int32)
            else:
                train_scene_indices = np.linspace(
                    0,
                    total_num_scenes - 1,
                    max(int(total_num_scenes * train_split_ratio), 1)
                ).astype(np.int32)
                val_scene_indices = np.setdiff1d(np.arange(total_num_scenes), train_scene_indices)
                if not train_val_split:
                    train_scene_indices = np.arange(total_num_scenes)

                if self.split == 'train':
                    scene_indices = train_scene_indices
                elif self.split == 'trainval':
                    scene_indices = train_scene_indices[:int(0.1 * len(train_scene_indices) + 1)]
                elif self.split == 'val':
                    scene_indices = val_scene_indices
                else:
                    raise ValueError(f"Invalid split: {self.split}")

            print(f"[{dataset_name}] Real: {use_real}")

            print(f"[{dataset_name}] Number of scenes in {self.split} set: {len(scene_indices)}")
            valid_scene_mask = np.ones(total_num_scenes, dtype=bool)
            if unused_indices:
                valid_scene_mask[np.asarray(unused_indices, dtype=np.int32)] = False
            valid_scene_indices = np.asarray(scene_indices, dtype=np.int32)[valid_scene_mask[scene_indices]]
            split_data = list(chain.from_iterable(data_list_group_by_scene[scene_idx] for scene_idx in valid_scene_indices))
            print(f"[{dataset_name}] Number of sequences in {self.split} set: {len(split_data)}")
            split_data_by_dataset[dataset_name] = split_data

        # Build a mixed dataset with integer weights by duplicating each dataset split list.
        self.data_list = []
        for dataset_name in dataset_names:
            dataset_split_list = split_data_by_dataset[dataset_name]
            use_weight = int(self.data_dir_info[dataset_name]["use_weight"])
            if len(dataset_split_list) == 0:
                print(f"[{dataset_name}] Empty split list in {self.split}; skipping")
                continue
            print(f"[{dataset_name}] Before sampling weighted: {len(dataset_split_list)} sequences in {self.split} split.")
            dataset_split_list = self.apply_data_sampling_weight(dataset_split_list)
            print(f"[{dataset_name}] After sampling weighted: Adding {len(dataset_split_list) * use_weight} sequences with the length of {len(dataset_split_list)} and weight {use_weight} to the mixed {self.split} set")
            self.data_list.extend(dataset_split_list * use_weight)

        np.random.seed(0)
        np.random.shuffle(self.data_list)
        print(f"Total number of mixed sequences in {self.split} set: {len(self.data_list)}")

    def apply_data_sampling_weight(self, data_list):
        weighted_data_list = []

        for data_dict in data_list:
            sample_weight_scene = data_dict["sample_weight_scene"]
            weighted_data_list.extend([data_dict] * sample_weight_scene)

        return weighted_data_list


    def __len__(self):
        return len(self.data_list)


    def augment_points(self, points, augment=True):

        # 2. Augment (Apply rotation matrix to points here)
        points, augment_transform = point_augment(
            points,
            augment=augment,
            continuous_yaw=(
                self.scene_yaw_enabled
                and self.scene_yaw_sampling == "continuous"
            ),
            continuous_yaw_degrees=self.continuous_scene_yaw_degrees,
            discrete_yaw_degrees=(
                self.scene_yaw_degrees
                if self.scene_yaw_sampling == "discrete"
                else (0.0, 90.0, 180.0, 270.0)
            ),
        )
        points, norm_transform = point_normalize(points)

        augment_info = {
            'augment_transform': augment_transform,
            'norm_transform': norm_transform
        }

        return points, augment_info

    def _build_all_point_room_region(self, points):
        """Build a conservative non-rectangular XY room mask from all points.

        All instances contribute observations, so furniture occlusion does not
        punch artificial holes into a background-only mask. Morphological gap
        closing connects nearby observations and fills enclosed visibility
        holes without imposing a rectangular room prior.
        """
        grid_size = self.instance_augment_room_grid_size
        xy = np.asarray(points[:, :2], dtype=np.float64)
        xy_min = np.floor(xy.min(axis=0) / grid_size) * grid_size - grid_size
        xy_max = np.ceil(xy.max(axis=0) / grid_size) * grid_size + grid_size
        shape = np.maximum(1, np.ceil((xy_max - xy_min) / grid_size).astype(np.int64) + 1)
        # Avoid pathological memory use for unusually large-coordinate scenes.
        max_cells = 4_000_000
        if int(shape[0] * shape[1]) > max_cells:
            grid_size *= np.sqrt(float(shape[0] * shape[1]) / max_cells)
            xy_min = np.floor(xy.min(axis=0) / grid_size) * grid_size - grid_size
            xy_max = np.ceil(xy.max(axis=0) / grid_size) * grid_size + grid_size
            shape = np.maximum(1, np.ceil((xy_max - xy_min) / grid_size).astype(np.int64) + 1)

        indices = np.floor((xy - xy_min) / grid_size).astype(np.int64)
        indices = np.clip(indices, 0, shape - 1)
        occupied = np.zeros(tuple(shape.tolist()), dtype=bool)
        occupied[indices[:, 0], indices[:, 1]] = True

        close_cells = max(1, int(np.ceil(self.instance_augment_room_gap_close / grid_size)))
        yy, xx = np.ogrid[-close_cells:close_cells + 1, -close_cells:close_cells + 1]
        close_disk = (xx * xx + yy * yy) <= close_cells * close_cells
        dilated = ndimage.binary_dilation(occupied, structure=close_disk)
        region = ndimage.binary_fill_holes(dilated)
        # Undo the visibility-gap expansion, then apply an explicit inward
        # safety margin. Fall back to the filled dilation if sparse sampling
        # would otherwise erase the region completely.
        region = ndimage.binary_erosion(region, structure=close_disk)
        margin_cells = max(0, int(np.ceil(self.instance_augment_room_boundary_margin / grid_size)))
        if margin_cells:
            margin_disk = ndimage.generate_binary_structure(2, 1)
            region = ndimage.binary_erosion(region, structure=margin_disk, iterations=margin_cells)
        if not region.any():
            region = ndimage.binary_fill_holes(dilated)

        valid_cells = np.argwhere(region)
        z_min = float(points[:, 2].min()) - grid_size
        z_max = float(points[:, 2].max()) + grid_size
        return {
            "grid_size": float(grid_size),
            "origin": xy_min,
            "mask": region,
            "valid_cells": valid_cells,
            "z_min": z_min,
            "z_max": z_max,
        }

    @staticmethod
    def _apply_world_delta(points, world_delta):
        points_h = np.concatenate([
            np.asarray(points, dtype=np.float64),
            np.ones((points.shape[0], 1), dtype=np.float64),
        ], axis=1)
        return (points_h @ world_delta.T)[:, :3]

    @staticmethod
    def _points_inside_room_region(points, room_region):
        grid_size = room_region["grid_size"]
        indices = np.floor((points[:, :2] - room_region["origin"]) / grid_size).astype(np.int64)
        shape = np.asarray(room_region["mask"].shape, dtype=np.int64)
        in_bounds = np.all((indices >= 0) & (indices < shape), axis=1)
        inside = np.zeros(points.shape[0], dtype=bool)
        valid = np.flatnonzero(in_bounds)
        if valid.size:
            inside[valid] = room_region["mask"][indices[valid, 0], indices[valid, 1]]
        inside &= points[:, 2] >= room_region["z_min"]
        inside &= points[:, 2] <= room_region["z_max"]
        return bool(inside.all())

    @staticmethod
    def _voxel_keys(points, voxel_size):
        if points.size == 0:
            return set()
        voxels = np.floor(np.asarray(points, dtype=np.float64) / voxel_size).astype(np.int64)
        return {tuple(row) for row in np.unique(voxels, axis=0)}

    def _robust_point_extents(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 3:
            return None
        low, high = self.external_object_volume_quantiles
        extents = np.quantile(points, high, axis=0) - np.quantile(points, low, axis=0)
        if not np.all(np.isfinite(extents)) or np.any(extents <= 1e-4):
            return None
        return extents

    def _get_external_object_names(self):
        """Lazily load train-only names; disabled augmentation performs no I/O."""
        if self._external_object_names is None:
            with open(self.external_object_train_manifest, "r") as f:
                manifest = json.load(f)
            if manifest.get("split") != "train":
                raise ValueError(
                    f"External object manifest must be train split: {self.external_object_train_manifest}"
                )
            names = manifest.get("object_names", [])
            if not names:
                raise ValueError(f"No object_names in {self.external_object_train_manifest}")
            self._external_object_names = tuple(str(name) for name in names)
        return self._external_object_names

    def _external_object_dir(self, encoded_name):
        sources = (
            "ObjaverseXL_sketchfab", "ObjaverseXL_github",
            "3D-FUTURE", "HSSD", "ABO",
        )
        for source in sources:
            prefix = source + "_"
            if encoded_name.startswith(prefix):
                return os.path.join(
                    self.external_object_render_root,
                    source,
                    encoded_name[len(prefix):],
                ), source
        raise ValueError(f"Unknown object-bank source in name: {encoded_name}")

    @staticmethod
    def _camera_axes_from_render_frame(frame):
        extrinsics = frame["extrinsics"]
        eye = np.asarray(extrinsics["eye"], dtype=np.float64)
        lookat = np.asarray(extrinsics["lookat"], dtype=np.float64)
        up = np.asarray(extrinsics["up"], dtype=np.float64)
        forward = lookat - eye
        forward /= np.linalg.norm(forward) + 1e-12
        up /= np.linalg.norm(up) + 1e-12
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-12
        return eye, right, up, forward

    def _load_external_object_view(self, encoded_name, target_height, target_width):
        """Load one view and produce one canonical 3D point per valid DINO patch."""
        object_dir, source = self._external_object_dir(encoded_name)
        with open(os.path.join(object_dir, "cameras.json"), "r") as f:
            cameras = json.load(f)
        frame_index = int(np.random.randint(0, len(cameras["frames"])))
        frame = cameras["frames"][frame_index]
        stem = os.path.splitext(frame["file"])[0]
        rgb = np.asarray(Image.open(
            os.path.join(object_dir, "frames", frame["file"])
        ).convert("RGB"), dtype=np.uint8)
        with np.load(os.path.join(object_dir, "depth", stem + ".npz")) as data:
            depth = np.asarray(data["depth"] if "depth" in data else data[data.files[0]], dtype=np.float32)
        with np.load(os.path.join(object_dir, "mask", stem + ".npz")) as data:
            mask = np.asarray(data["mask"] if "mask" in data else data[data.files[0]], dtype=bool)

        source_height, source_width = depth.shape
        rgb_resized = np.asarray(
            Image.fromarray(rgb).resize((target_width, target_height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        depth_resized = np.asarray(
            Image.fromarray(depth, mode="F").resize(
                (target_width, target_height), Image.Resampling.BILINEAR
            ), dtype=np.float32,
        )
        mask_resized = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (target_width, target_height), Image.Resampling.NEAREST
            ), dtype=np.uint8,
        ) > 0

        intrinsics = frame["intrinsics"]
        fx = float(intrinsics["fx"]) * target_width / source_width
        fy = float(intrinsics["fy"]) * target_height / source_height
        cx = float(intrinsics["cx"]) * target_width / source_width
        cy = float(intrinsics["cy"]) * target_height / source_height
        eye, right, up, forward = self._camera_axes_from_render_frame(frame)
        patch_height = target_height // 16
        patch_width = target_width // 16
        patch_mask = np.zeros((patch_height, patch_width), dtype=bool)
        patch_points = []
        for patch_y in range(patch_height):
            y0, y1 = patch_y * 16, (patch_y + 1) * 16
            for patch_x in range(patch_width):
                x0, x1 = patch_x * 16, (patch_x + 1) * 16
                local_valid = (
                    mask_resized[y0:y1, x0:x1]
                    & np.isfinite(depth_resized[y0:y1, x0:x1])
                    & (depth_resized[y0:y1, x0:x1] > 0)
                )
                if not local_valid.any():
                    continue
                local_y, local_x = np.nonzero(local_valid)
                v = local_y.astype(np.float64) + y0
                u = local_x.astype(np.float64) + x0
                z = depth_resized[y0:y1, x0:x1][local_valid].astype(np.float64)
                x = (u - cx) * z / fx
                y = -(v - cy) * z / fy
                points = (
                    eye[None] + x[:, None] * right[None]
                    + y[:, None] * up[None] + z[:, None] * forward[None]
                )
                patch_points.append(np.median(points, axis=0))
                patch_mask[patch_y, patch_x] = True
        return {
            "encoded_name": encoded_name,
            "source": source,
            "frame_index": frame_index,
            "rgb": rgb_resized,
            "patch_mask": patch_mask.reshape(-1),
            "points": np.asarray(patch_points, dtype=np.float64).reshape(-1, 3),
            "robust_extents": self._robust_point_extents(
                np.asarray(patch_points, dtype=np.float64).reshape(-1, 3)
            ),
        }

    def _augment_external_objects(
        self, points, instance_ids, objects_transforms, rgbs,
        valid_point_masks, point_source_indices, metadata,
        apply_instance_augment=None,
    ):
        """Paste train-bank objects while extending the unified RGB/DINO row set."""
        if (
            self.split != "train"
            or not self.instance_augment_enabled
            or not self.enable_instance_augment
            or not self.external_object_paste_enabled
            or self.external_object_paste_count == 0
            or apply_instance_augment is False
        ):
            return (
                points, instance_ids, objects_transforms, rgbs,
                valid_point_masks, point_source_indices, metadata,
            )

        object_keys = sorted(k for k in objects_transforms if not k.startswith("layout_"))
        capacity = max(0, (MAX_SCENE_OBJECTS - 1) - len(object_keys))
        current_external_max = int(round(
            self.external_object_paste_count * self.augmentation_intensity
        ))
        target_count = min(
            int(np.random.randint(0, current_external_max + 1)), capacity
        )
        if target_count == 0:
            return (
                points, instance_ids, objects_transforms, rgbs,
                valid_point_masks, point_source_indices, metadata,
            )
        names = self._get_external_object_names()
        room_region = self._build_all_point_room_region(points)
        canonical_corners = np.asarray([
            [x, y, z]
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
        ], dtype=np.float64)
        scene_scales = np.asarray([
            float(objects_transforms[key]["scale"])
            for key in object_keys
            if np.isfinite(objects_transforms[key]["scale"])
            and float(objects_transforms[key]["scale"]) > 0
        ], dtype=np.float64)
        if scene_scales.size == 0:
            scene_scales = np.asarray([1.0], dtype=np.float64)
        target_volume_candidates = []
        for raw_instance_id, object_key in enumerate(object_keys, start=1):
            target_points = points[instance_ids == raw_instance_id]
            if target_points.shape[0] < 4:
                continue
            transform_entry = objects_transforms[object_key]
            center = np.asarray(transform_entry["trans"], dtype=np.float64)
            rotation = _euler_to_rotation_matrix_sxyz(transform_entry["angles"])
            # World points expressed in the target GT-local axes, retaining
            # world units so the extent product is the occupied world volume.
            target_local_world = (target_points.astype(np.float64) - center) @ rotation
            target_extents = self._robust_point_extents(target_local_world)
            if target_extents is None:
                continue
            target_volume_candidates.append({
                "raw_instance_id": raw_instance_id,
                "object_key": object_key,
                "extents": target_extents,
                "volume": float(np.prod(target_extents)),
                "gt_scale": float(transform_entry["scale"]),
            })

        points_out = points.copy()
        instance_ids_out = instance_ids.copy()
        transforms_out = copy.deepcopy(objects_transforms)
        rgbs_out = rgbs
        if valid_point_masks is None:
            scene_dino_rows = rgbs.shape[0] * (rgbs.shape[1] // 16) * (rgbs.shape[2] // 16)
            valid_masks_out = np.ones(scene_dino_rows, dtype=bool)
        else:
            valid_masks_out = valid_point_masks.copy()
        if point_source_indices is None:
            source_out = np.arange(points.shape[0], dtype=np.int64)
        else:
            source_out = point_source_indices.copy()
        operations = []

        for slot in range(target_count):
            accepted = None
            rejection_counts = {
                "empty_view": 0, "outside_room": 0,
                "foreground_collision": 0, "background_collision": 0,
            }
            for _ in range(self.external_object_paste_attempts_per_object):
                encoded_name = str(names[int(np.random.randint(0, len(names)))])
                try:
                    view = self._load_external_object_view(
                        encoded_name, rgbs.shape[1], rgbs.shape[2]
                    )
                except (OSError, KeyError, ValueError):
                    rejection_counts["empty_view"] += 1
                    continue
                canonical_points = view["points"]
                if canonical_points.shape[0] < 4:
                    rejection_counts["empty_view"] += 1
                    continue
                scale_metadata = {"scale_strategy": self.external_object_scale_strategy}
                bank_extents = view.get("robust_extents")
                if (
                    self.external_object_scale_strategy == "volume_match"
                    and target_volume_candidates
                    and bank_extents is not None
                ):
                    target_size = target_volume_candidates[
                        int(np.random.randint(0, len(target_volume_candidates)))
                    ]
                    bank_volume = float(np.prod(bank_extents))
                    raw_scale = float((target_size["volume"] / bank_volume) ** (1.0 / 3.0))
                    clamp_low = self.external_object_volume_scale_clamp_ratio[0] * target_size["gt_scale"]
                    clamp_high = self.external_object_volume_scale_clamp_ratio[1] * target_size["gt_scale"]
                    scale = float(np.clip(raw_scale, clamp_low, clamp_high))
                    scale_metadata.update({
                        "scale_strategy": "volume_match",
                        "volume_target_raw_instance_id": int(target_size["raw_instance_id"]),
                        "volume_target_object_key": target_size["object_key"],
                        "volume_target_extents": target_size["extents"].astype(float).tolist(),
                        "volume_target": float(target_size["volume"]),
                        "volume_bank_canonical_extents": np.asarray(bank_extents, dtype=float).tolist(),
                        "volume_bank_canonical": bank_volume,
                        "volume_match_scale_unclamped": raw_scale,
                        "volume_match_scale_clamped": bool(scale != raw_scale),
                    })
                else:
                    scale = float(np.random.choice(scene_scales)) * float(
                        np.random.uniform(*self.external_object_scale_jitter)
                    )
                    scale_metadata["scale_strategy"] = "scene_scale_fallback"
                scale = max(scale, self.instance_augment_room_grid_size)
                valid_cells = room_region["valid_cells"]
                if not valid_cells.size:
                    break
                anchor_cell = valid_cells[int(np.random.randint(0, valid_cells.shape[0]))]
                anchor_xy = room_region["origin"] + (
                    anchor_cell.astype(np.float64) + 0.5
                ) * room_region["grid_size"]
                nearby_xy = np.linalg.norm(points_out[:, :2] - anchor_xy[None], axis=1) <= max(0.5, scale)
                support_candidates = points_out[nearby_xy, 2]
                if support_candidates.size:
                    support_z = float(np.quantile(support_candidates, 0.05))
                else:
                    support_z = float(np.quantile(points_out[:, 2], 0.02))
                yaw_degrees = float(np.random.uniform(*self.instance_augment_yaw_degrees))
                yaw = np.deg2rad(yaw_degrees)
                rotation = np.asarray([
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                center = np.asarray([anchor_xy[0], anchor_xy[1], support_z + 0.5 * scale])
                candidate_points = canonical_points @ (scale * rotation).T + center
                candidate_corners = canonical_corners @ (scale * rotation).T + center
                if (
                    not self._points_inside_room_region(candidate_points, room_region)
                    or not self._points_inside_room_region(candidate_corners, room_region)
                ):
                    rejection_counts["outside_room"] += 1
                    continue
                candidate_voxels = self._voxel_keys(
                    candidate_points, self.instance_augment_collision_voxel_size
                )
                foreground_voxels = self._voxel_keys(
                    points_out[instance_ids_out >= 1], self.instance_augment_collision_voxel_size
                )
                foreground_ratio = len(candidate_voxels & foreground_voxels) / max(1, len(candidate_voxels))
                if foreground_ratio > self.instance_augment_max_collision_ratio:
                    rejection_counts["foreground_collision"] += 1
                    continue
                support_cutoff = float(candidate_points[:, 2].min()) + 1.5 * self.instance_augment_collision_voxel_size
                nonsupport_voxels = self._voxel_keys(
                    candidate_points[candidate_points[:, 2] > support_cutoff],
                    self.instance_augment_collision_voxel_size,
                )
                background_voxels = self._voxel_keys(
                    points_out[instance_ids_out == 0], self.instance_augment_collision_voxel_size
                )
                background_ratio = len(nonsupport_voxels & background_voxels) / max(1, len(nonsupport_voxels))
                if background_ratio > self.instance_augment_max_collision_ratio:
                    rejection_counts["background_collision"] += 1
                    continue
                accepted = (view, candidate_points, center, scale, yaw_degrees, foreground_ratio, background_ratio)
                break
            if accepted is None:
                continue
            view, candidate_points, center, scale, yaw_degrees, foreground_ratio, background_ratio = accepted
            raw_instance_id = len(object_keys) + len(operations) + 1
            object_key = f"zzzz_aug_external_{slot:03d}"
            while object_key in transforms_out:
                object_key += "_"
            dino_row_start = int(valid_masks_out.sum())
            num_external_rows = int(view["patch_mask"].sum())
            points_out = np.concatenate([points_out, candidate_points.astype(points.dtype)], axis=0)
            instance_ids_out = np.concatenate([
                instance_ids_out,
                np.full(candidate_points.shape[0], raw_instance_id, dtype=instance_ids.dtype),
            ])
            source_out = np.concatenate([
                source_out,
                dino_row_start + np.arange(num_external_rows, dtype=np.int64),
            ])
            rgbs_out = np.concatenate([rgbs_out, view["rgb"][None].astype(rgbs.dtype)], axis=0)
            valid_masks_out = np.concatenate([valid_masks_out, view["patch_mask"]])
            transforms_out[object_key] = {
                "scale": scale,
                "angles": np.asarray([0.0, 0.0, np.deg2rad(yaw_degrees)], dtype=np.float64),
                "trans": center,
            }
            operations.append({
                "operation": "external_object_paste",
                "raw_instance_id": raw_instance_id,
                "object_key": object_key,
                "bank_object": view["encoded_name"],
                "bank_source": view["source"],
                "bank_frame": int(view["frame_index"]),
                "point_count": int(candidate_points.shape[0]),
                "dino_row_start": dino_row_start,
                "dino_row_count": num_external_rows,
                "scale": scale,
                **scale_metadata,
                "yaw_degrees": yaw_degrees,
                "translation": center.astype(float).tolist(),
                "foreground_collision_ratio": float(foreground_ratio),
                "background_collision_ratio": float(background_ratio),
                "rejection_counts": rejection_counts,
            })

        if not operations:
            return (
                points, instance_ids, objects_transforms, rgbs,
                valid_point_masks, point_source_indices, metadata,
            )
        if metadata is None:
            metadata = {
                "operation": "external_object_paste",
                "object_ratio": 0.0,
                "eligible_objects": 0,
                "selected_objects": target_count,
                "committed_objects": 0,
                "raw_instance_ids": [],
                "visible_raw_instance_ids": [],
                "deleted_raw_instance_ids": [],
                "points_before": int(points.shape[0]),
                "operations": [],
            }
        metadata["operations"].extend(operations)
        metadata["raw_instance_ids"].extend(op["raw_instance_id"] for op in operations)
        metadata["visible_raw_instance_ids"].extend(op["raw_instance_id"] for op in operations)
        metadata["committed_objects"] += len(operations)
        metadata["points_after"] = int(points_out.shape[0])
        metadata["external_objects_target"] = int(target_count)
        metadata["external_objects_accepted"] = int(len(operations))
        return (
            points_out, instance_ids_out, transforms_out, rgbs_out,
            valid_masks_out, source_out, metadata,
        )

    def _augment_instances(
        self, points, instance_ids, objects_transforms,
        apply_instance_augment=None,
    ):
        """First-version instance augmentation: spatial deletion or in-place yaw.

        The method is called only when the option is enabled. It returns a
        source-row map only when an edit is committed, keeping the disabled
        path exactly unchanged.
        """
        if (
            self.split != "train"
            or not self.instance_augment_enabled
            or not self.enable_instance_augment
            or not self.instance_augment_operation_names
            or apply_instance_augment is False
        ):
            return points, instance_ids, objects_transforms, None, None
        if (
            apply_instance_augment is None
            and np.random.rand() >= self.instance_augment_probability
        ):
            return points, instance_ids, objects_transforms, None, None

        unique_ids, counts = np.unique(instance_ids, return_counts=True)
        eligible_ids = unique_ids[
            (unique_ids >= 1)
            & (counts >= self.instance_augment_min_source_points)
        ]
        if eligible_ids.size == 0:
            return points, instance_ids, objects_transforms, None, None

        effective_object_ratio = self.instance_augment_object_ratio * self.augmentation_intensity
        num_selected = max(
            1,
            min(
                eligible_ids.size,
                int(np.ceil(eligible_ids.size * effective_object_ratio)),
            ),
        )
        selected_ids = np.random.choice(
            eligible_ids,
            size=num_selected,
            replace=False,
        )

        points_out = points.copy()
        instance_ids_out = instance_ids.copy()
        point_source_indices = np.arange(points.shape[0], dtype=np.int64)
        objects_transforms_out = objects_transforms
        object_keys = sorted(
            key for key in objects_transforms
            if not key.startswith("layout_")
        )
        operations = []
        room_region = None
        copies_committed = 0
        visible_object_count = int(np.count_nonzero(unique_ids >= 1))

        pending_instances = [(int(selected_id), None) for selected_id in selected_ids]
        copy_requests_by_source = {}
        copy_attempts_by_source = {}
        for selected_id_value, forced_operation in pending_instances:
            selected_id = int(selected_id_value)
            selected_mask = instance_ids_out == selected_id
            selected_points = points_out[selected_mask]
            if selected_points.shape[0] < self.instance_augment_min_source_points:
                continue
            operation = forced_operation or str(np.random.choice(
                self.instance_augment_operation_names,
                p=self.instance_augment_operation_probs,
            ))
            if operation == "copy_paste" and forced_operation is None:
                copy_count_values = np.arange(
                    self.instance_augment_copy_count_min,
                    self.instance_augment_copy_count_max + 1,
                    dtype=np.int64,
                )
                max_copy_count = max(1, int(np.ceil(
                    self.instance_augment_copy_count_max * self.augmentation_intensity
                )))
                allowed = copy_count_values <= max_copy_count
                allowed_probs = self.instance_augment_copy_count_probs[allowed]
                allowed_probs = allowed_probs / allowed_probs.sum()
                requested_copies = int(np.random.choice(
                    copy_count_values[allowed], p=allowed_probs,
                ))
                copy_requests_by_source[selected_id] = requested_copies
                pending_instances.extend(
                    (selected_id, "copy_paste") for _ in range(requested_copies - 1)
                )

            if operation == "partial_delete":
                direction = np.random.normal(size=3)
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-8:
                    continue
                direction /= direction_norm
                projections = (selected_points - selected_points.mean(axis=0)) @ direction
                delete_ratio = np.random.uniform(*self.instance_augment_delete_ratio) * self.augmentation_intensity
                delete_low_side = bool(np.random.randint(0, 2))
                quantile = delete_ratio if delete_low_side else 1.0 - delete_ratio
                threshold = np.quantile(projections, quantile)
                delete_selected = projections <= threshold if delete_low_side else projections >= threshold
                remaining = selected_points.shape[0] - int(delete_selected.sum())
                if remaining < self.instance_augment_min_remaining_points:
                    continue

                delete_mask = np.zeros(points_out.shape[0], dtype=bool)
                delete_mask[np.flatnonzero(selected_mask)[delete_selected]] = True
                keep_mask = ~delete_mask
                deleted_points = int(delete_mask.sum())
                operations.append({
                    "operation": operation,
                    "raw_instance_id": selected_id,
                    "deleted_points": deleted_points,
                    "delete_ratio": float(deleted_points / selected_points.shape[0]),
                    "plane_direction": direction.astype(float).tolist(),
                })
                points_out = points_out[keep_mask]
                instance_ids_out = instance_ids_out[keep_mask]
                point_source_indices = point_source_indices[keep_mask]
                continue

            if operation == "full_delete":
                if selected_id > len(object_keys):
                    continue
                object_key = object_keys[selected_id - 1]
                keep_mask = ~selected_mask
                deleted_points = int(selected_mask.sum())
                operations.append({
                    "operation": operation,
                    "raw_instance_id": selected_id,
                    "object_key": object_key,
                    "deleted_points": deleted_points,
                    "delete_ratio": 1.0,
                })
                points_out = points_out[keep_mask]
                instance_ids_out = instance_ids_out[keep_mask]
                point_source_indices = point_source_indices[keep_mask]
                continue

            if operation in ("transform", "copy_paste"):
                if selected_id > len(object_keys):
                    continue
                is_copy = operation == "copy_paste"
                if is_copy:
                    copy_attempts_by_source[selected_id] = (
                        copy_attempts_by_source.get(selected_id, 0) + 1
                    )
                if is_copy and (
                    copies_committed >= self.instance_augment_max_copies_per_scene
                    or visible_object_count + copies_committed >= MAX_SCENE_OBJECTS - 1
                ):
                    continue
                if room_region is None:
                    room_region = self._build_all_point_room_region(points)

                object_key = object_keys[selected_id - 1]
                transform_entry = objects_transforms_out[object_key]
                center = np.asarray(transform_entry["trans"], dtype=np.float64)
                angles_before = np.asarray(transform_entry["angles"], dtype=np.float64).copy()
                scale_before = float(transform_entry["scale"])
                footprint_size = max(
                    self.instance_augment_room_grid_size,
                    float(np.linalg.norm(np.ptp(selected_points[:, :2], axis=0))),
                )
                bottom = float(np.quantile(selected_points[:, 2], 0.02))
                pivot = np.asarray([center[0], center[1], bottom], dtype=np.float64)
                canonical_corners = np.asarray([
                    [x, y, z]
                    for x in (-0.5, 0.5)
                    for y in (-0.5, 0.5)
                    for z in (-0.5, 0.5)
                ], dtype=np.float64)
                object_rotation = _euler_to_rotation_matrix_sxyz(
                    transform_entry["angles"]
                )
                original_obb_corners = (
                    canonical_corners @ (scale_before * object_rotation).T + center
                )

                if is_copy:
                    other_foreground = points_out[instance_ids_out >= 1]
                else:
                    other_foreground = points_out[
                        (instance_ids_out >= 1) & (instance_ids_out != selected_id)
                    ]
                other_voxels = self._voxel_keys(
                    other_foreground, self.instance_augment_collision_voxel_size
                )
                background_voxels = self._voxel_keys(
                    points_out[instance_ids_out == 0],
                    self.instance_augment_collision_voxel_size,
                )
                rejection_counts = {
                    "translation_too_small": 0,
                    "outside_room": 0,
                    "foreground_collision": 0,
                    "background_collision": 0,
                }
                accepted = None
                valid_cells = room_region["valid_cells"]
                if valid_cells.size:
                    anchor_indices = np.random.randint(
                        0,
                        valid_cells.shape[0],
                        size=self.instance_augment_translation_candidates,
                    )
                    for anchor_index in anchor_indices:
                        anchor_cell = valid_cells[int(anchor_index)]
                        anchor_xy = room_region["origin"] + (
                            anchor_cell.astype(np.float64) + 0.5
                        ) * room_region["grid_size"]
                        translation_xy = anchor_xy - center[:2]
                        if np.linalg.norm(translation_xy) < (
                            self.instance_augment_min_translation_size_ratio * footprint_size
                        ):
                            rejection_counts["translation_too_small"] += 1
                            continue

                        for _ in range(self.instance_augment_yaw_candidates):
                            yaw_degrees = float(np.random.uniform(*self.instance_augment_yaw_degrees)) * self.augmentation_intensity
                            yaw = np.deg2rad(yaw_degrees)
                            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                            rotation = np.asarray([
                                [cos_yaw, -sin_yaw, 0.0],
                                [sin_yaw, cos_yaw, 0.0],
                                [0.0, 0.0, 1.0],
                            ], dtype=np.float64)
                            scale_target = float(np.random.uniform(*self.instance_augment_scale_range))
                            scale_factor = 1.0 + self.augmentation_intensity * (scale_target - 1.0)
                            linear = scale_factor * rotation
                            translation = np.asarray([
                                translation_xy[0], translation_xy[1], 0.0
                            ], dtype=np.float64)
                            world_delta = np.eye(4, dtype=np.float64)
                            world_delta[:3, :3] = linear
                            world_delta[:3, 3] = pivot + translation - linear @ pivot
                            candidate_points = self._apply_world_delta(selected_points, world_delta)
                            candidate_obb_corners = self._apply_world_delta(
                                original_obb_corners, world_delta
                            )

                            if (
                                not self._points_inside_room_region(candidate_points, room_region)
                                or not self._points_inside_room_region(candidate_obb_corners, room_region)
                            ):
                                rejection_counts["outside_room"] += 1
                                continue
                            candidate_voxels = self._voxel_keys(
                                candidate_points, self.instance_augment_collision_voxel_size
                            )
                            denominator = max(1, len(candidate_voxels))
                            foreground_ratio = len(candidate_voxels & other_voxels) / denominator
                            if foreground_ratio > self.instance_augment_max_collision_ratio:
                                rejection_counts["foreground_collision"] += 1
                                continue

                            # Ignore the bottom support slab when comparing to
                            # background voxels so floor contact remains legal.
                            support_cutoff = float(candidate_points[:, 2].min()) + (
                                1.5 * self.instance_augment_collision_voxel_size
                            )
                            nonsupport_voxels = self._voxel_keys(
                                candidate_points[candidate_points[:, 2] > support_cutoff],
                                self.instance_augment_collision_voxel_size,
                            )
                            background_ratio = len(nonsupport_voxels & background_voxels) / max(
                                1, len(nonsupport_voxels)
                            )
                            if background_ratio > self.instance_augment_max_collision_ratio:
                                rejection_counts["background_collision"] += 1
                                continue

                            accepted = (
                                world_delta,
                                candidate_points,
                                yaw_degrees,
                                scale_factor,
                                translation_xy,
                                foreground_ratio,
                                background_ratio,
                            )
                            break
                        if accepted is not None:
                            break

                if accepted is None:
                    continue
                (
                    world_delta,
                    candidate_points,
                    yaw_degrees,
                    scale_factor,
                    translation_xy,
                    foreground_ratio,
                    background_ratio,
                ) = accepted
                new_scale, new_angles, new_trans = transform_6d_from_transform_np(
                    transform_entry["scale"],
                    transform_entry["angles"],
                    transform_entry["trans"],
                    world_delta,
                )
                if objects_transforms_out is objects_transforms:
                    objects_transforms_out = copy.deepcopy(objects_transforms)
                target_raw_instance_id = selected_id
                target_object_key = object_key
                if is_copy:
                    target_raw_instance_id = len(object_keys) + copies_committed + 1
                    target_object_key = f"zz_aug_copy_{copies_committed:03d}"
                    while target_object_key in objects_transforms_out:
                        target_object_key += "_"
                    selected_source_indices = point_source_indices[selected_mask].copy()
                    points_out = np.concatenate([
                        points_out,
                        candidate_points.astype(points.dtype, copy=False),
                    ], axis=0)
                    instance_ids_out = np.concatenate([
                        instance_ids_out,
                        np.full(candidate_points.shape[0], target_raw_instance_id, dtype=instance_ids_out.dtype),
                    ], axis=0)
                    point_source_indices = np.concatenate([
                        point_source_indices,
                        selected_source_indices,
                    ], axis=0)
                    objects_transforms_out[target_object_key] = {
                        "scale": new_scale,
                        "angles": new_angles,
                        "trans": new_trans,
                    }
                    copies_committed += 1
                else:
                    points_out[selected_mask] = candidate_points.astype(points.dtype, copy=False)
                    objects_transforms_out[object_key]["scale"] = new_scale
                    objects_transforms_out[object_key]["angles"] = new_angles
                    objects_transforms_out[object_key]["trans"] = new_trans
                operations.append({
                    "operation": operation,
                    "raw_instance_id": target_raw_instance_id,
                    "source_raw_instance_id": selected_id,
                    "copy_index_for_source": (
                        int(copy_attempts_by_source[selected_id]) if is_copy else None
                    ),
                    "copies_requested_for_source": (
                        int(copy_requests_by_source.get(selected_id, 1)) if is_copy else None
                    ),
                    "object_key": target_object_key,
                    "translation_xy": translation_xy.astype(float).tolist(),
                    "translation_distance": float(np.linalg.norm(translation_xy)),
                    "yaw_degrees": yaw_degrees,
                    "scale_factor": scale_factor,
                    "scale_before": scale_before,
                    "scale_after": float(new_scale),
                    "angles_before": angles_before.astype(float).tolist(),
                    "angles_after": np.asarray(new_angles, dtype=float).tolist(),
                    "foreground_collision_ratio": float(foreground_ratio),
                    "background_collision_ratio": float(background_ratio),
                    "rejection_counts": rejection_counts,
                    "room_region_source": "all_scene_points_xy",
                    "world_delta": world_delta.astype(float).tolist(),
                })
                continue

            if operation == "rotate":
                if selected_id > len(object_keys):
                    continue
                object_key = object_keys[selected_id - 1]
                transform_entry = objects_transforms_out[object_key]
                center = np.asarray(transform_entry["trans"], dtype=np.float64)
                angles_before = np.asarray(
                    transform_entry["angles"], dtype=np.float64
                ).copy()
                yaw_degrees = np.random.uniform(*self.instance_augment_yaw_degrees) * self.augmentation_intensity
                yaw = np.deg2rad(yaw_degrees)
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                rotation = np.asarray([
                    [cos_yaw, -sin_yaw, 0.0],
                    [sin_yaw, cos_yaw, 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                # One world-space delta is the single source of truth for both
                # the selected points and the GT object transform.
                world_yaw_about_center = np.eye(4, dtype=np.float64)
                world_yaw_about_center[:3, :3] = rotation
                world_yaw_about_center[:3, 3] = center - rotation @ center
                selected_points_h = np.concatenate([
                    selected_points.astype(np.float64),
                    np.ones((selected_points.shape[0], 1), dtype=np.float64),
                ], axis=1)
                rotated = (selected_points_h @ world_yaw_about_center.T)[:, :3]
                points_out[selected_mask] = rotated.astype(points.dtype, copy=False)

                new_scale, new_angles, new_trans = transform_6d_from_transform_np(
                    transform_entry["scale"],
                    transform_entry["angles"],
                    transform_entry["trans"],
                    world_yaw_about_center,
                )
                if objects_transforms_out is objects_transforms:
                    objects_transforms_out = copy.deepcopy(objects_transforms)
                objects_transforms_out[object_key]["scale"] = new_scale
                objects_transforms_out[object_key]["angles"] = new_angles
                objects_transforms_out[object_key]["trans"] = new_trans
                operations.append({
                    "operation": operation,
                    "raw_instance_id": selected_id,
                    "object_key": object_key,
                    "yaw_degrees": float(yaw_degrees),
                    "angles_before": angles_before.astype(float).tolist(),
                    "angles_after": np.asarray(new_angles, dtype=float).tolist(),
                    "world_delta": world_yaw_about_center.astype(float).tolist(),
                })

        if not operations:
            return points, instance_ids, objects_transforms, None, None

        operation_names = {item["operation"] for item in operations}
        metadata = {
            "operation": next(iter(operation_names)) if len(operation_names) == 1 else "mixed",
            "object_ratio": float(effective_object_ratio),
            "augmentation_intensity": float(self.augmentation_intensity),
            "eligible_objects": int(eligible_ids.size),
            "selected_objects": int(num_selected),
            "committed_objects": int(len(operations)),
            "raw_instance_ids": [item["raw_instance_id"] for item in operations],
            "visible_raw_instance_ids": [
                item["raw_instance_id"]
                for item in operations
                if item["operation"] != "full_delete"
            ],
            "deleted_raw_instance_ids": [
                item["raw_instance_id"]
                for item in operations
                if item["operation"] == "full_delete"
            ],
            "points_before": int(points.shape[0]),
            "points_after": int(points_out.shape[0]),
            "operations": operations,
        }
        return (
            points_out,
            instance_ids_out,
            objects_transforms_out,
            point_source_indices,
            metadata,
        )

    def _sample_frame_indices(self, num_frames, augmentation_intensity=None):
        intensity = self.augmentation_intensity if augmentation_intensity is None else float(augmentation_intensity)
        if self.split != 'train' or not self.enable_noise or intensity <= 0.0 or not self.frame_subsample or num_frames <= 1:
            return np.arange(num_frames, dtype=np.int64)

        # print(f"Sample frame indices: {num_frames}")
        # Keep the full sequence most of the time, and otherwise sample a
        # shorter clip with a strong bias toward larger clip lengths.
        # The final recipe subsamples half the samples. At lower schedule
        # intensity that probability decreases proportionally.
        if np.random.rand() >= 0.5 * intensity:
            num_keep = num_frames
        else:
            keep_ratio = np.random.power(8.0)
            num_keep = int(np.ceil(1 + (num_frames - 2) * keep_ratio))
            num_keep = np.clip(num_keep, 1, num_frames - 1)
        start_idx = np.random.randint(0, num_frames - num_keep + 1)
        return np.arange(start_idx, start_idx + num_keep, dtype=np.int64)

    def _sample_single_frame_index(
        self,
        instance_ids,
        valid_point_masks=None,
    ):
        """Sample one valid frame, preferring views with more FG instances."""
        instance_ids = np.asarray(instance_ids)
        if instance_ids.ndim < 2:
            raise ValueError(
                "single-frame sampling expects frame-major instance IDs"
            )
        num_frames = int(instance_ids.shape[0])
        if num_frames < 1:
            raise ValueError("cannot sample from an empty frame sequence")

        flat_instance_ids = instance_ids.reshape(num_frames, -1)
        if valid_point_masks is None:
            flat_valid_masks = np.ones_like(flat_instance_ids, dtype=bool)
        else:
            flat_valid_masks = np.asarray(valid_point_masks, dtype=bool).reshape(
                num_frames, -1
            )
            if flat_valid_masks.shape != flat_instance_ids.shape:
                raise ValueError(
                    "valid_point_masks and instance_ids must have matching "
                    "frame-major shapes"
                )

        valid_counts = flat_valid_masks.sum(axis=1)
        candidate_indices = np.flatnonzero(valid_counts > 0)
        if candidate_indices.size == 0:
            raise RuntimeError(
                "single-frame sampling found no valid points in any frame"
            )

        foreground_counts = np.zeros(num_frames, dtype=np.float64)
        for frame_idx in candidate_indices:
            frame_ids = flat_instance_ids[frame_idx, flat_valid_masks[frame_idx]]
            foreground_counts[frame_idx] = np.unique(
                frame_ids[frame_ids >= 1]
            ).size

        candidate_foreground_counts = foreground_counts[candidate_indices]
        if np.any(candidate_foreground_counts > 0):
            # When at least one frame sees a foreground object, do not waste a
            # single-image sample on an empty-background-only view.
            weights = np.power(
                candidate_foreground_counts,
                self.single_frame_sampling_instance_count_power,
            )
        else:
            # A scene can contain only background. Keep the input non-empty and
            # sample among its valid views without introducing a fixed bias.
            weights = np.ones(candidate_indices.shape[0], dtype=np.float64)
        probabilities = weights / weights.sum()
        selected_index = int(
            np.random.choice(candidate_indices, p=probabilities)
        )
        return np.asarray([selected_index], dtype=np.int64)

    def _select_frame_indices(
        self,
        instance_ids,
        valid_point_masks=None,
        augmentation_intensity=None,
    ):
        # This conditional deliberately precedes all work and RNG in the new
        # path. With the option disabled, the historical method is called
        # directly and consumes exactly the same random numbers as before.
        if self.split == "train" and self.single_frame_sampling_enabled:
            return self._sample_single_frame_index(
                instance_ids,
                valid_point_masks=valid_point_masks,
            )
        return self._sample_frame_indices(
            int(np.asarray(instance_ids).shape[0]),
            augmentation_intensity=augmentation_intensity,
        )

    def _apply_depth_noise(self, depths, augmentation_intensity=None):
        intensity = self.augmentation_intensity if augmentation_intensity is None else float(augmentation_intensity)
        if self.split != 'train' or not self.enable_noise or intensity <= 0.0 or self.depth_noise_std <= 0:
            return depths
        # print(f"Apply depth noise: {self.depth_noise_std}")
        noisy_depths = depths + np.random.normal(
            loc=0.0,
            scale=self.depth_noise_std * intensity,
            size=depths.shape,
        ).astype(np.float32)
        return np.clip(noisy_depths, 1e-6, None)

    def _apply_camera_noise(self, ray_dirs_world, cam_origins_world, augmentation_intensity=None):
        intensity = self.augmentation_intensity if augmentation_intensity is None else float(augmentation_intensity)
        if self.split != 'train' or not self.enable_noise or intensity <= 0.0:
            return ray_dirs_world, cam_origins_world

        if self.camera_rotation_noise_std <= 0 and self.camera_translation_noise_std <= 0:
            return ray_dirs_world, cam_origins_world

        # print(f"Apply camera noise: {self.camera_rotation_noise_std}, {self.camera_translation_noise_std}")

        num_frames = ray_dirs_world.shape[0]
        angles = np.random.normal(
            loc=0.0,
            scale=self.camera_rotation_noise_std * intensity,
            size=(num_frames, 3),
        ).astype(np.float32)
        translations = np.random.normal(
            loc=0.0,
            scale=self.camera_translation_noise_std * intensity,
            size=(num_frames, 1, 3),
        ).astype(np.float32)

        cx, sx = np.cos(angles[:, 0]), np.sin(angles[:, 0])
        cy, sy = np.cos(angles[:, 1]), np.sin(angles[:, 1])
        cz, sz = np.cos(angles[:, 2]), np.sin(angles[:, 2])

        rx = np.zeros((num_frames, 3, 3), dtype=np.float32)
        ry = np.zeros((num_frames, 3, 3), dtype=np.float32)
        rz = np.zeros((num_frames, 3, 3), dtype=np.float32)

        rx[:, 0, 0] = 1.0
        rx[:, 1, 1] = cx
        rx[:, 1, 2] = -sx
        rx[:, 2, 1] = sx
        rx[:, 2, 2] = cx

        ry[:, 0, 0] = cy
        ry[:, 0, 2] = sy
        ry[:, 1, 1] = 1.0
        ry[:, 2, 0] = -sy
        ry[:, 2, 2] = cy

        rz[:, 0, 0] = cz
        rz[:, 0, 1] = -sz
        rz[:, 1, 0] = sz
        rz[:, 1, 1] = cz
        rz[:, 2, 2] = 1.0

        rot = rz @ ry @ rx
        rot_t = np.transpose(rot, (0, 2, 1))

        ray_dirs_noisy = np.matmul(ray_dirs_world, rot_t)
        cam_origins_noisy = cam_origins_world + translations

        return ray_dirs_noisy, cam_origins_noisy

    def _reconstruct_points(self, depths, ray_dirs_world, cam_origins_world):
        return cam_origins_world + ray_dirs_world * depths[..., None]

    def __getitem__(self, idx):
        # print("self.enable_noise: ", self.enable_noise)


        # 1. Load Data
        # rgbs: (N_im, H, W, 3), depths: (N_im, H, W, 1)
        data_dict = self.data_list[idx]

        data_name = data_dict['data_name']

        transforms_path = data_dict['transforms_path']

        with open(transforms_path, 'rb') as f:
            objects_transforms = pickle.load(f)

        camera_path = data_dict['camera_path']
        use_real = data_dict['use_real']
        if use_real:
            frames_dir = data_dict['realistic_frames_dir']
        else:
            frames_dir = data_dict['frames_dir']
        depth_dir = data_dict['depth_dir']
        masks_dir = data_dict['masks_dir']
        prune_masks_dir = data_dict['prune_masks_dir']

        intrinsics, c2ws, (height, width) = read_cameras(camera_path)
        rgbs = read_rgbs(frames_dir, height, width, parallel=True, max_workers=8) # [N, H, W, 3]
        depths = read_depths(depth_dir, height, width, parallel=True, max_workers=8)
        masks, existing_indices = read_masks_v2(masks_dir, height, width, parallel=True, max_workers=8)
        if os.path.exists(prune_masks_dir):
            prune_masks = read_prune_masks(prune_masks_dir, height, width, parallel=True, max_workers=8)
            valid_point_masks = np.logical_not(prune_masks)
            valid_point_masks = downsample_valid_masks(valid_point_masks, height, width, factor=2*16)
        else:
            valid_point_masks = None

        rgbs, depths, masks, intrinsics, height, width = downsample_all(rgbs, depths, masks, intrinsics, height, width, factor=2)

        points, depths, ray_dirs_world, instance_ids = project_depth_to_world_patch_geometry_with_instance_mask(
            depths, intrinsics, c2ws,
            masks,
            downsample=16, # same as the DINO feature receptive field
        ) # points: (N, (H // 16)*(W // 16), 3)

        cam_origins_world = c2ws[:, None, :, 3].astype(np.float32)

        # A single sample-level gate controls every corruption source. Before
        # clean_steps, intensity is exactly zero and no RNG is consumed here.
        sample_augmentation_intensity = 0.0
        if (
            self.split == "train"
            and self.enable_noise
            and self.augmentation_intensity > 0.0
            and np.random.rand()
            < self.all_augmentation_probability * self.augmentation_intensity
        ):
            sample_augmentation_intensity = self.augmentation_intensity

        frame_indices = self._select_frame_indices(
            instance_ids,
            valid_point_masks=valid_point_masks,
            augmentation_intensity=sample_augmentation_intensity,
        )

        if valid_point_masks is not None:
            valid_point_masks_tmp = valid_point_masks[frame_indices]
            if valid_point_masks_tmp.sum() == 0:
                # Scene has no valid points after pruning and frame subsampling
                # so we don't subsample frames for this scene to keep at least some points for training.
                frame_indices = np.arange(points.shape[0], dtype=np.int64)

        rgbs = rgbs[frame_indices]
        points = points[frame_indices]
        depths = depths[frame_indices]
        ray_dirs_world = ray_dirs_world[frame_indices]
        cam_origins_world = cam_origins_world[frame_indices]
        instance_ids = instance_ids[frame_indices]

        if valid_point_masks is not None:
            valid_point_masks = valid_point_masks[frame_indices]

        depths = self._apply_depth_noise(depths, sample_augmentation_intensity)
        ray_dirs_world, cam_origins_world = self._apply_camera_noise(
            ray_dirs_world, cam_origins_world, sample_augmentation_intensity
        )

        if (
            self.split == 'train'
            and (
                self.depth_noise_std > 0
                or self.camera_rotation_noise_std > 0
                or self.camera_translation_noise_std > 0
            )
        ):
            points = self._reconstruct_points(depths, ray_dirs_world, cam_origins_world)

        existing_indices = np.unique(instance_ids.reshape(-1)).tolist()

        points = points.reshape(-1, 3)
        instance_ids = instance_ids.reshape(-1)

        if valid_point_masks is not None:
            valid_point_masks = valid_point_masks.reshape(-1)
            points = points[valid_point_masks]
            instance_ids = instance_ids[valid_point_masks]

        # One sample-level decision gates the in-scene mixed recipe and the
        # following external paste together. The fully disabled path consumes
        # no random number and remains byte-identical.
        apply_instance_augment = False
        if (
            self.split == "train"
            and self.instance_augment_enabled
            and self.enable_instance_augment
            and (self.instance_augment_operation_names or self.external_object_paste_enabled)
        ):
            apply_instance_augment = (
                sample_augmentation_intensity > 0.0
                and
                np.random.rand()
                < self.instance_augment_probability * self.augmentation_intensity
            )
        points, instance_ids, objects_transforms, point_source_indices, instance_augmentation = (
            self._augment_instances(
                points,
                instance_ids,
                objects_transforms,
                apply_instance_augment=apply_instance_augment,
            )
        )
        (
            points,
            instance_ids,
            objects_transforms,
            rgbs,
            valid_point_masks,
            point_source_indices,
            instance_augmentation,
        ) = self._augment_external_objects(
            points,
            instance_ids,
            objects_transforms,
            rgbs,
            valid_point_masks,
            point_source_indices,
            instance_augmentation,
            apply_instance_augment=apply_instance_augment,
        )
        if instance_augmentation is not None:
            # Copy-paste can append a new raw instance ID and transform. The
            # disabled path retains the original pre-augmentation list.
            existing_indices = np.unique(instance_ids).tolist()

        # scene-level augmentation
        points, augment_info = self.augment_points(points, augment=self.augment)

        # read the objects info and convert to tokens
        objects_tokens = read_objects_to_tokens_wo_latents_with_instance_ids(objects_transforms, augment_info, existing_indices, instance_ids)

        # Keep the raw camera-mask IDs and the spatially sorted token IDs linked
        # for augmentation visualization/debugging. This runs only when an
        # augmentation committed, preserving the disabled data path exactly.
        if instance_augmentation is not None:
            dense_instance_ids = []
            for raw_instance_id in instance_augmentation["visible_raw_instance_ids"]:
                dense_ids = np.unique(
                    objects_tokens["instance_ids"][instance_ids == raw_instance_id]
                )
                if dense_ids.size != 1:
                    raise RuntimeError(
                        f"raw instance {raw_instance_id} maps to dense IDs {dense_ids.tolist()}"
                    )
                dense_instance_ids.append(int(dense_ids[0]))
            instance_augmentation["dense_instance_ids"] = dense_instance_ids

        # change all the numpy arrays to torch tensors
        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(rgbs).float()
        if point_source_indices is not None:
            point_source_indices = torch.from_numpy(point_source_indices).long()
        if valid_point_masks is not None:
            valid_point_masks = torch.from_numpy(valid_point_masks).bool()
        objects_tokens["translations"] = torch.from_numpy(objects_tokens["translations"])
        objects_tokens["angles"] = torch.from_numpy(objects_tokens["angles"])
        objects_tokens["scales"] = torch.from_numpy(objects_tokens["scales"])
        objects_tokens["instance_ids"] = torch.from_numpy(objects_tokens["instance_ids"]).long()
        num_objects = objects_tokens["translations"].shape[0]

        result = {
            "split": self.split,
            "dataset_name": data_dict["dataset_name"],
            "data_name": data_name,
            "object_translations": objects_tokens["translations"],
            "object_angles": objects_tokens["angles"],
            "object_scales": objects_tokens["scales"],
            "num_objects": num_objects,
            "points": points,
            "rgbs": rgbs,
            "valid_point_masks": valid_point_masks,
            "instance_ids": objects_tokens["instance_ids"],
        }
        if point_source_indices is not None:
            result["point_source_indices"] = point_source_indices
            result["instance_augmentation"] = instance_augmentation
        return result

    def get_inference_data(self, idx, image_downsample = 1):
        data_dict = self.data_list[idx]

        camera_path = data_dict['camera_path']
        use_real = data_dict['use_real']
        if use_real:
            frames_dir = data_dict['realistic_frames_dir']
        else:
            frames_dir = data_dict['frames_dir']
        depth_dir = data_dict['depth_dir']
        data_name = data_dict['data_name']

        prune_masks_dir = data_dict['prune_masks_dir']

        parallel_io = True

        intrinsics, c2ws, (height, width) = read_cameras(camera_path)

        rgbs = read_rgbs(frames_dir, height, width, parallel=parallel_io)
        depths = read_depths(depth_dir, height, width, parallel=parallel_io)

        if os.path.exists(prune_masks_dir):
            prune_masks = read_prune_masks(prune_masks_dir, height, width, parallel=True, max_workers=8)
            valid_point_masks = np.logical_not(prune_masks)
        else:
            valid_point_masks = None

        points, points_rgbs = project_depth_to_points(
            rgbs, depths, intrinsics, c2ws,
            downsample=image_downsample,
        )
        points = points.reshape(-1, 3)
        points_rgbs = points_rgbs.reshape(-1, 3)
        if valid_point_masks is not None:
            valid_point_masks = valid_point_masks.reshape(-1)
            points = points[valid_point_masks]
            points_rgbs = points_rgbs[valid_point_masks]
        points, norm_transform = point_normalize(points)

        return_dict = {
            "points": points,
            "points_rgbs": points_rgbs,
            "rgbs": rgbs,
            "data_name": data_name,
            "preprocess_transform": norm_transform,
        }

        if valid_point_masks is not None:
            return_dict["mask"] = valid_point_masks

        return return_dict





def sparse_collate_fn(batch):
    """
    Custom collate to handle variable number of voxels per sample.
    Batches voxel_coords and voxel_feats for SpConv sparse tensor construction.
    """
    batch_size = len(batch)
    if batch_size == 1:
        # Fast path: just unsqueeze to add batch dimension (no copying)
        item = batch[0]
        num_objects = min(item['num_objects'], MAX_SCENE_OBJECTS)

        data_name = item['data_name']
        dataset_name = item.get("dataset_name")
        deterministic_scene_index = item.get("deterministic_scene_index")
        deterministic_augmentation_index = item.get(
            "deterministic_augmentation_index"
        )
        deterministic_seed = item.get("deterministic_seed")
        # Add batch dimension without copying
        points = item['points'].unsqueeze(0)
        rgbs = item['rgbs'].unsqueeze(0)
        valid_point_masks = item['valid_point_masks'].unsqueeze(0) if item['valid_point_masks'] is not None else None
        point_source_indices = item.get('point_source_indices')
        if point_source_indices is not None:
            point_source_indices = point_source_indices.unsqueeze(0)
        instance_ids = item['instance_ids'].unsqueeze(0)
        instance_ids[instance_ids>=MAX_SCENE_OBJECTS] = 0
        object_translations = item['object_translations'][:num_objects].unsqueeze(0)
        object_angles = item['object_angles'][:num_objects].unsqueeze(0)
        object_scales = item['object_scales'][:num_objects].unsqueeze(0)
        object_valid_masks = torch.ones(1, num_objects, dtype=torch.bool)
        max_num_objects = num_objects
        point_valid_masks = None
    else:
        max_num_points = max(int(item['points'].shape[0]) for item in batch)
        max_num_objects = max(
            min(int(item['num_objects']), MAX_SCENE_OBJECTS)
            for item in batch
        )
        if max_num_points <= 0:
            raise ValueError("cannot collate a batch with no points")
        if max_num_objects <= 0:
            raise ValueError("cannot collate a batch with no object tokens")

        first_rgb_shape = batch[0]['rgbs'].shape
        if len(first_rgb_shape) != 4:
            raise ValueError(
                f"expected rgbs shape [F,H,W,C], got {tuple(first_rgb_shape)}"
            )
        _, height, width, channels = first_rgb_shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "RGB height/width must be divisible by DINO patch size 16, "
                f"got {(height, width)}"
            )
        max_num_frames = max(int(item['rgbs'].shape[0]) for item in batch)
        patch_rows_per_frame = (height // 16) * (width // 16)
        max_source_rows = max_num_frames * patch_rows_per_frame

        points = torch.zeros(
            batch_size, max_num_points, 3, dtype=batch[0]['points'].dtype
        )
        instance_ids = torch.zeros(
            batch_size, max_num_points, dtype=batch[0]['instance_ids'].dtype
        )
        point_valid_masks = torch.zeros(
            batch_size, max_num_points, dtype=torch.bool
        )
        point_source_indices = torch.zeros(
            batch_size, max_num_points, dtype=torch.long
        )
        rgbs = torch.zeros(
            batch_size,
            max_num_frames,
            height,
            width,
            channels,
            dtype=batch[0]['rgbs'].dtype,
        )
        valid_point_masks = torch.zeros(
            batch_size, max_source_rows, dtype=torch.bool
        )
        object_translations = torch.zeros(
            batch_size,
            max_num_objects,
            3,
            dtype=batch[0]['object_translations'].dtype,
        )
        object_angles = torch.zeros(
            batch_size,
            max_num_objects,
            3,
            dtype=batch[0]['object_angles'].dtype,
        )
        object_scales = torch.zeros(
            batch_size,
            max_num_objects,
            1,
            dtype=batch[0]['object_scales'].dtype,
        )
        object_valid_masks = torch.zeros(
            batch_size, max_num_objects, dtype=torch.bool
        )

        data_name = []
        dataset_name = []
        deterministic_scene_index = []
        deterministic_augmentation_index = []
        deterministic_seed = []
        item = batch[0]
        for batch_index, item in enumerate(batch):
            if item['rgbs'].shape[1:] != (height, width, channels):
                raise ValueError(
                    "all RGB tensors in a batch must share [H,W,C], got "
                    f"{tuple(item['rgbs'].shape[1:])} vs {(height, width, channels)}"
                )
            num_points = int(item['points'].shape[0])
            num_frames = int(item['rgbs'].shape[0])
            num_objects = min(int(item['num_objects']), MAX_SCENE_OBJECTS)
            source_rows = num_frames * patch_rows_per_frame

            points[batch_index, :num_points] = item['points']
            rgbs[batch_index, :num_frames] = item['rgbs']
            point_valid_masks[batch_index, :num_points] = True

            item_instance_ids = item['instance_ids'].clone()
            item_instance_ids[item_instance_ids >= MAX_SCENE_OBJECTS] = 0
            instance_ids[batch_index, :num_points] = item_instance_ids

            source_mask = item['valid_point_masks']
            if source_mask is None:
                source_mask = torch.ones(source_rows, dtype=torch.bool)
            else:
                source_mask = source_mask.reshape(-1).bool()
                if source_mask.numel() != source_rows:
                    raise ValueError(
                        "valid_point_masks/source rows mismatch for "
                        f"{item['data_name']}: {source_mask.numel()} vs {source_rows}"
                    )
            valid_point_masks[batch_index, :source_rows] = source_mask

            item_source_indices = item.get('point_source_indices')
            if item_source_indices is None:
                source_count = int(source_mask.sum().item())
                if source_count != num_points:
                    raise ValueError(
                        "cannot infer point_source_indices when point count "
                        f"differs from valid source rows for {item['data_name']}: "
                        f"{num_points} vs {source_count}"
                    )
                item_source_indices = torch.arange(num_points, dtype=torch.long)
            else:
                item_source_indices = item_source_indices.reshape(-1).long()
                if item_source_indices.numel() != num_points:
                    raise ValueError(
                        "point_source_indices/points mismatch for "
                        f"{item['data_name']}: "
                        f"{item_source_indices.numel()} vs {num_points}"
                    )
            point_source_indices[batch_index, :num_points] = item_source_indices

            object_translations[batch_index, :num_objects] = (
                item['object_translations'][:num_objects]
            )
            object_angles[batch_index, :num_objects] = (
                item['object_angles'][:num_objects]
            )
            object_scales[batch_index, :num_objects] = (
                item['object_scales'][:num_objects]
            )
            object_valid_masks[batch_index, :num_objects] = True

            data_name.append(item['data_name'])
            dataset_name.append(item.get("dataset_name"))
            deterministic_scene_index.append(
                item.get("deterministic_scene_index")
            )
            deterministic_augmentation_index.append(
                item.get("deterministic_augmentation_index")
            )
            deterministic_seed.append(item.get("deterministic_seed"))

    return {
        # Point cloud data (to be voxelized on GPU)
        "data_name": data_name,
        "dataset_name": dataset_name,
        "deterministic_scene_index": deterministic_scene_index,
        "deterministic_augmentation_index": deterministic_augmentation_index,
        "deterministic_seed": deterministic_seed,
        "points": points,                                # (B, N, 3)
        "rgbs": rgbs,                                # (B, num_frames, output_H, output_W, 3)
        "instance_ids": instance_ids,                    # (B, N)
        "valid_point_masks": valid_point_masks,          # (B, source DINO rows)
        "point_valid_masks": point_valid_masks,          # (B, N) for padded point rows
        "point_source_indices": point_source_indices,    # (B, N_aug), indices after valid mask
        # Object tokens
        "object_translations": object_translations,      # (B, max_num_objects, 3)
        "object_angles": object_angles,                  # (B, max_num_objects, 3)
        "object_scales": object_scales,                  # (B, max_num_objects, 1)
        "object_valid_masks": object_valid_masks,        # (B, max_num_objects)
        "max_num_objects": max_num_objects,
    }


@contextmanager
def _fixed_rng(seed):
    """Temporarily seed CPU RNGs and restore caller state afterwards."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


class DeterministicAugmentedSceneDataset(Dataset):
    """Repeat fixed validation scenes with stable, index-addressed augmentations."""

    def __init__(self, base_dataset, scene_indices, repeats=5, seed=20260808):
        self.base_dataset = base_dataset
        self.scene_indices = [int(index) for index in scene_indices]
        self.repeats = int(repeats)
        self.seed = int(seed)
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        base_dataset.enable_noise = True
        base_dataset.enable_instance_augment = True
        base_dataset.augmentation_intensity = 1.0

    def __len__(self):
        return len(self.scene_indices) * self.repeats

    def selected_scene_manifest(self):
        manifest = []
        for scene_position, source_index in enumerate(self.scene_indices):
            item = self.base_dataset.data_list[source_index]
            manifest.append({
                "scene_position": int(scene_position),
                "source_index": int(source_index),
                "dataset_name": str(item.get("dataset_name", "")),
                "data_name": str(item.get("data_name", "")),
                "transforms_path": str(item.get("transforms_path", "")),
                "augmentation_seeds": [
                    int(self.seed + source_index * 1009 + aug_index * 9176)
                    for aug_index in range(self.repeats)
                ],
            })
        return manifest

    def __getitem__(self, index):
        scene_position, augmentation_index = divmod(index, self.repeats)
        source_index = self.scene_indices[scene_position]
        sample_seed = self.seed + source_index * 1009 + augmentation_index * 9176
        with _fixed_rng(sample_seed):
            sample = self.base_dataset[source_index]
        sample["deterministic_scene_index"] = scene_position
        sample["deterministic_augmentation_index"] = augmentation_index
        sample["deterministic_seed"] = sample_seed
        return sample


def _new_dataset(config, split, *, augment_validation=False):
    return VoxelizationDataset(
        data_dir_info=config['data_dir_info'],
        scene_scale=POS_MAX,
        split=split,
        train_split_ratio=config.get('train_split_ratio', 0.9),
        train_val_split=config.get('train_val_split', True),
        num_max_scenes=config.get('num_max_scenes', None),
        augment=config.get('augment', True) if (split == 'train' or augment_validation) else False,
        depth_noise_std=config.get('depth_noise_std', 0.0),
        camera_translation_noise_std=config.get('camera_translation_noise_std', 0.0),
        camera_rotation_noise_std=config.get('camera_rotation_noise_std', 0.0),
        frame_subsample=config.get('frame_subsample', False),
        sample_full_frame_prob=config.get('sample_full_frame_prob', 0.1),
        single_frame_sampling=config.get('single_frame_sampling', None),
        all_augmentation_probability=config.get('all_augmentation_probability', 1.0),
        split_manifest_dir=config.get('split_manifest_dir', DEFAULT_SPLIT_DIR),
        split_manifest_prefix=config.get('split_manifest_prefix', DEFAULT_SPLIT_PREFIX),
        use_split_manifests=config.get('use_split_manifests', True),
        strict_split_manifests=config.get('strict_split_manifests', True),
        instance_augment=config.get('instance_augment', None),
        scene_yaw_augmentation=config.get('scene_yaw_augmentation', None),
    )


def get_dataloader(config, split):
    """
    Create a DataLoader for the VoxelizationDataset.

    Args:
        config: Data config dict with keys:
            - data_dir: Path to the data directory
            - train_split_ratio: Train/val split ratio (default: 0.9)
            - batch_size: Batch size (default: 1)
            - num_workers: Number of dataloader workers (default: 4)
            - parallel_io: Use multi-threading for file reading (default: True)
        split: 'train' or 'val'

    Returns:
        DataLoader instance
    """
    dataset = _new_dataset(config, split)

    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 1) if split == 'train' else 1,
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 4),
        collate_fn=sparse_collate_fn,
        pin_memory=True,
        drop_last=(split == 'train'),
        # Add prefetch_factor
        prefetch_factor=4 if split == 'train' and config.get('num_workers', 8) > 0 else None,
        # Keep workers alive between epochs
        persistent_workers=True if split == 'train' and config.get('num_workers', 8) > 0 else False,
    )

    return dataloader

def get_dataset(config, split):
    return _new_dataset(config, split)


def get_deterministic_val_dataset(config, num_scenes, repeats=5, seed=20260808):
    """Create the fixed 8x5 mini-val or 100x5 checkpoint-selection set.

    The base dataset remains the legacy point-normalized det/seg dataset.
    Its item path always calls ``augment_points()``, which applies
    ``point_normalize()`` before object pose discretization.
    """

    base = _new_dataset(config, "val", augment_validation=True)
    unique_indices, seen = [], set()
    for index, item in enumerate(base.data_list):
        identity = str(item.get("transforms_path", item.get("data_name", index)))
        if identity not in seen:
            seen.add(identity)
            unique_indices.append(index)
    if len(unique_indices) < num_scenes:
        raise ValueError(
            f"requested {num_scenes} unique val scenes, found {len(unique_indices)}"
        )
    rng = np.random.RandomState(seed)
    chosen = np.asarray(unique_indices)[
        rng.choice(len(unique_indices), size=num_scenes, replace=False)
    ].tolist()
    return DeterministicAugmentedSceneDataset(base, chosen, repeats, seed)
