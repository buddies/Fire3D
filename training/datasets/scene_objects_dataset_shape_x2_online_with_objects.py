"""Online Shape VAE X2 dataset built from raw TRELLIS2 shape latents.

This is intentionally forked from scene_objects_dataset_feat_multiple_with_objects
so the legacy precomputed feature-latent training path remains unchanged.
"""
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from utils.read_frames import (
    read_rgbs,
    read_depths,
    read_masks_v2,
    read_cameras,
)
import pickle
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from utils.project import (
    downsample_all,
    downsample_valid_masks,
    project_depth_to_world_patch_geometry_with_instance_mask,
)
from utils.read_objects import read_objects_to_tokens_with_instance_ids_raw_shape
from utils.transforms import point_augment, point_normalize, get_transform_matrix_batch, _rotation_matrix_to_euler_sxyz_batch, transform_6d_from_transform_batch
from utils.discrete import discrete_transform_batch, continue_transform_batch
from utils.constants import MAX_SCENE_OBJECTS, POS_MIN, POS_MAX, SCALE_MIN, SCALE_MAX
from training.datasets.split_manifest_utils import (
    DEFAULT_SPLIT_DIR,
    load_interleaved_split,
    load_object_split,
    materialize_interleaved_entries,
    reorder_object_items_from_split,
    stable_uint32,
)

DEFAULT_SPLIT_PREFIX = "obj_gen_feat_with_objects"
DINO_PATCH_DOWNSAMPLE = 16


def _normalize_shape_rotation_label(rotation):
    label = str(rotation).strip().lower().replace("rot", "")
    if label == "":
        raise ValueError("Empty object_shape_trellis2_rotation entry")
    try:
        return f"{int(label):03d}"
    except ValueError as exc:
        raise ValueError(f"Invalid object_shape_trellis2_rotation entry: {rotation!r}") from exc


def _normalize_shape_rotations(rotations):
    if isinstance(rotations, (list, tuple)):
        raw_rotations = rotations
    else:
        raw_rotations = [rotations]
    normalized = []
    for rotation in raw_rotations:
        label = _normalize_shape_rotation_label(rotation)
        if label not in normalized:
            normalized.append(label)
    if not normalized:
        raise ValueError("object_shape_trellis2_rotation must contain at least one rotation")
    return normalized


try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _reconstruct_points_numba(depths, ray_dirs_world, cam_origins_world):
        num_frames, num_points = depths.shape[0], depths.shape[1]
        points = np.empty((num_frames, num_points, 3), dtype=ray_dirs_world.dtype)
        for frame_idx in prange(num_frames):
            for point_idx in range(num_points):
                depth = depths[frame_idx, point_idx]
                for coord_idx in range(3):
                    points[frame_idx, point_idx, coord_idx] = (
                        cam_origins_world[frame_idx, 0, coord_idx]
                        + ray_dirs_world[frame_idx, point_idx, coord_idx] * depth
                    )
        return points

    @njit(cache=True)
    def _compute_recall_replacements_numba(
        dropped_ids_int,
        kept_ids_int,
        object_centers,
        object_scales,
        choose_neighbor_mask,
        radius_sq,
        scale_ratio_thresh,
    ):
        replacement_ids = np.zeros(dropped_ids_int.shape[0], dtype=dropped_ids_int.dtype)
        for dropped_idx in range(dropped_ids_int.shape[0]):
            if not choose_neighbor_mask[dropped_idx]:
                continue

            dropped_obj_idx = dropped_ids_int[dropped_idx] - 1
            dropped_scale = object_scales[dropped_obj_idx]
            best_dist_sq = np.inf
            best_id = 0

            for kept_idx in range(kept_ids_int.shape[0]):
                kept_obj_idx = kept_ids_int[kept_idx] - 1
                kept_scale = object_scales[kept_obj_idx]
                denom = max(max(dropped_scale, kept_scale), 1e-6)
                scale_ratio_diff = abs(dropped_scale - kept_scale) / denom
                if scale_ratio_diff > scale_ratio_thresh:
                    continue

                dist_sq = 0.0
                for coord_idx in range(3):
                    diff = (
                        object_centers[dropped_obj_idx, coord_idx]
                        - object_centers[kept_obj_idx, coord_idx]
                    )
                    dist_sq += diff * diff
                if dist_sq <= radius_sq and dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_id = kept_ids_int[kept_idx]

            replacement_ids[dropped_idx] = best_id
        return replacement_ids

class VoxelizationDataset(Dataset):
    def __init__(
        self,
        data_dir_info, # dict
        scene_scale=POS_MAX,
        split='train',
        train_split_ratio=0.9,
        train_val_split=True,
        augment=True,
        train_num_max_objects=64,
        num_max_scenes=None, # only use the first x scenes for training for overfitting
        depth_noise_std=0.0,
        camera_translation_noise_std=0.0,
        camera_rotation_noise_std=0.0,
        perception_translation_noise_std=0.0,
        perception_rotation_noise_std=0.0,
        perception_scale_noise_std=0.0,
        perception_instance_id_recall=1.0,
        perception_instance_id_recall_background_prob=0.25,
        perception_instance_id_corruption_prob=0.0,
        perception_instance_id_corruption_background_prob=0.25,
        perception_instance_id_corruption_near_radius=2.0,
        perception_instance_id_recall_scale_ratio_thresh=0.5,
        frame_subsample=False,
        sample_full_frame_prob=0.1,
        scene_data_weight=1,
        object_data_weight=3,
        num_object_data_per_sample=64,
        object_data_sources=None,
        object_data_grid_cols=8,
        object_data_spacing=1.2,
        object_data_scale_min=0.8,
        object_data_scale_max=1.2,
        object_shape_trellis2_rotation="000",
        scene_shape_trellis2_rotation=None,
        split_manifest_dir=DEFAULT_SPLIT_DIR,
        split_manifest_prefix=DEFAULT_SPLIT_PREFIX,
        use_split_manifests=True,
        strict_split_manifests=True,
        dino_upsample=1,
    ):
        self.data_dir_info = data_dir_info
        self.split = split

        self.scene_scale = scene_scale
        self.augment = augment
        self.train_num_max_objects = train_num_max_objects
        self.train_val_split = bool(train_val_split)
        self.object_train_split_ratio = 0.99
        self.depth_noise_std = float(depth_noise_std)
        self.camera_translation_noise_std = float(camera_translation_noise_std)
        self.camera_rotation_noise_std = float(camera_rotation_noise_std)
        self.perception_translation_noise_std = float(perception_translation_noise_std)
        self.perception_rotation_noise_std = float(perception_rotation_noise_std)
        self.perception_scale_noise_std = float(perception_scale_noise_std)
        self.perception_instance_id_recall = float(perception_instance_id_recall)
        self.perception_instance_id_recall_background_prob = float(perception_instance_id_recall_background_prob)
        self.perception_instance_id_corruption_prob = float(perception_instance_id_corruption_prob)
        self.perception_instance_id_corruption_background_prob = float(perception_instance_id_corruption_background_prob)
        self.perception_instance_id_corruption_near_radius = float(perception_instance_id_corruption_near_radius)
        self.perception_instance_id_recall_scale_ratio_thresh = float(perception_instance_id_recall_scale_ratio_thresh)
        self.frame_subsample = bool(frame_subsample)
        self.sample_full_frame_prob = sample_full_frame_prob
        self.invalid_raw_instance_id_warning_count = 0
        self.scene_data_weight = int(scene_data_weight)
        self.object_data_weight = int(object_data_weight)
        self.num_object_data_per_sample = int(num_object_data_per_sample)
        self.object_data_sources = object_data_sources
        self.object_data_grid_cols = int(object_data_grid_cols)
        self.object_data_spacing = float(object_data_spacing)
        self.object_data_scale_min = float(object_data_scale_min)
        self.object_data_scale_max = float(object_data_scale_max)
        self.object_shape_trellis2_rotations = _normalize_shape_rotations(object_shape_trellis2_rotation)
        self.object_shape_trellis2_rotation = self.object_shape_trellis2_rotations[0]
        self.scene_shape_trellis2_rotations = (
            _normalize_shape_rotations(scene_shape_trellis2_rotation)
            if scene_shape_trellis2_rotation is not None
            else []
        )
        self.scene_shape_trellis2_rotation = (
            self.scene_shape_trellis2_rotations[0]
            if self.scene_shape_trellis2_rotations
            else None
        )
        self.split_manifest_dir = split_manifest_dir
        self.split_manifest_prefix = split_manifest_prefix
        self.use_split_manifests = bool(use_split_manifests)
        self.strict_split_manifests = bool(strict_split_manifests)
        self.dino_upsample = int(dino_upsample)
        if self.dino_upsample < 1:
            raise ValueError(f"dino_upsample must be >= 1, got {self.dino_upsample}")
        if DINO_PATCH_DOWNSAMPLE % self.dino_upsample != 0:
            raise ValueError(
                f"dino_upsample must divide {DINO_PATCH_DOWNSAMPLE}, got {self.dino_upsample}"
            )
        self.dino_patch_downsample = DINO_PATCH_DOWNSAMPLE // self.dino_upsample
        self.object_data_list = []

        if not (0.0 <= self.perception_instance_id_recall <= 1.0):
            raise ValueError(
                "perception_instance_id_recall must be in [0, 1]"
            )
        if not (0.0 <= self.perception_instance_id_recall_background_prob <= 1.0):
            raise ValueError(
                "perception_instance_id_recall_background_prob must be in [0, 1]"
            )
        if self.perception_instance_id_recall_scale_ratio_thresh < 0.0:
            raise ValueError(
                "perception_instance_id_recall_scale_ratio_thresh must be >= 0"
            )

        """
        data_dir_info

        dataset_name: {
            data_dir: str,
            use_weight: int, (>=1)
        }

        """

        if not isinstance(self.data_dir_info, dict) or len(self.data_dir_info) == 0:
            raise ValueError("data_dir_info must be a non-empty dict")

        if "scene_data" in self.data_dir_info or "object_data" in self.data_dir_info:
            scene_data_info = self.data_dir_info.get("scene_data", {})
            object_data_info = self.data_dir_info.get("object_data", None)
        else:
            scene_data_info = self.data_dir_info
            object_data_info = None

        dataset_names = sorted(scene_data_info.keys())
        for dataset_name in dataset_names:
            dataset_cfg = scene_data_info[dataset_name]
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


        for dataset_name in dataset_names:
            dataset_cfg = scene_data_info[dataset_name]
            data_dir = dataset_cfg["data_dir"]
            num_videos_per_scene = dataset_cfg["num_videos_per_scene"]
            renders_dir = os.path.join(data_dir, 'renders')
            transforms_dir = os.path.join(data_dir, 'transforms')

            unused_scene_ids_path = os.path.join("preprocess/unused_data", f"{dataset_name}.json")
            if os.path.exists(unused_scene_ids_path):
                with open(unused_scene_ids_path, 'r') as f:
                    unused_scene_ids = set(json.load(f))
                print(f"[{dataset_name}] Loaded {len(unused_scene_ids)} unused scene IDs from {unused_scene_ids_path}")
            else:
                unused_scene_ids = set()
                print(f"[{dataset_name}] No unused scene IDs file found at {unused_scene_ids_path}; using all scenes")

            shape_latents_dir = dataset_cfg.get('shape_trellis2_latents_dir', os.path.join(data_dir, 'trellis2_shape_latents'))
            pbr_latents_dir = os.path.join(data_dir, 'pbr_latents')

            if not os.path.isdir(renders_dir):
                raise ValueError(f"Dataset '{dataset_name}' renders_dir not found: {renders_dir}")
            if not os.path.isdir(transforms_dir):
                raise ValueError(f"Dataset '{dataset_name}' transforms_dir not found: {transforms_dir}")

            scene_ids = sorted(os.listdir(renders_dir))
            # debug: only use the first x scenes
            if num_max_scenes is not None:
                scene_ids = scene_ids[:num_max_scenes]
                print(
                    f"[{dataset_name}] Using only the first {num_max_scenes} scenes for overfitting"
                )

            data_list_group_by_scene = []
            unused_indices = []

            for scene_idx, scene_id in enumerate(scene_ids):

                scene_data_list = []
                for video_i in range(num_videos_per_scene):
                    shape_latents_path = os.path.join(shape_latents_dir, scene_id)

                    pbr_latents_path = os.path.join(pbr_latents_dir, scene_id, f'pbr_latents.pkl')
                    if not os.path.exists(pbr_latents_path):
                        pbr_latents_path = os.path.join(pbr_latents_dir, f'{scene_id}.pkl')

                    data_dict = {
                        'dataset_name': dataset_name,
                        'scene_id': scene_id,
                        'video_id': video_i,
                        "data_name": f"{dataset_name}_{scene_id}_{video_i}",
                        'camera_path': os.path.join(renders_dir, scene_id, f'{video_i}.json'),
                        'frames_dir': os.path.join(renders_dir, scene_id, f'{video_i}_frames'),
                        'masks_dir': os.path.join(renders_dir, scene_id, f'{video_i}_masks'),
                        'depth_dir': os.path.join(renders_dir, scene_id, f'{video_i}_depth'),
                        'shape_latents_path': shape_latents_path,
                        'pbr_latents_path': pbr_latents_path,
                        'transforms_path': os.path.join(transforms_dir, f'{scene_id}.pkl'),
                        'sample_type': 'scene',
                    }
                    if os.path.exists(data_dict['camera_path']) \
                        and os.path.exists(data_dict['frames_dir']) \
                        and os.path.exists(data_dict['masks_dir']) \
                        and os.path.exists(data_dict['depth_dir']) \
                        and os.path.exists(data_dict['pbr_latents_path']) \
                        and os.path.exists(data_dict['transforms_path']) \
                        and os.path.isdir(data_dict['shape_latents_path']):
                        scene_data_list.append(data_dict)
                    else:
                        # print(f"Skipping scene {scene_id} video {video_i} because it does not exist")
                        continue
                if len(scene_data_list) > 0:
                    data_list_group_by_scene.append(scene_data_list)

                    if scene_id in unused_scene_ids:
                        unused_indices.append(len(data_list_group_by_scene) - 1)

            print(f"[{dataset_name}] Total number of scenes: {len(data_list_group_by_scene)}")

            total_num_scenes = len(data_list_group_by_scene)
            if total_num_scenes == 0:
                split_data_by_dataset[dataset_name] = []
                continue

            if self.use_split_manifests:
                valid_scene_indices = np.arange(total_num_scenes, dtype=np.int32)
                if unused_indices:
                    valid_scene_mask = np.ones(total_num_scenes, dtype=bool)
                    valid_scene_mask[np.asarray(unused_indices, dtype=np.int32)] = False
                    valid_scene_indices = valid_scene_indices[valid_scene_mask]
                split_data = list(chain.from_iterable(data_list_group_by_scene[scene_idx] for scene_idx in valid_scene_indices))
                print(f"[{dataset_name}] Number of usable sequences before manifest filtering: {len(split_data)}")
                split_data_by_dataset[dataset_name] = split_data
                continue

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

            print(f"[{dataset_name}] Number of scenes in {self.split} set: {len(scene_indices)}")
            # split_data = [
            #     data_dict
            #     for scene_idx in scene_indices
            #     if scene_idx not in unused_indices
            #     for data_dict in data_list_group_by_scene[scene_idx]
            # ]
            valid_scene_mask = np.ones(total_num_scenes, dtype=bool)
            if unused_indices:
                valid_scene_mask[np.asarray(unused_indices, dtype=np.int32)] = False
            valid_scene_indices = np.asarray(scene_indices, dtype=np.int32)[valid_scene_mask[scene_indices]]
            split_data = list(chain.from_iterable(data_list_group_by_scene[scene_idx] for scene_idx in valid_scene_indices))

            print(f"[{dataset_name}] Number of sequences in {self.split} set: {len(split_data)}")
            split_data_by_dataset[dataset_name] = split_data

        self.scene_data_list = []
        for dataset_name in dataset_names:
            dataset_split_list = split_data_by_dataset[dataset_name]
            use_weight = int(scene_data_info[dataset_name]["use_weight"])
            if len(dataset_split_list) == 0:
                print(f"[{dataset_name}] Empty split list in {self.split}; skipping")
                continue
            self.scene_data_list.extend(dataset_split_list * use_weight)

        self.object_data_list = self._build_object_data_list(object_data_info)
        self.object_data_by_name = {item["data_name"]: item for item in self.object_data_list}

        self.data_list = []
        if self.use_split_manifests:
            object_split = load_object_split(
                self.split,
                split_dir=self.split_manifest_dir,
                prefix=self.split_manifest_prefix,
            )
            self.object_data_list, missing_object_names = reorder_object_items_from_split(
                self.object_data_list, object_split, strict=self.strict_split_manifests
            )
            if missing_object_names:
                print(
                    f"[split_manifest] Skipped {len(missing_object_names)} missing object names "
                    f"from {self.split} split"
                )
            self.object_data_by_name = {item["data_name"]: item for item in self.object_data_list}
            self.split_object_names = [item["data_name"] for item in self.object_data_list]

            interleaved_split = load_interleaved_split(
                self.split,
                split_dir=self.split_manifest_dir,
                prefix=self.split_manifest_prefix,
            )
            scene_data_by_name = {item["data_name"]: item for item in self.scene_data_list}
            self.data_list, missing_scene_names = materialize_interleaved_entries(
                interleaved_split, scene_data_by_name, strict=self.strict_split_manifests
            )
            self.scene_data_list = [
                item for item in self.data_list if item.get("sample_type") == "scene"
            ]
            if missing_scene_names:
                print(
                    f"[split_manifest] Skipped {len(missing_scene_names)} missing scene data names "
                    f"from {self.split} split"
                )
            print(
                f"[split_manifest] Loaded deterministic {self.split} order from "
                f"{self.split_manifest_dir}/{self.split_manifest_prefix}_interleaved_{self.split}.json"
            )
        else:
            self.split_object_names = [item["data_name"] for item in self.object_data_list]
            if self.scene_data_weight > 0:
                self.data_list.extend(self.scene_data_list * self.scene_data_weight)
            if self.object_data_weight > 0 and len(self.object_data_list) > 0:
                pseudo_len = max(len(self.scene_data_list), 1)
                self.data_list.extend(
                    {"sample_type": "object", "data_name": f"object_pseudo_{i:08d}", "object_start": i * self.num_object_data_per_sample}
                    for i in range(pseudo_len * self.object_data_weight)
                )

        print(f"Total number of scene sequences in {self.split} set: {len(self.scene_data_list)}")
        print(f"Total number of object render items: {len(self.object_data_list)}")
        print(f"Total number of mixed sequences in {self.split} set: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def augment_points(self, points, augment=True):

        # 2. Augment (Apply rotation matrix to points here)
        points, augment_transform = point_augment(points, augment=augment)
        points, norm_transform = point_normalize(points)

        augment_info = {
            'augment_transform': augment_transform,
            'norm_transform': norm_transform
        }

        return points, augment_info


    def _build_object_data_list(self, object_data_info):
        if object_data_info is None:
            return []

        data_dir = object_data_info.get("data_dir")
        if data_dir is None:
            raise ValueError("data_dir_info.object_data must contain data_dir")

        render_root = object_data_info.get("render_dir", os.path.join(data_dir, "render"))
        shape_latents_root = object_data_info.get(
            "shape_trellis2_latents_dir",
            os.path.join(data_dir, "shape_vae_x2_rot_aug"),
        )
        sources = object_data_info.get("sources", self.object_data_sources)
        if sources is None:
            sources = sorted(os.listdir(render_root))
        elif isinstance(sources, str):
            sources = [sources]

        object_items = []
        total_usable_object_items = 0
        for source in sources:
            source_render_dir = os.path.join(render_root, source)
            source_latent_dir = os.path.join(shape_latents_root, source, "latents", "trellis2_shape_encoding")
            if not os.path.isdir(source_render_dir) or not os.path.isdir(source_latent_dir):
                print(f"[object_data] Skipping source {source}: missing render or shape_latents dir")
                continue

            source_items = []
            for object_id in sorted(os.listdir(source_render_dir)):
                object_render_dir = os.path.join(source_render_dir, object_id)
                item = {
                    "sample_type": "object_item",
                    "dataset_name": source,
                    "object_id": object_id,
                    "data_name": f"{source}_{object_id}",
                    "camera_path": os.path.join(object_render_dir, "cameras.json"),
                    "frames_dir": os.path.join(object_render_dir, "frames"),
                    "depth_dir": os.path.join(object_render_dir, "depth"),
                    "masks_dir": os.path.join(object_render_dir, "mask"),
                    "shape_latent_path": os.path.join(source_latent_dir, f"{object_id}__rot{self.object_shape_trellis2_rotation}.npz"),
                }
                if (
                    os.path.exists(item["camera_path"])
                    and os.path.isdir(item["frames_dir"])
                    and os.path.isdir(item["depth_dir"])
                    and os.path.isdir(item["masks_dir"])
                    and os.path.exists(item["shape_latent_path"])
                ):
                    source_items.append(item)

            num_source_items = len(source_items)
            total_usable_object_items += num_source_items
            if num_source_items == 0:
                print(f"[object_data:{source}] Empty usable object list in {self.split}; skipping")
                continue

            if getattr(self, "use_split_manifests", False):
                object_items.extend(source_items)
                print(
                    f"[object_data:{source}] Number of usable objects before manifest filtering: "
                    f"{num_source_items}"
                )
                continue

            train_object_indices = np.linspace(
                0,
                num_source_items - 1,
                max(int(num_source_items * self.object_train_split_ratio), 1),
            ).astype(np.int32)
            val_object_indices = np.setdiff1d(np.arange(num_source_items), train_object_indices)
            if not self.train_val_split:
                train_object_indices = np.arange(num_source_items)

            if self.split == 'train':
                object_indices = train_object_indices
            elif self.split == 'trainval':
                object_indices = train_object_indices[:int(0.1 * len(train_object_indices) + 1)]
            elif self.split == 'val':
                object_indices = val_object_indices
            else:
                raise ValueError(f"Invalid split: {self.split}")

            source_split_items = [source_items[object_idx] for object_idx in object_indices]
            object_items.extend(source_split_items)
            print(
                f"[object_data:{source}] Number of objects in {self.split} set: "
                f"{len(source_split_items)} / {num_source_items}"
            )

        print(
            f"[object_data] Found {total_usable_object_items} usable object render items; "
            f"using {len(object_items)} for {self.split}"
        )
        return object_items

    def _read_object_camera_frame_from_data(self, camera_data, frame_idx):
        height = int(camera_data["height"])
        width = int(camera_data["width"])
        frame = camera_data["frames"][frame_idx]
        K = np.asarray(frame["intrinsics"]["matrix"], dtype=np.float32)
        extr = frame["extrinsics"]
        eye = np.asarray(extr["eye"], dtype=np.float32)
        lookat = np.asarray(extr["lookat"], dtype=np.float32)
        up_vec = np.asarray(extr["up"], dtype=np.float32)

        forward = lookat - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up_vec)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.column_stack([right, -up, forward])
        c2w[:3, 3] = eye
        return K[None], c2w[None, :3], height, width

    def _read_object_camera_frame(self, camera_path, frame_idx):
        with open(camera_path, "r") as f:
            camera_data = json.load(f)
        return self._read_object_camera_frame_from_data(camera_data, frame_idx)

    def _read_object_render_frame_arrays(self, object_item, frame_idx):
        frame_name = f"{frame_idx:03d}"
        rgb_path = os.path.join(object_item["frames_dir"], f"{frame_name}.jpg")
        depth_path = os.path.join(object_item["depth_dir"], f"{frame_name}.npz")
        mask_path = os.path.join(object_item["masks_dir"], f"{frame_name}.npz")

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        with np.load(depth_path) as depth_archive:
            depth_key = "depth" if "depth" in depth_archive else "arr_0"
            depth = depth_archive[depth_key].astype(np.float32)

        with np.load(mask_path) as mask_archive:
            mask_key = "mask" if "mask" in mask_archive else "arr_0"
            valid_mask = mask_archive[mask_key].astype(np.int32) > 0
        return rgb, depth, valid_mask.astype(np.int32)

    def _read_object_render_frame(self, object_item, frame_idx):
        rgb, depth, valid_mask = self._read_object_render_frame_arrays(object_item, frame_idx)
        return rgb[None], depth[None], valid_mask[None].astype(np.int32)

    def _load_object_shape_latent(self, object_item):
        with np.load(object_item["shape_latent_path"]) as latent_archive:
            return {
                "feats": latent_archive["feats"].astype(np.float32),
                "coords": latent_archive["coords"].astype(np.int32),
            }

    def _object_item_with_selected_shape_rotation(self, object_item, frame_seed):
        shape_latent_paths = object_item.get("shape_latent_paths")
        if not shape_latent_paths:
            return object_item

        rotations = object_item.get("shape_latent_rotations")
        if rotations is None:
            rotations = sorted(shape_latent_paths.keys())
        rotations = [rotation for rotation in rotations if rotation in shape_latent_paths]
        if len(rotations) == 0:
            return object_item

        if len(rotations) == 1:
            selected_rotation = rotations[0]
        else:
            selected_index = stable_uint32(
                object_item.get("data_name", object_item.get("object_id", "")),
                int(frame_seed),
                "shape_latent_rotation",
            ) % len(rotations)
            selected_rotation = rotations[int(selected_index)]

        selected_path = shape_latent_paths[selected_rotation]
        if (
            object_item.get("shape_latent_path") == selected_path
            and object_item.get("shape_latent_rotation") == selected_rotation
        ):
            return object_item

        selected_item = dict(object_item)
        selected_item["shape_latent_path"] = selected_path
        selected_item["shape_latent_rotation"] = selected_rotation
        return selected_item

    def _shape_latent_rotation_radians(self, object_item):
        rotation = object_item.get("shape_latent_rotation", self.object_shape_trellis2_rotation)
        return np.deg2rad(int(_normalize_shape_rotation_label(rotation)))

    def _select_scene_shape_rotation(self, data_dict):
        if not self.scene_shape_trellis2_rotations:
            return None
        if len(self.scene_shape_trellis2_rotations) == 1:
            return self.scene_shape_trellis2_rotations[0]
        selected_index = stable_uint32(
            self.split,
            data_dict.get("data_name", ""),
            data_dict.get("scene_id", ""),
            data_dict.get("video_id", 0),
            "scene_shape_latent_rotation",
        ) % len(self.scene_shape_trellis2_rotations)
        return self.scene_shape_trellis2_rotations[int(selected_index)]

    def _select_object_token_subset(
        self,
        feats_list_np,
        coords_list_np,
        object_transforms,
        instance_ids,
    ):
        num_objects = len(feats_list_np)
        if self.split == 'train' and self.train_num_max_objects is not None and num_objects > self.train_num_max_objects:
            if num_objects > 1:
                selected_indices = torch.randperm(num_objects - 1, device='cpu')[:self.train_num_max_objects - 1] + 1
                selected_indices = torch.cat([torch.tensor([0], device='cpu'), selected_indices])
            else:
                selected_indices = torch.arange(num_objects, device='cpu')
        else:
            selected_indices = torch.arange(num_objects, device='cpu')
        selected_indices = selected_indices.long()

        feats_list = [torch.from_numpy(feats_list_np[int(i)]).float() for i in selected_indices]
        object_feats = torch.cat(feats_list, dim=0).float()
        coords_list = [
            torch.cat([
                torch.full((coords_list_np[int(i)].shape[0], 1), new_idx, dtype=torch.int32),
                torch.from_numpy(coords_list_np[int(i)]).int(),
            ], dim=-1) for new_idx, i in enumerate(selected_indices)
        ]
        object_coords = torch.cat(coords_list, dim=0).int()

        remapped_instance_ids = torch.full_like(instance_ids, -1)
        if instance_ids.numel() > 0 and selected_indices.numel() > 0:
            max_existing_id = int(instance_ids.max().item())
            max_selected_id = int(selected_indices.max().item())
            max_id = max(max_existing_id, max_selected_id)
            if max_id >= 0:
                id_map = torch.full(
                    (max_id + 1,),
                    -1,
                    dtype=instance_ids.dtype,
                    device=instance_ids.device,
                )
                selected_on_device = selected_indices.to(device=instance_ids.device)
                in_range = selected_on_device <= max_id
                id_map[selected_on_device[in_range]] = torch.arange(
                    int(in_range.sum().item()),
                    dtype=instance_ids.dtype,
                    device=instance_ids.device,
                )
                valid_instance_mask = (instance_ids >= 0) & (instance_ids <= max_id)
                remapped_instance_ids[valid_instance_mask] = id_map[instance_ids[valid_instance_mask]]

        selected_object_transforms = object_transforms[selected_indices]
        selected_indices = torch.arange(selected_indices.numel(), device='cpu', dtype=torch.long)
        return (
            object_feats,
            object_coords,
            selected_object_transforms,
            remapped_instance_ids,
            int(selected_indices.numel()),
            selected_indices,
        )

    def _read_selected_object_render(self, args):
        out_idx, object_ref, frame_seed = args
        if isinstance(object_ref, dict):
            object_item = object_ref
        else:
            object_item = self.object_data_list[int(object_ref)]
        object_item = self._object_item_with_selected_shape_rotation(object_item, frame_seed)
        with open(object_item["camera_path"], "r") as f:
            camera_data = json.load(f)

        rng = np.random.default_rng(int(frame_seed))
        frame_idx = int(rng.integers(0, len(camera_data["frames"])))
        intrinsics, c2ws, height, width = self._read_object_camera_frame_from_data(
            camera_data, frame_idx
        )
        rgb, depth, mask = self._read_object_render_frame_arrays(object_item, frame_idx)
        return {
            "out_idx": out_idx,
            "object_item": object_item,
            "intrinsics": intrinsics[0],
            "c2ws": c2ws[0],
            "height": height,
            "width": width,
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
        }

    def _sample_object_transform(self, object_idx):
        cols = max(self.object_data_grid_cols, 1)
        row = object_idx // cols
        col = object_idx % cols
        grid_center = (cols - 1) * 0.5
        scale = np.random.uniform(self.object_data_scale_min, self.object_data_scale_max)
        angles = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)], dtype=np.float32)
        trans = np.array(
            [
                (col - grid_center) * self.object_data_spacing,
                (row - grid_center) * self.object_data_spacing,
                0.0,
            ],
            dtype=np.float32,
        )
        return float(scale), angles, trans

    def _make_object_to_scene_matrix(self, scale, angles, trans):
        cz, sz = np.cos(angles[2]), np.sin(angles[2])
        rot = np.array(
            [
                [cz, -sz, 0.0],
                [sz, cz, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rot * scale
        transform[:3, 3] = trans
        return transform

    def _sample_frame_indices(self, num_frames):
        if self.split != 'train' or not self.frame_subsample or num_frames <= 1:
            return np.arange(num_frames, dtype=np.int64)

        # print(f"Sample frame indices: {num_frames}")
        # Keep the full sequence most of the time, and otherwise sample a
        # shorter clip with a strong bias toward larger clip lengths.
        if np.random.rand() < self.sample_full_frame_prob:
            num_keep = num_frames
        else:
            keep_ratio = np.random.power(8.0)
            num_keep = int(np.ceil(1 + (num_frames - 2) * keep_ratio))
            num_keep = max(1, min(num_keep, num_frames - 1))
        start_idx = np.random.randint(0, num_frames - num_keep + 1)
        return np.arange(start_idx, start_idx + num_keep, dtype=np.int64)

    def _apply_depth_noise(self, depths):
        if self.split != 'train' or self.depth_noise_std <= 0:
            return depths
        # print(f"Apply depth noise: {self.depth_noise_std}")
        noise = np.random.normal(
            loc=0.0,
            scale=self.depth_noise_std,
            size=depths.shape,
        ).astype(depths.dtype, copy=False)
        noisy_depths = depths + noise
        return np.maximum(noisy_depths, np.array(1e-6, dtype=noisy_depths.dtype))

    def _apply_camera_noise(self, ray_dirs_world, cam_origins_world):
        if self.split != 'train':
            return ray_dirs_world, cam_origins_world

        apply_rotation_noise = self.camera_rotation_noise_std > 0
        apply_translation_noise = self.camera_translation_noise_std > 0
        if not apply_rotation_noise and not apply_translation_noise:
            return ray_dirs_world, cam_origins_world

        # print(f"Apply camera noise: {self.camera_rotation_noise_std}, {self.camera_translation_noise_std}")

        num_frames = ray_dirs_world.shape[0]
        cam_origins_noisy = cam_origins_world
        if apply_translation_noise:
            translations = np.random.normal(
                loc=0.0,
                scale=self.camera_translation_noise_std,
                size=(num_frames, 1, 3),
            ).astype(cam_origins_world.dtype, copy=False)
            cam_origins_noisy = cam_origins_world + translations

        if not apply_rotation_noise:
            return ray_dirs_world, cam_origins_noisy

        dtype = ray_dirs_world.dtype
        angles = np.random.normal(
            loc=0.0,
            scale=self.camera_rotation_noise_std,
            size=(num_frames, 3),
        ).astype(dtype, copy=False)

        cx, sx = np.cos(angles[:, 0]), np.sin(angles[:, 0])
        cy, sy = np.cos(angles[:, 1]), np.sin(angles[:, 1])
        cz, sz = np.cos(angles[:, 2]), np.sin(angles[:, 2])

        rot_t = np.empty((num_frames, 3, 3), dtype=dtype)
        rot_t[:, 0, 0] = cy * cz
        rot_t[:, 0, 1] = cy * sz
        rot_t[:, 0, 2] = -sy
        rot_t[:, 1, 0] = cz * sy * sx - sz * cx
        rot_t[:, 1, 1] = sz * sy * sx + cz * cx
        rot_t[:, 1, 2] = cy * sx
        rot_t[:, 2, 0] = cz * sy * cx + sz * sx
        rot_t[:, 2, 1] = sz * sy * cx - cz * sx
        rot_t[:, 2, 2] = cy * cx

        ray_dirs_noisy = np.matmul(ray_dirs_world, rot_t)

        return ray_dirs_noisy, cam_origins_noisy

    def _reconstruct_points(self, depths, ray_dirs_world, cam_origins_world):
        if _HAS_NUMBA and depths.ndim == 2 and ray_dirs_world.ndim == 3 and cam_origins_world.ndim == 3:
            return _reconstruct_points_numba(depths, ray_dirs_world, cam_origins_world)
        depths_expanded = depths[..., None]
        points = ray_dirs_world * depths_expanded
        points += cam_origins_world
        return points

    def _prepare_perception_recall_metadata(self, objects_transforms):
        object_ids = sorted(
            obj_id for obj_id in objects_transforms.keys()
            if not obj_id.startswith("layout_")
        )
        num_objects = len(object_ids)
        object_centers = np.empty((num_objects, 3), dtype=np.float32)
        object_scales = np.empty((num_objects,), dtype=np.float32)
        for obj_idx, obj_id in enumerate(object_ids):
            transform_dict = objects_transforms[obj_id]
            object_centers[obj_idx] = transform_dict["trans"]
            object_scales[obj_idx] = transform_dict["scale"]
        return object_centers, object_scales, num_objects

    def _apply_perception_recall_noise(
        self,
        instance_ids,
        existing_indices,
        object_centers,
        object_scales,
        num_objects,
        data_name,
    ):
        if self.split != 'train' or self.perception_instance_id_recall >= 1.0:
            return instance_ids, existing_indices

        existing_indices_array = np.asarray(existing_indices, dtype=instance_ids.dtype)
        foreground_ids = existing_indices_array[existing_indices_array > 0]
        valid_foreground_ids = foreground_ids[foreground_ids <= num_objects]
        invalid_foreground_ids = foreground_ids[foreground_ids > num_objects]
        if invalid_foreground_ids.size > 0:
            self.invalid_raw_instance_id_warning_count += 1
            print(
                "[VoxelizationDataset] Invalid raw instance IDs in recall noise: "
                f"data_name={data_name}, "
                f"count={invalid_foreground_ids.size}, "
                f"max_id={int(invalid_foreground_ids.max())}, "
                f"num_objects={num_objects}, "
                f"warning_idx={self.invalid_raw_instance_id_warning_count}"
            )
        num_valid_foreground_ids = valid_foreground_ids.shape[0]
        if num_valid_foreground_ids == 0:
            return instance_ids, existing_indices

        num_keep = int(np.round(self.perception_instance_id_recall * num_valid_foreground_ids))
        num_keep = np.clip(num_keep, 0, num_valid_foreground_ids)
        num_drop = num_valid_foreground_ids - num_keep
        if num_drop <= 0:
            return instance_ids, existing_indices

        dropped_ids = np.random.choice(
            valid_foreground_ids,
            size=num_drop,
            replace=False,
        )

        noisy_instance_ids = instance_ids.copy()
        dropped_ids_int = dropped_ids.astype(np.int64, copy=False)
        id_selected_mask = np.zeros(num_objects + 1, dtype=np.bool_)
        id_selected_mask[dropped_ids_int] = True
        kept_ids_int = valid_foreground_ids[~id_selected_mask[valid_foreground_ids]].astype(np.int64, copy=False)
        radius_sq = self.perception_instance_id_corruption_near_radius ** 2

        if kept_ids_int.size == 0:
            noisy_instance_ids[id_selected_mask[instance_ids]] = 0
        else:
            choose_neighbor_mask = (
                np.random.rand(dropped_ids_int.shape[0])
                >= self.perception_instance_id_recall_background_prob
            )
            if _HAS_NUMBA:
                replacement_ids = _compute_recall_replacements_numba(
                    dropped_ids_int,
                    kept_ids_int,
                    object_centers,
                    object_scales,
                    choose_neighbor_mask,
                    radius_sq,
                    self.perception_instance_id_recall_scale_ratio_thresh,
                ).astype(instance_ids.dtype, copy=False)
            else:
                kept_centers = object_centers[kept_ids_int - 1]
                kept_scales = object_scales[kept_ids_int - 1]
                dropped_centers = object_centers[dropped_ids_int - 1]
                dropped_scales = object_scales[dropped_ids_int - 1]
                center_diff = dropped_centers[:, None, :] - kept_centers[None, :, :]
                dist_sq = np.sum(center_diff * center_diff, axis=2)
                scale_ratio_diff = np.abs(
                    dropped_scales[:, None] - kept_scales[None, :]
                ) / np.maximum(
                    np.maximum(dropped_scales[:, None], kept_scales[None, :]),
                    1e-6,
                )
                valid_mask = (
                    (dist_sq <= radius_sq)
                    & (scale_ratio_diff <= self.perception_instance_id_recall_scale_ratio_thresh)
                )

                replacement_ids = np.zeros(dropped_ids_int.shape[0], dtype=instance_ids.dtype)
                candidate_mask = valid_mask & choose_neighbor_mask[:, None]
                has_candidate = np.any(candidate_mask, axis=1)
                masked_dist_sq = np.where(candidate_mask, dist_sq, np.inf)
                nearest_keep_idx = np.argmin(masked_dist_sq, axis=1)
                replacement_ids[has_candidate] = kept_ids_int[nearest_keep_idx[has_candidate]].astype(
                    instance_ids.dtype,
                    copy=False,
                )

            remap_ids = np.arange(int(instance_ids.max()) + 1, dtype=instance_ids.dtype)
            remap_ids[dropped_ids_int] = replacement_ids
            noisy_instance_ids = remap_ids[noisy_instance_ids]

        keep_mask = ~id_selected_mask[existing_indices_array]
        updated_existing_indices = existing_indices_array[keep_mask].tolist()
        return noisy_instance_ids, updated_existing_indices

    def _axis_angle_to_matrix_torch(self, axes, angles):
        axes = axes / torch.clamp(torch.linalg.norm(axes, dim=1, keepdim=True), min=1e-8)

        x = axes[:, 0]
        y = axes[:, 1]
        z = axes[:, 2]

        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)
        one_minus_cos = 1.0 - cos_theta

        rot = torch.empty((axes.shape[0], 3, 3), dtype=axes.dtype, device=axes.device)
        rot[:, 0, 0] = cos_theta + x * x * one_minus_cos
        rot[:, 0, 1] = x * y * one_minus_cos - z * sin_theta
        rot[:, 0, 2] = x * z * one_minus_cos + y * sin_theta
        rot[:, 1, 0] = y * x * one_minus_cos + z * sin_theta
        rot[:, 1, 1] = cos_theta + y * y * one_minus_cos
        rot[:, 1, 2] = y * z * one_minus_cos - x * sin_theta
        rot[:, 2, 0] = z * x * one_minus_cos - y * sin_theta
        rot[:, 2, 1] = z * y * one_minus_cos + x * sin_theta
        rot[:, 2, 2] = cos_theta + z * z * one_minus_cos
        return rot

    def _apply_perception_noise(
        self,
        points, # (num_points, 3)
        colors,  # (num_points, C)
        instance_ids,  # (num_points,)
        object_transforms,  # (num_objects, 4, 4)
        num_objects
    ):
        if self.split != 'train':
            return points, colors, instance_ids, object_transforms

        apply_transform_noise = (
            self.perception_translation_noise_std > 0
            or self.perception_rotation_noise_std > 0
            or self.perception_scale_noise_std > 0
        )
        apply_instance_noise = (
            self.perception_instance_id_corruption_prob > 0
            and num_objects > 1
            and instance_ids.numel() > 0
        )
        if not apply_transform_noise and not apply_instance_noise:
            return points, colors, instance_ids, object_transforms

        noisy_object_transforms = object_transforms
        if apply_transform_noise:
            noisy_object_transforms = object_transforms.clone()

        if apply_transform_noise and num_objects > 0:
            all_transforms = torch.linalg.inv(noisy_object_transforms[:num_objects])

            linear = all_transforms[:, :3, :3]
            scales = torch.linalg.norm(linear, dim=1).mean(dim=1)
            safe_scales = torch.clamp(scales, min=1e-6)
            rotations = linear / safe_scales[:, None, None]
            translations = all_transforms[:, :3, 3]

            if self.perception_translation_noise_std > 0:
                translation_noise_std = safe_scales[:, None] * self.perception_translation_noise_std
                translations = translations + torch.randn_like(translations) * translation_noise_std
                translations = translations.clamp(min=POS_MIN, max=POS_MAX)

            if self.perception_rotation_noise_std > 0:
                rot_angles_deg = torch.randn(
                    (all_transforms.shape[0],),
                    dtype=all_transforms.dtype,
                    device=all_transforms.device,
                ) * self.perception_rotation_noise_std
                rot_angles_rad = torch.deg2rad(rot_angles_deg)
                rot_axes = torch.randn(
                    (all_transforms.shape[0], 3),
                    dtype=all_transforms.dtype,
                    device=all_transforms.device,
                )
                rotations = self._axis_angle_to_matrix_torch(rot_axes, rot_angles_rad) @ rotations

            if self.perception_scale_noise_std > 0:
                scale_noise = torch.randn_like(safe_scales) * (
                    safe_scales * self.perception_scale_noise_std
                )
                safe_scales = torch.clamp(
                    safe_scales + scale_noise,
                    min=max(SCALE_MIN, 1e-4),
                    max=SCALE_MAX,
                )

            all_transforms_noisy = all_transforms.clone()
            all_transforms_noisy[:, :3, :3] = rotations * safe_scales[:, None, None]
            all_transforms_noisy[:, :3, 3] = translations

            all_transforms_noisy_np = all_transforms_noisy.detach().cpu().numpy()
            linear_noisy = all_transforms_noisy_np[:, :3, :3]
            scales_noisy = np.linalg.norm(linear_noisy, axis=1).mean(axis=1)
            scales_noisy = np.clip(scales_noisy, max(SCALE_MIN, 1e-4), SCALE_MAX)
            rotations_noisy = linear_noisy / np.maximum(scales_noisy[:, None, None], 1e-8)
            angles_noisy = _rotation_matrix_to_euler_sxyz_batch(rotations_noisy)
            trans_noisy = all_transforms_noisy_np[:, :3, 3]

            d_scales, d_angles, d_trans = discrete_transform_batch(
                scales_noisy,
                angles_noisy,
                trans_noisy,
            )
            cont_scales, cont_angles, cont_trans = continue_transform_batch(
                d_scales,
                d_angles,
                d_trans,
            )
            all_transforms_consistent = get_transform_matrix_batch(
                cont_scales,
                cont_angles,
                cont_trans,
            )
            linear_consistent = all_transforms_consistent[:, :3, :3]
            trans_consistent = all_transforms_consistent[:, :3, 3]
            linear_consistent_inv = np.linalg.inv(linear_consistent)
            inverse_consistent = np.zeros_like(all_transforms_consistent)
            inverse_consistent[:, :3, :3] = linear_consistent_inv
            inverse_consistent[:, :3, 3] = -np.einsum(
                "nij,nj->ni",
                linear_consistent_inv,
                trans_consistent,
            )
            inverse_consistent[:, 3, 3] = 1.0
            noisy_object_transforms[:num_objects] = torch.from_numpy(
                inverse_consistent.astype(np.float32, copy=False)
            ).to(device=noisy_object_transforms.device, dtype=noisy_object_transforms.dtype)

        if apply_instance_noise:
            linear = noisy_object_transforms[:num_objects, :3, :3]
            translations = noisy_object_transforms[:num_objects, :3, 3]
            centers_all = -torch.linalg.solve(linear, translations.unsqueeze(-1)).squeeze(-1)
            has_background_slot = bool((instance_ids == 0).any().item())
            if has_background_slot:
                object_centers = centers_all[1:]
                foreground_mask = instance_ids > 0
                object_col_offset = 1
            else:
                object_centers = centers_all
                foreground_mask = instance_ids >= 0
                object_col_offset = 0

            if object_centers.shape[0] > 1 and foreground_mask.any():
                corrupt_mask = foreground_mask & (
                    torch.rand(instance_ids.shape, device=instance_ids.device)
                    < self.perception_instance_id_corruption_prob
                )

                if corrupt_mask.any():
                    selected_points = points[corrupt_mask]
                    selected_instance_ids = instance_ids[corrupt_mask]

                    point_to_center = selected_points[:, None, :] - object_centers[None, :, :]
                    dist_sq = (point_to_center * point_to_center).sum(dim=-1)
                    radius_sq = self.perception_instance_id_corruption_near_radius ** 2

                    current_object_cols = selected_instance_ids - object_col_offset
                    valid_current = (current_object_cols >= 0) & (current_object_cols < object_centers.shape[0])
                    row_ids = torch.arange(
                        selected_points.shape[0],
                        device=instance_ids.device,
                    )
                    dist_sq[row_ids[valid_current], current_object_cols[valid_current]] = float("inf")

                    nearest_dist_sq, nearest_object_cols = dist_sq.min(dim=1)

                    bg_switch = torch.rand(
                        selected_points.shape[0],
                        device=instance_ids.device,
                    ) < self.perception_instance_id_corruption_background_prob
                    has_near_neighbor = nearest_dist_sq <= radius_sq
                    nearest_ids = nearest_object_cols + object_col_offset

                    replacement_ids = torch.where(
                        bg_switch,
                        torch.zeros_like(selected_instance_ids),
                        torch.where(
                            has_near_neighbor,
                            nearest_ids.to(selected_instance_ids.dtype),
                            torch.zeros_like(selected_instance_ids),
                        ),
                    )

                    instance_ids = instance_ids.clone()
                    instance_ids[corrupt_mask] = replacement_ids

        return points, colors, instance_ids, noisy_object_transforms



    def _make_object_tokens_from_latents(
        self,
        object_shape_latents,
        scales,
        angles,
        translations,
        augment_info,
        instance_ids,
    ):
        combined_transform = np.asarray(augment_info["norm_transform"]) @ np.asarray(augment_info["augment_transform"])
        scales = np.asarray(scales, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        translations = np.asarray(translations, dtype=np.float64)

        new_scales, new_angles, new_trans = transform_6d_from_transform_batch(
            scales, angles, translations, combined_transform
        )
        d_scales, d_angles, d_trans = discrete_transform_batch(new_scales, new_angles, new_trans)
        cont_scales, cont_angles, cont_trans = continue_transform_batch(d_scales, d_angles, d_trans)
        object_to_world = get_transform_matrix_batch(cont_scales, cont_angles, cont_trans)
        object_transforms = np.linalg.inv(object_to_world).astype(np.float32)

        feats_list = []
        coords_list = []
        for latent in object_shape_latents:
            feats = latent["feats"]
            if feats.ndim != 2 or feats.shape[1] != 32:
                raise ValueError(f"Expected raw TRELLIS2 shape latent with 32 channels, got {feats.shape}")
            feats_list.append(feats.astype(np.float32, copy=False))
            coords_list.append(latent["coords"].astype(np.int32, copy=False))

        return {
            "feats": feats_list,
            "coords": coords_list,
            "transforms": object_transforms,
            "instance_ids": instance_ids.astype(np.int32, copy=False),
        }

    def _load_scene_sample(self, data_dict):
        shape_latents_path = data_dict['shape_latents_path']
        transforms_path = data_dict['transforms_path']
        data_name = data_dict['data_name']

        with open(transforms_path, 'rb') as f:
            objects_transforms = pickle.load(f)
        object_centers, object_scales, num_valid_objects = self._prepare_perception_recall_metadata(
            objects_transforms
        )
        shape_latents_dir = shape_latents_path
        scene_shape_latent_rotation = self._select_scene_shape_rotation(data_dict)

        intrinsics, c2ws, (height, width) = read_cameras(data_dict['camera_path'])
        rgbs = read_rgbs(data_dict['frames_dir'], height, width, parallel=True, max_workers=8)
        depths = read_depths(data_dict['depth_dir'], height, width, parallel=True, max_workers=8)
        masks, _ = read_masks_v2(data_dict['masks_dir'], height, width, parallel=True, max_workers=8)
        rgbs, depths, masks, intrinsics, height, width = downsample_all(
            rgbs, depths, masks, intrinsics, height, width, factor=2
        )

        points, depths, ray_dirs_world, instance_ids = project_depth_to_world_patch_geometry_with_instance_mask(
            depths, intrinsics, c2ws, masks, downsample=self.dino_patch_downsample
        )
        cam_origins_world = c2ws[:, None, :, 3].astype(np.float32)

        frame_indices = self._sample_frame_indices(points.shape[0])
        rgbs = rgbs[frame_indices]
        points = points[frame_indices]
        depths = depths[frame_indices]
        ray_dirs_world = ray_dirs_world[frame_indices]
        cam_origins_world = cam_origins_world[frame_indices]
        instance_ids = instance_ids[frame_indices]

        depths = self._apply_depth_noise(depths)
        ray_dirs_world, cam_origins_world = self._apply_camera_noise(ray_dirs_world, cam_origins_world)

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
        instance_ids, existing_indices = self._apply_perception_recall_noise(
            instance_ids,
            existing_indices,
            object_centers,
            object_scales,
            num_valid_objects,
            data_name,
        )

        points = points.reshape(-1, 3)
        instance_ids = instance_ids.reshape(-1)
        points, augment_info = self.augment_points(points, augment=self.augment)

        objects_tokens = read_objects_to_tokens_with_instance_ids_raw_shape(
            shape_latents_dir,
            objects_transforms,
            augment_info,
            existing_indices,
            instance_ids,
            latent_rotation=scene_shape_latent_rotation,
        )

        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(rgbs).float()
        instance_ids = torch.from_numpy(objects_tokens["instance_ids"]).long()
        feats_list_np = objects_tokens["feats"]
        coords_list_np = objects_tokens["coords"]
        object_transforms = torch.from_numpy(objects_tokens["transforms"])
        (
            object_feats,
            object_coords,
            object_transforms,
            instance_ids,
            num_objects,
            selected_indices,
        ) = self._select_object_token_subset(
            feats_list_np, coords_list_np, object_transforms, instance_ids
        )

        points, rgbs, instance_ids, object_transforms = self._apply_perception_noise(
            points, rgbs, instance_ids, object_transforms, num_objects
        )

        return {
            "split": self.split,
            "sample_type": "scene",
            "data_name": data_name,
            "points": points,
            "rgbs": rgbs,
            "valid_point_masks": None,
            "instance_ids": instance_ids,
            "object_feats": object_feats,
            "object_coords": object_coords,
            "object_transforms": object_transforms,
            "num_objects": num_objects,
            "selected_indices": selected_indices,
            "scene_shape_latent_rotation": scene_shape_latent_rotation,
        }

    def _load_object_sample(self, data_dict):
        if len(self.object_data_list) == 0:
            raise RuntimeError("object_data was sampled but no object render items are available")

        manifest_num_objects = data_dict.get("num_objects")
        num_objects = min(
            int(manifest_num_objects) if manifest_num_objects is not None else self.num_object_data_per_sample,
            len(self.object_data_list),
        )
        if (
            self.split == 'train'
            and self.dino_upsample > 1
            and self.train_num_max_objects is not None
        ):
            num_objects = min(num_objects, int(self.train_num_max_objects))
        if "object_start" in data_dict and self.split_object_names:
            object_start = int(data_dict["object_start"])
            selected_object_names = [
                self.split_object_names[(object_start + offset) % len(self.split_object_names)]
                for offset in range(num_objects)
            ]
            selected_object_refs = [self.object_data_by_name[name] for name in selected_object_names]
            frame_seeds = np.asarray(
                [
                    stable_uint32(data_dict["data_name"], name, out_idx, data_dict.get("seed", 0))
                    for out_idx, name in enumerate(selected_object_names)
                ],
                dtype=np.uint32,
            )
        else:
            replace = len(self.object_data_list) < num_objects
            selected_indices = np.random.choice(len(self.object_data_list), size=num_objects, replace=replace)
            selected_object_refs = selected_indices
            frame_seeds = np.random.randint(
                0,
                np.iinfo(np.uint32).max,
                size=num_objects,
                dtype=np.uint32,
            )
        read_args = [
            (out_idx, object_ref, frame_seeds[out_idx])
            for out_idx, object_ref in enumerate(selected_object_refs)
        ]
        max_workers = min(8, num_objects)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                object_samples = list(executor.map(self._read_selected_object_render, read_args))
        else:
            object_samples = [self._read_selected_object_render(args) for args in read_args]

        projected_samples = []
        groups_by_shape = {}
        for sample_idx, sample in enumerate(object_samples):
            groups_by_shape.setdefault((sample["height"], sample["width"]), []).append(sample_idx)

        for (height, width), group_indices in groups_by_shape.items():
            group_rgbs = np.stack([object_samples[i]["rgb"] for i in group_indices], axis=0)
            group_depths = np.stack([object_samples[i]["depth"] for i in group_indices], axis=0)
            group_masks = np.stack([object_samples[i]["mask"] for i in group_indices], axis=0)
            group_intrinsics = np.stack([object_samples[i]["intrinsics"] for i in group_indices], axis=0)
            group_c2ws = np.stack([object_samples[i]["c2ws"] for i in group_indices], axis=0)

            group_rgbs, group_depths, group_masks, group_intrinsics, _, _ = downsample_all(
                group_rgbs, group_depths, group_masks, group_intrinsics, height, width, factor=1
            )
            group_points, group_depths_ds, group_ray_dirs_world, group_instance_ids = (
                project_depth_to_world_patch_geometry_with_instance_mask(
                    group_depths, group_intrinsics, group_c2ws, group_masks, downsample=self.dino_patch_downsample
                )
            )
            group_cam_origins_world = group_c2ws[:, None, :, 3].astype(np.float32)

            group_depths_ds = self._apply_depth_noise(group_depths_ds)
            group_ray_dirs_world, group_cam_origins_world = self._apply_camera_noise(
                group_ray_dirs_world, group_cam_origins_world
            )

            if (
                self.split == 'train'
                and (
                    self.depth_noise_std > 0
                    or self.camera_rotation_noise_std > 0
                    or self.camera_translation_noise_std > 0
                )
            ):
                group_points = self._reconstruct_points(
                    group_depths_ds,
                    group_ray_dirs_world,
                    group_cam_origins_world,
                )

            for group_pos, sample_idx in enumerate(group_indices):
                sample = object_samples[sample_idx]
                projected_samples.append({
                    "out_idx": sample["out_idx"],
                    "object_item": sample["object_item"],
                    "rgb": group_rgbs[group_pos],
                    "points": group_points[group_pos],
                    "valid_mask": group_instance_ids[group_pos].reshape(-1) > 0,
                })

        projected_samples.sort(key=lambda sample: sample["out_idx"])
        valid_samples = [sample for sample in projected_samples if np.any(sample["valid_mask"])]
        if len(valid_samples) == 0:
            return self._load_object_sample(data_dict)

        scales = []
        angles = []
        latent_angles = []
        translations = []
        object_to_scene_mats = []
        for sample in valid_samples:
            scale, angle, trans = self._sample_object_transform(sample["out_idx"])
            rotation_radians = self._shape_latent_rotation_radians(sample["object_item"])
            latent_angle = np.array(angle, dtype=np.float32, copy=True)
            latent_angle[2] -= rotation_radians
            scales.append(scale)
            angles.append(angle)
            latent_angles.append(latent_angle)
            translations.append(trans)
            object_to_scene_mats.append(self._make_object_to_scene_matrix(scale, angle, trans))

        object_to_scene_mats = np.stack(object_to_scene_mats, axis=0)
        point_counts = np.asarray(
            [sample["valid_mask"].sum() for sample in valid_samples],
            dtype=np.int64,
        )
        if len({sample["points"].shape[0] for sample in valid_samples}) == 1:
            points_batch = np.stack([sample["points"] for sample in valid_samples], axis=0)
            points_batch = (
                points_batch @ np.swapaxes(object_to_scene_mats[:, :3, :3], 1, 2)
                + object_to_scene_mats[:, None, :3, 3]
            )
            valid_masks = np.stack([sample["valid_mask"] for sample in valid_samples], axis=0)
            points = points_batch[valid_masks].astype(np.float32, copy=False)
            valid_point_masks = valid_masks.reshape(-1).astype(np.bool_, copy=False)
        else:
            points = np.concatenate([
                (
                    sample["points"] @ object_to_scene[:3, :3].T
                    + object_to_scene[:3, 3]
                )[sample["valid_mask"]]
                for sample, object_to_scene in zip(valid_samples, object_to_scene_mats)
            ], axis=0).astype(np.float32, copy=False)
            valid_point_masks = np.concatenate(
                [sample["valid_mask"] for sample in valid_samples],
                axis=0,
            ).astype(np.bool_, copy=False)
        instance_ids = np.repeat(
            np.arange(len(valid_samples), dtype=np.int32),
            point_counts,
        )
        rgbs = np.stack([sample["rgb"] for sample in valid_samples], axis=0).astype(np.float32, copy=False)

        latent_args = [sample["object_item"] for sample in valid_samples]
        max_workers = min(8, len(latent_args))
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                object_shape_latents = list(executor.map(self._load_object_shape_latent, latent_args))
        else:
            object_shape_latents = [self._load_object_shape_latent(item) for item in latent_args]

        points, augment_info = self.augment_points(points, augment=self.augment)
        objects_tokens = self._make_object_tokens_from_latents(
            object_shape_latents,
            scales,
            np.stack(latent_angles, axis=0),
            np.stack(translations, axis=0),
            augment_info,
            instance_ids,
        )

        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(rgbs).float()
        valid_point_masks = torch.from_numpy(valid_point_masks).bool()
        instance_ids = torch.from_numpy(objects_tokens["instance_ids"]).long()
        feats_list_np = objects_tokens["feats"]
        coords_list_np = objects_tokens["coords"]
        object_transforms = torch.from_numpy(objects_tokens["transforms"])
        (
            object_feats,
            object_coords,
            object_transforms,
            instance_ids,
            num_objects,
            selected_indices,
        ) = self._select_object_token_subset(
            feats_list_np, coords_list_np, object_transforms, instance_ids
        )

        points, rgbs, instance_ids, object_transforms = self._apply_perception_noise(
            points, rgbs, instance_ids, object_transforms, num_objects
        )

        return {
            "split": self.split,
            "sample_type": "object",
            "data_name": data_dict["data_name"],
            "points": points,
            "rgbs": rgbs,
            "valid_point_masks": valid_point_masks,
            "instance_ids": instance_ids,
            "object_feats": object_feats,
            "object_coords": object_coords,
            "object_transforms": object_transforms,
            "num_objects": num_objects,
            "selected_indices": selected_indices,
            "object_shape_latent_rotations": [
                sample["object_item"].get("shape_latent_rotation", self.object_shape_trellis2_rotation)
                for sample in valid_samples
            ],
            "object_shape_latent_paths": [
                sample["object_item"].get("shape_latent_path")
                for sample in valid_samples
            ],
        }

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        if data_dict.get("sample_type") == "object":
            return self._load_object_sample(data_dict)
        return self._load_scene_sample(data_dict)

def sparse_collate_fn(batch):
    """
    Custom collate to handle variable number of points/objects.
    """
    batch_size = len(batch)
    if batch_size == 1:
        item = batch[0]
        num_objects = item['num_objects']

        points = item['points'].unsqueeze(0)
        rgbs = item['rgbs'].unsqueeze(0)
        valid_point_masks = (
            item['valid_point_masks'].unsqueeze(0)
            if item['valid_point_masks'] is not None
            else None
        )
        instance_ids = item['instance_ids'].unsqueeze(0)
        data_name = item['data_name']
        sample_type = item.get('sample_type')
        object_feats = item['object_feats']
        object_transforms = item['object_transforms'].unsqueeze(0)
        object_coords = item['object_coords']
        selected_indices = item['selected_indices'].unsqueeze(0)
        max_num_objects = num_objects
    else:
        assert False, "Not implemented"

    return {
        "points": points,
        "rgbs": rgbs,
        "valid_point_masks": valid_point_masks,
        "instance_ids": instance_ids,
        "data_name": data_name,
        "sample_type": sample_type,
        "num_objects": num_objects,
        "object_feats": object_feats,
        "object_coords": object_coords,
        "object_transforms": object_transforms,
        "selected_indices": selected_indices,
        "max_num_objects": max_num_objects,
    }

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
    dataset = VoxelizationDataset(
        data_dir_info=config['data_dir_info'],
        scene_scale=POS_MAX,
        split=split,
        train_split_ratio=config.get('train_split_ratio', 0.9),
        train_val_split=config.get('train_val_split', True),
        num_max_scenes=config.get('num_max_scenes', None),
        augment=config.get('augment', True) if split == 'train' else False,
        depth_noise_std=config.get('depth_noise_std', 0.0),
        camera_translation_noise_std=config.get('camera_translation_noise_std', 0.0),
        camera_rotation_noise_std=config.get('camera_rotation_noise_std', 0.0),
        perception_translation_noise_std=config.get('perception_translation_noise_std', 0.0),
        perception_rotation_noise_std=config.get('perception_rotation_noise_std', 0.0),
        perception_scale_noise_std=config.get('perception_scale_noise_std', 0.0),
        perception_instance_id_recall=config.get('perception_instance_id_recall', 1.0),
        perception_instance_id_recall_background_prob=config.get('perception_instance_id_recall_background_prob', 0.25),
        perception_instance_id_corruption_prob=config.get('perception_instance_id_corruption_prob', 0.0),
        perception_instance_id_corruption_background_prob=config.get('perception_instance_id_corruption_background_prob', 0.25),
        perception_instance_id_corruption_near_radius=config.get('perception_instance_id_corruption_near_radius', 2.0),
        perception_instance_id_recall_scale_ratio_thresh=config.get('perception_instance_id_recall_scale_ratio_thresh', 0.5),
        frame_subsample=config.get('frame_subsample', False),
        sample_full_frame_prob=config.get('sample_full_frame_prob', 0.1),
        scene_data_weight=config.get('scene_data_weight', 1),
        object_data_weight=config.get('object_data_weight', 3),
        num_object_data_per_sample=config.get('num_object_data_per_sample', 64),
        object_data_sources=config.get('object_data_sources', None),
        object_data_grid_cols=config.get('object_data_grid_cols', 8),
        object_data_spacing=config.get('object_data_spacing', 1.2),
        object_data_scale_min=config.get('object_data_scale_min', 0.8),
        object_data_scale_max=config.get('object_data_scale_max', 1.2),
        object_shape_trellis2_rotation=config.get('object_shape_trellis2_rotation', '000'),
        scene_shape_trellis2_rotation=config.get('scene_shape_trellis2_rotation', None),
        train_num_max_objects=config.get('train_num_max_objects', 64) if split == 'train' else None,
        split_manifest_dir=config.get('split_manifest_dir', DEFAULT_SPLIT_DIR),
        split_manifest_prefix=config.get('split_manifest_prefix', DEFAULT_SPLIT_PREFIX),
        use_split_manifests=config.get('use_split_manifests', True),
        strict_split_manifests=config.get('strict_split_manifests', True),
        dino_upsample=config.get('dino_upsample', 1),
    )

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
