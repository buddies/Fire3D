"""Scene/object dataset that reads precomputed Shape VAE X2 LC512 latents."""

import json
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.datasets.scene_objects_dataset_shape_x2_online_with_objects import (
    DEFAULT_SPLIT_DIR,
    DEFAULT_SPLIT_PREFIX,
    VoxelizationDataset as OnlineShapeX2Dataset,
    sparse_collate_fn,
)
from training.datasets.split_manifest_utils import reorder_object_items_from_split, stable_uint32
from utils.project import (
    downsample_all,
    project_depth_to_world_patch_geometry_with_instance_mask,
)
from utils.read_frames import read_cameras, read_depths, read_masks_v2, read_rgbs
from utils.read_objects import read_objects_to_tokens_with_instance_ids_shape_x2
from utils.discrete import continue_transform_batch, discrete_transform_batch
from utils.transforms import (
    get_transform_matrix_batch,
    transform_6d_from_transform_batch,
)
from utils.constants import POS_MAX


class VoxelizationDataset(OnlineShapeX2Dataset):
    """Online Shape X2 dataset variant backed by saved raw X2 latent npz files."""

    def __init__(self, *args, shape_x2_channels=512, **kwargs):
        self.shape_x2_channels = int(shape_x2_channels)
        if self.shape_x2_channels <= 0:
            raise ValueError(
                f"shape_x2_channels must be positive, got {self.shape_x2_channels}"
            )
        super().__init__(*args, **kwargs)

    def _build_object_data_list(self, object_data_info):
        if object_data_info is None:
            return []

        data_dir = object_data_info.get("data_dir")
        if data_dir is None:
            raise ValueError("data_dir_info.object_data must contain data_dir")

        render_root = object_data_info.get("render_dir", os.path.join(data_dir, "render"))
        shape_x2_root = object_data_info.get(
            "shape_x2_latents_dir",
            os.path.join(data_dir, "shape_hcvae_latents"),
        )
        shape_x2_key = object_data_info.get("shape_x2_latent_key", "trellis2_shape_x2_encoding")
        sources = object_data_info.get("sources", self.object_data_sources)
        if sources is None:
            sources = sorted(os.listdir(render_root))
        elif isinstance(sources, str):
            sources = [sources]

        object_items = []
        total_usable_object_items = 0
        for source in sources:
            source_render_dir = os.path.join(render_root, source)
            source_latent_dir = os.path.join(shape_x2_root, source, "latents", shape_x2_key)
            if not os.path.isdir(source_render_dir) or not os.path.isdir(source_latent_dir):
                print(f"[object_data] Skipping source {source}: missing render or shape_x2 latents dir")
                continue

            source_items = []
            for object_id in sorted(os.listdir(source_render_dir)):
                object_render_dir = os.path.join(source_render_dir, object_id)
                shape_latent_paths = {
                    rotation: os.path.join(source_latent_dir, f"{object_id}__rot{rotation}.npz")
                    for rotation in self.object_shape_trellis2_rotations
                }
                existing_shape_latent_paths = {
                    rotation: path
                    for rotation, path in shape_latent_paths.items()
                    if os.path.exists(path)
                }
                shape_latent_rotations = [
                    rotation
                    for rotation in self.object_shape_trellis2_rotations
                    if rotation in existing_shape_latent_paths
                ]
                if len(shape_latent_rotations) == 0:
                    continue

                first_rotation = shape_latent_rotations[0]
                item = {
                    "sample_type": "object_item",
                    "dataset_name": source,
                    "object_id": object_id,
                    "data_name": f"{source}_{object_id}",
                    "camera_path": os.path.join(object_render_dir, "cameras.json"),
                    "frames_dir": os.path.join(object_render_dir, "frames"),
                    "depth_dir": os.path.join(object_render_dir, "depth"),
                    "masks_dir": os.path.join(object_render_dir, "mask"),
                    "shape_latent_path": existing_shape_latent_paths[first_rotation],
                    "shape_latent_paths": existing_shape_latent_paths,
                    "shape_latent_rotations": shape_latent_rotations,
                    "shape_latent_rotation": first_rotation,
                }
                if (
                    os.path.exists(item["camera_path"])
                    and os.path.isdir(item["frames_dir"])
                    and os.path.isdir(item["depth_dir"])
                    and os.path.isdir(item["masks_dir"])
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
                    f"[object_data:{source}] Number of usable X2 objects before manifest filtering: "
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

            if self.split == "train":
                object_indices = train_object_indices
            elif self.split == "trainval":
                object_indices = train_object_indices[: int(0.1 * len(train_object_indices) + 1)]
            elif self.split == "val":
                object_indices = val_object_indices
            else:
                raise ValueError(f"Invalid split: {self.split}")

            source_split_items = [source_items[object_idx] for object_idx in object_indices]
            object_items.extend(source_split_items)
            print(
                f"[object_data:{source}] Number of X2 objects in {self.split} set: "
                f"{len(source_split_items)} / {num_source_items}"
            )

        print(
            f"[object_data] Found {total_usable_object_items} usable X2 object render items; "
            f"using {len(object_items)} for {self.split}"
        )
        return object_items

    def _load_object_shape_latent(self, object_item):
        with np.load(object_item["shape_latent_path"]) as latent_archive:
            feats = latent_archive["feats"].astype(np.float32, copy=False)
            coords = latent_archive["coords"].astype(np.int32, copy=False)
        if feats.ndim != 2 or feats.shape[1] != self.shape_x2_channels:
            raise ValueError(
                f"Expected Shape VAE X2 latent with {self.shape_x2_channels} channels, "
                f"got {feats.shape}"
            )
        return {"feats": feats, "coords": coords}

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
            if feats.ndim != 2 or feats.shape[1] != self.shape_x2_channels:
                raise ValueError(
                    f"Expected Shape VAE X2 latent with {self.shape_x2_channels} channels, "
                    f"got {feats.shape}"
                )
            feats_list.append(feats.astype(np.float32, copy=False))
            coords_list.append(latent["coords"].astype(np.int32, copy=False))

        return {
            "feats": feats_list,
            "coords": coords_list,
            "transforms": object_transforms,
            "instance_ids": instance_ids.astype(np.int32, copy=False),
        }

    def _load_scene_sample(self, data_dict):
        shape_latents_dir = data_dict["shape_latents_path"]
        transforms_path = data_dict["transforms_path"]
        data_name = data_dict["data_name"]
        scene_shape_latent_rotation = self._select_scene_shape_rotation(data_dict)

        with open(transforms_path, "rb") as f:
            objects_transforms = pickle.load(f)
        object_centers, object_scales, num_valid_objects = self._prepare_perception_recall_metadata(
            objects_transforms
        )

        intrinsics, c2ws, (height, width) = read_cameras(data_dict["camera_path"])
        rgbs = read_rgbs(data_dict["frames_dir"], height, width, parallel=True, max_workers=8)
        depths = read_depths(data_dict["depth_dir"], height, width, parallel=True, max_workers=8)
        masks, _ = read_masks_v2(data_dict["masks_dir"], height, width, parallel=True, max_workers=8)
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
            self.split == "train"
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

        objects_tokens = read_objects_to_tokens_with_instance_ids_shape_x2(
            shape_latents_dir,
            objects_transforms,
            augment_info,
            existing_indices,
            instance_ids,
            latent_rotation=scene_shape_latent_rotation,
            latent_channels=self.shape_x2_channels,
        )
        object_local_names = list(objects_tokens["local_names"])

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
            "object_local_names": object_local_names,
        }


def _seed_worker(worker_id):
    """Seed NumPy/Python from the deterministic PyTorch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config, split):
    dataset = VoxelizationDataset(
        data_dir_info=config["data_dir_info"],
        scene_scale=POS_MAX,
        split=split,
        train_split_ratio=config.get("train_split_ratio", 0.9),
        train_val_split=config.get("train_val_split", True),
        num_max_scenes=config.get("num_max_scenes", None),
        augment=config.get("augment", True) if split == "train" else False,
        depth_noise_std=config.get("depth_noise_std", 0.0),
        camera_translation_noise_std=config.get("camera_translation_noise_std", 0.0),
        camera_rotation_noise_std=config.get("camera_rotation_noise_std", 0.0),
        perception_translation_noise_std=config.get("perception_translation_noise_std", 0.0),
        perception_rotation_noise_std=config.get("perception_rotation_noise_std", 0.0),
        perception_scale_noise_std=config.get("perception_scale_noise_std", 0.0),
        perception_instance_id_recall=config.get("perception_instance_id_recall", 1.0),
        perception_instance_id_recall_background_prob=config.get("perception_instance_id_recall_background_prob", 0.25),
        perception_instance_id_corruption_prob=config.get("perception_instance_id_corruption_prob", 0.0),
        perception_instance_id_corruption_background_prob=config.get("perception_instance_id_corruption_background_prob", 0.25),
        perception_instance_id_corruption_near_radius=config.get("perception_instance_id_corruption_near_radius", 2.0),
        perception_instance_id_recall_scale_ratio_thresh=config.get("perception_instance_id_recall_scale_ratio_thresh", 0.5),
        frame_subsample=config.get("frame_subsample", False),
        sample_full_frame_prob=config.get("sample_full_frame_prob", 0.1),
        scene_data_weight=config.get("scene_data_weight", 1),
        object_data_weight=config.get("object_data_weight", 3),
        num_object_data_per_sample=config.get("num_object_data_per_sample", 64),
        object_data_sources=config.get("object_data_sources", None),
        object_data_grid_cols=config.get("object_data_grid_cols", 8),
        object_data_spacing=config.get("object_data_spacing", 1.2),
        object_data_scale_min=config.get("object_data_scale_min", 0.8),
        object_data_scale_max=config.get("object_data_scale_max", 1.2),
        object_shape_trellis2_rotation=config.get("object_shape_trellis2_rotation", "000"),
        scene_shape_trellis2_rotation=config.get("scene_shape_trellis2_rotation", None),
        train_num_max_objects=config.get("train_num_max_objects", 64) if split == "train" else None,
        split_manifest_dir=config.get("split_manifest_dir", DEFAULT_SPLIT_DIR),
        split_manifest_prefix=config.get("split_manifest_prefix", DEFAULT_SPLIT_PREFIX),
        use_split_manifests=config.get("use_split_manifests", True),
        strict_split_manifests=config.get("strict_split_manifests", True),
        dino_upsample=config.get("dino_upsample", 1),
        shape_x2_channels=config.get("shape_x2_channels", 512),
    )

    loader_seed = config.get("loader_seed")
    generator = None
    worker_init_fn = None
    if loader_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(stable_uint32(int(loader_seed), split))
        worker_init_fn = _seed_worker

    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1) if split == "train" else 1,
        shuffle=(split == "train"),
        num_workers=config.get("num_workers", 4),
        collate_fn=sparse_collate_fn,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
