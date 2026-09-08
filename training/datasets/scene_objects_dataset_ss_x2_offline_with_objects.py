"""Scene/object conditioning dataset with dense LC64 SS-VAE X2 targets."""

from __future__ import annotations

import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.datasets.scene_objects_dataset_shape_x2_online_with_objects import (
    DEFAULT_SPLIT_DIR,
    DEFAULT_SPLIT_PREFIX,
    VoxelizationDataset as OnlineShapeX2Dataset,
)
from training.datasets.split_manifest_utils import stable_uint32
from utils.constants import POS_MAX
from utils.discrete import continue_transform_batch, discrete_transform_batch
from utils.project import (
    downsample_all,
    project_depth_to_world_patch_geometry_with_instance_mask,
)
from utils.read_frames import read_cameras, read_depths, read_masks_v2, read_rgbs
from utils.read_objects import _apply_latent_rotation_to_pose, _split_bg_and_object_ids
from utils.ss_x2_latent_contract import (
    ROTATIONS,
    load_and_validate_provenance,
    load_ss_x2_latent,
    normalize_rotation_label,
    resolve_latent_path,
)
from utils.transforms import get_transform_matrix_batch, transform_6d_from_transform_batch


def _scene_ss_tokens(
    latent_root,
    objects_transforms,
    augment_info,
    existing_indices,
    instance_ids,
    *,
    latent_rotation,
    expected_input_resolution,
    expected_latent_resolution,
    expected_latent_channels,
):
    """Load dense targets in the same spatial/object order as the shape-flow path."""
    combined_transform = np.asarray(augment_info["norm_transform"]) @ np.asarray(
        augment_info["augment_transform"]
    )
    existing = set(existing_indices)
    bg_id, obj_ids = _split_bg_and_object_ids(objects_transforms)
    obj_ids_existing = []
    original_indices_filtered = [0]
    for index, object_id in enumerate(obj_ids, start=1):
        if index in existing:
            obj_ids_existing.append(object_id)
            original_indices_filtered.append(index)
    local_names = [bg_id, *obj_ids_existing]

    count = len(local_names)
    scales = np.empty(count, dtype=np.float64)
    angles = np.empty((count, 3), dtype=np.float64)
    translations = np.empty((count, 3), dtype=np.float64)
    for index, name in enumerate(local_names):
        transform = objects_transforms[name]
        scales[index] = transform["scale"]
        angles[index] = transform["angles"]
        translations[index] = transform["trans"]

    scales, angles, translations = _apply_latent_rotation_to_pose(
        scales, angles, translations, latent_rotation
    )
    scales, angles, translations = transform_6d_from_transform_batch(
        scales, angles, translations, combined_transform
    )
    d_scales, d_angles, d_trans = discrete_transform_batch(scales, angles, translations)
    if count > 1:
        object_translations = d_trans[1:]
        order = np.lexsort(
            (
                object_translations[:, 0],
                object_translations[:, 1],
                object_translations[:, 2],
            )
        )
        final_order = np.concatenate(([0], 1 + order))
    else:
        final_order = np.asarray([0], dtype=np.int64)

    d_scales = d_scales[final_order]
    d_angles = d_angles[final_order]
    d_trans = d_trans[final_order]
    scales, angles, translations = continue_transform_batch(d_scales, d_angles, d_trans)
    object_to_world = get_transform_matrix_batch(scales, angles, translations)
    object_transforms = np.linalg.inv(object_to_world).astype(np.float32)
    local_names = [local_names[int(index)] for index in final_order]
    ordered_original_indices = [original_indices_filtered[int(index)] for index in final_order]

    max_original_id = max(
        int(instance_ids.max()) if instance_ids.size else 0,
        max(ordered_original_indices, default=0),
    )
    id_map = np.zeros(max_original_id + 1, dtype=np.int32)
    for new_index, old_index in enumerate(ordered_original_indices):
        id_map[old_index] = new_index
    mapped_instance_ids = np.zeros_like(instance_ids, dtype=np.int32)
    valid_ids = (instance_ids >= 0) & (instance_ids <= max_original_id)
    mapped_instance_ids[valid_ids] = id_map[instance_ids[valid_ids]]

    latents_by_name = {}
    paths_by_name = {}
    for name in dict.fromkeys(local_names):
        path = resolve_latent_path(latent_root, name, latent_rotation)
        latent, _ = load_ss_x2_latent(
            path,
            expected_input_resolution=expected_input_resolution,
            expected_latent_resolution=expected_latent_resolution,
            expected_latent_channels=expected_latent_channels,
        )
        latents_by_name[name] = latent
        paths_by_name[name] = str(path)

    return {
        "feats": [latents_by_name[name] for name in local_names],
        "coords": [np.empty((0, 3), dtype=np.int32) for _ in local_names],
        "paths": [paths_by_name[name] for name in local_names],
        "transforms": object_transforms,
        "instance_ids": mapped_instance_ids,
    }


class VoxelizationDataset(OnlineShapeX2Dataset):
    """Shape-flow conditioning with raw dense `[K,8,2,2,2]` SS targets."""

    def __init__(
        self,
        *args,
        ss_x2_channels=8,
        ss_x2_resolution=2,
        ss_input_resolution=8,
        ss_latent_key="ss_vae_x2_encoding",
        require_ss_provenance=True,
        expected_ss_artifact_id=None,
        expected_ss_step=None,
        expected_ss_encoder_sha256=None,
        **kwargs,
    ):
        self.ss_x2_channels = int(ss_x2_channels)
        self.ss_x2_resolution = int(ss_x2_resolution)
        self.ss_input_resolution = int(ss_input_resolution)
        self.ss_latent_key = str(ss_latent_key)
        self.require_ss_provenance = bool(require_ss_provenance)
        self.expected_ss_artifact_id = expected_ss_artifact_id
        self.expected_ss_step = None if expected_ss_step is None else int(expected_ss_step)
        self.expected_ss_encoder_sha256 = expected_ss_encoder_sha256
        if self.ss_x2_channels != 8 or self.ss_x2_resolution != 2 or self.ss_input_resolution != 8:
            raise ValueError(
                "LC64 SS-flow requires input resolution 8 and dense latent [8,2,2,2]; "
                f"got channels={self.ss_x2_channels}, latent_resolution={self.ss_x2_resolution}, "
                f"input_resolution={self.ss_input_resolution}"
            )

        data_dir_info = kwargs.get("data_dir_info", args[0] if args else None)
        if self.require_ss_provenance:
            self._validate_configured_roots(data_dir_info)
        super().__init__(*args, **kwargs)

    def _validate_configured_roots(self, data_dir_info):
        if not isinstance(data_dir_info, dict):
            raise ValueError("data_dir_info is required for SS provenance validation")
        scene_info = data_dir_info.get("scene_data", data_dir_info)
        object_info = data_dir_info.get("object_data") if "scene_data" in data_dir_info else None
        roots = []
        for dataset_name, dataset_config in scene_info.items():
            root = dataset_config.get("shape_trellis2_latents_dir")
            if root is None:
                raise ValueError(f"Scene dataset {dataset_name} is missing shape_trellis2_latents_dir")
            roots.append(root)
        if object_info is not None:
            root = object_info.get("ss_x2_latents_dir")
            if root is None:
                raise ValueError("object_data is missing ss_x2_latents_dir")
            roots.append(root)
        for root in dict.fromkeys(roots):
            load_and_validate_provenance(
                root,
                expected_ss_artifact_id=self.expected_ss_artifact_id,
                expected_ss_step=self.expected_ss_step,
                expected_ss_encoder_sha256=self.expected_ss_encoder_sha256,
            )

    def _build_object_data_list(self, object_data_info):
        if object_data_info is None:
            return []
        data_dir = object_data_info.get("data_dir")
        if data_dir is None:
            raise ValueError("data_dir_info.object_data must contain data_dir")
        render_root = object_data_info.get("render_dir", os.path.join(data_dir, "render"))
        latent_root = object_data_info.get("ss_x2_latents_dir")
        if latent_root is None:
            raise ValueError("data_dir_info.object_data must contain ss_x2_latents_dir")
        latent_key = object_data_info.get("ss_x2_latent_key", self.ss_latent_key)
        sources = object_data_info.get("sources", self.object_data_sources)
        if sources is None:
            sources = sorted(os.listdir(render_root))
        elif isinstance(sources, str):
            sources = [sources]

        object_items = []
        for source in sources:
            source_render_dir = os.path.join(render_root, source)
            source_latent_dir = os.path.join(latent_root, source, "latents", latent_key)
            if not os.path.isdir(source_render_dir) or not os.path.isdir(source_latent_dir):
                print(f"[object_data] Skipping {source}: missing render or SS X2 latent directory")
                continue
            source_items = []
            for object_id in sorted(os.listdir(source_render_dir)):
                object_render_dir = os.path.join(source_render_dir, object_id)
                paths = {}
                for rotation in self.object_shape_trellis2_rotations:
                    try:
                        paths[rotation] = str(resolve_latent_path(source_latent_dir, object_id, rotation))
                    except FileNotFoundError:
                        paths = {}
                        break
                if len(paths) != len(self.object_shape_trellis2_rotations):
                    continue
                first_rotation = self.object_shape_trellis2_rotations[0]
                item = {
                    "sample_type": "object_item",
                    "dataset_name": source,
                    "object_id": object_id,
                    "data_name": f"{source}_{object_id}",
                    "camera_path": os.path.join(object_render_dir, "cameras.json"),
                    "frames_dir": os.path.join(object_render_dir, "frames"),
                    "depth_dir": os.path.join(object_render_dir, "depth"),
                    "masks_dir": os.path.join(object_render_dir, "mask"),
                    "shape_latent_path": paths[first_rotation],
                    "shape_latent_paths": paths,
                    "shape_latent_rotations": list(self.object_shape_trellis2_rotations),
                    "shape_latent_rotation": first_rotation,
                }
                if (
                    os.path.isfile(item["camera_path"])
                    and os.path.isdir(item["frames_dir"])
                    and os.path.isdir(item["depth_dir"])
                    and os.path.isdir(item["masks_dir"])
                ):
                    source_items.append(item)
            if getattr(self, "use_split_manifests", False):
                object_items.extend(source_items)
                print(f"[object_data:{source}] Usable strict four-rotation SS objects: {len(source_items)}")
                continue
            num_items = len(source_items)
            if num_items == 0:
                continue
            train_indices = np.linspace(
                0, num_items - 1, max(int(num_items * self.object_train_split_ratio), 1)
            ).astype(np.int32)
            val_indices = np.setdiff1d(np.arange(num_items), train_indices)
            if not self.train_val_split:
                train_indices = np.arange(num_items)
            if self.split == "train":
                indices = train_indices
            elif self.split == "trainval":
                indices = train_indices[: int(0.1 * len(train_indices) + 1)]
            elif self.split == "val":
                indices = val_indices
            else:
                raise ValueError(f"Invalid split: {self.split}")
            object_items.extend(source_items[int(index)] for index in indices)
        print(f"[object_data] Using {len(object_items)} strict SS X2 objects for {self.split}")
        return object_items

    def _load_object_shape_latent(self, object_item):
        latent, _ = load_ss_x2_latent(
            object_item["shape_latent_path"],
            expected_input_resolution=self.ss_input_resolution,
            expected_latent_resolution=self.ss_x2_resolution,
            expected_latent_channels=self.ss_x2_channels,
        )
        return {"ss_latent": latent}

    def _make_object_tokens_from_latents(
        self,
        object_shape_latents,
        scales,
        angles,
        translations,
        augment_info,
        instance_ids,
    ):
        combined_transform = np.asarray(augment_info["norm_transform"]) @ np.asarray(
            augment_info["augment_transform"]
        )
        scales, angles, translations = transform_6d_from_transform_batch(
            np.asarray(scales, dtype=np.float64),
            np.asarray(angles, dtype=np.float64),
            np.asarray(translations, dtype=np.float64),
            combined_transform,
        )
        scales, angles, translations = discrete_transform_batch(scales, angles, translations)
        scales, angles, translations = continue_transform_batch(scales, angles, translations)
        object_to_world = get_transform_matrix_batch(scales, angles, translations)
        return {
            "feats": [latent["ss_latent"] for latent in object_shape_latents],
            "coords": [np.empty((0, 3), dtype=np.int32) for _ in object_shape_latents],
            "transforms": np.linalg.inv(object_to_world).astype(np.float32),
            "instance_ids": instance_ids.astype(np.int32, copy=False),
        }

    def _select_object_token_subset(
        self,
        feats_list_np,
        coords_list_np,
        object_transforms,
        instance_ids,
        return_source_indices=False,
    ):
        del coords_list_np
        num_objects = len(feats_list_np)
        if num_objects == 0:
            raise ValueError("SS target list is empty")
        if (
            self.split == "train"
            and self.train_num_max_objects is not None
            and num_objects > self.train_num_max_objects
        ):
            if num_objects > 1:
                selected = torch.randperm(num_objects - 1)[: self.train_num_max_objects - 1] + 1
                selected = torch.cat((torch.zeros(1, dtype=torch.long), selected.long()))
            else:
                selected = torch.arange(num_objects)
        else:
            selected = torch.arange(num_objects)
        selected = selected.long()
        object_feats = torch.stack(
            [torch.from_numpy(feats_list_np[int(index)]).float() for index in selected], dim=0
        )
        expected = (len(selected), self.ss_x2_channels) + (self.ss_x2_resolution,) * 3
        if tuple(object_feats.shape) != expected:
            raise ValueError(f"Expected dense SS target {expected}, got {tuple(object_feats.shape)}")

        remapped_instance_ids = torch.full_like(instance_ids, -1)
        if instance_ids.numel() and selected.numel():
            max_id = max(int(instance_ids.max().item()), int(selected.max().item()))
            if max_id >= 0:
                id_map = torch.full(
                    (max_id + 1,), -1, dtype=instance_ids.dtype, device=instance_ids.device
                )
                selected_device = selected.to(instance_ids.device)
                id_map[selected_device] = torch.arange(
                    selected.numel(), dtype=instance_ids.dtype, device=instance_ids.device
                )
                valid = (instance_ids >= 0) & (instance_ids <= max_id)
                remapped_instance_ids[valid] = id_map[instance_ids[valid]]

        count = int(selected.numel())
        result = (
            object_feats,
            torch.empty((0, 4), dtype=torch.int32),
            object_transforms[selected],
            remapped_instance_ids,
            count,
            torch.arange(count, dtype=torch.long),
        )
        if return_source_indices:
            return result + (selected,)
        return result

    def _load_scene_sample(self, data_dict):
        transforms_path = data_dict["transforms_path"]
        data_name = data_dict["data_name"]
        rotation = self._select_scene_shape_rotation(data_dict)
        if rotation is None:
            raise ValueError("SS four-way scene training requires an explicit selected rotation")
        rotation = normalize_rotation_label(rotation)
        with open(transforms_path, "rb") as handle:
            objects_transforms = pickle.load(handle)
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
        points, depths, ray_dirs_world, instance_ids = (
            project_depth_to_world_patch_geometry_with_instance_mask(
                depths, intrinsics, c2ws, masks, downsample=self.dino_patch_downsample
            )
        )
        cam_origins_world = c2ws[:, None, :, 3].astype(np.float32)
        frame_indices = self._sample_frame_indices(points.shape[0])
        rgbs, points, depths = rgbs[frame_indices], points[frame_indices], depths[frame_indices]
        ray_dirs_world = ray_dirs_world[frame_indices]
        cam_origins_world = cam_origins_world[frame_indices]
        instance_ids = instance_ids[frame_indices]
        depths = self._apply_depth_noise(depths)
        ray_dirs_world, cam_origins_world = self._apply_camera_noise(
            ray_dirs_world, cam_origins_world
        )
        if self.split == "train" and (
            self.depth_noise_std > 0
            or self.camera_rotation_noise_std > 0
            or self.camera_translation_noise_std > 0
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
        tokens = _scene_ss_tokens(
            data_dict["shape_latents_path"],
            objects_transforms,
            augment_info,
            existing_indices,
            instance_ids,
            latent_rotation=rotation,
            expected_input_resolution=self.ss_input_resolution,
            expected_latent_resolution=self.ss_x2_resolution,
            expected_latent_channels=self.ss_x2_channels,
        )
        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(rgbs).float()
        instance_ids = torch.from_numpy(tokens["instance_ids"]).long()
        object_transforms = torch.from_numpy(tokens["transforms"])
        (
            object_feats,
            object_coords,
            object_transforms,
            instance_ids,
            num_objects,
            selected_indices,
            source_selected_indices,
        ) = self._select_object_token_subset(
            tokens["feats"],
            tokens["coords"],
            object_transforms,
            instance_ids,
            return_source_indices=True,
        )
        points, rgbs, instance_ids, object_transforms = self._apply_perception_noise(
            points, rgbs, instance_ids, object_transforms, num_objects
        )
        # Keep the read-only evaluation/debug paths aligned with the same source
        # objects selected above.  Training remaps instance IDs to [0, K), so
        # ``selected_indices`` must remain arange(K) for condition selection;
        # ``source_selected_indices`` preserves the indices into the untruncated
        # source-object list for metadata such as these paths.
        target_paths = [tokens["paths"][int(index)] for index in source_selected_indices]
        if len(target_paths) != num_objects:
            raise ValueError(
                f"SS target path/object count mismatch: paths={len(target_paths)} "
                f"objects={num_objects}"
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
            "scene_ss_latent_rotation": rotation,
            "scene_ss_latent_paths": target_paths,
            # Read-only inference/debug metadata. The training collate function
            # intentionally ignores these keys, so training tensors are unchanged.
            "scene_augment_transform": torch.from_numpy(
                np.asarray(augment_info["augment_transform"], dtype=np.float64)
            ),
            "scene_norm_transform": torch.from_numpy(
                np.asarray(augment_info["norm_transform"], dtype=np.float64)
            ),
            "scene_world_to_training": torch.from_numpy(
                np.asarray(augment_info["norm_transform"], dtype=np.float64)
                @ np.asarray(augment_info["augment_transform"], dtype=np.float64)
            ),
        }


def ss_x2_collate_fn(batch):
    if len(batch) != 1:
        raise ValueError(f"SS X2 flow requires dataloader batch size 1, got {len(batch)}")
    item = batch[0]
    object_feats = item["object_feats"]
    if object_feats.ndim != 5 or tuple(object_feats.shape[1:]) != (8, 2, 2, 2):
        raise ValueError(f"Expected object_feats [K,8,2,2,2], got {tuple(object_feats.shape)}")
    num_objects = int(item["num_objects"])
    if object_feats.shape[0] != num_objects:
        raise ValueError(
            f"SS target/object count mismatch: targets={object_feats.shape[0]} objects={num_objects}"
        )
    return {
        "points": item["points"].unsqueeze(0),
        "rgbs": item["rgbs"].unsqueeze(0),
        "valid_point_masks": (
            item["valid_point_masks"].unsqueeze(0)
            if item["valid_point_masks"] is not None
            else None
        ),
        "instance_ids": item["instance_ids"].unsqueeze(0),
        "data_name": item["data_name"],
        "sample_type": item.get("sample_type"),
        "num_objects": num_objects,
        "object_feats": object_feats,
        "object_coords": item["object_coords"],
        "object_transforms": item["object_transforms"].unsqueeze(0),
        "selected_indices": item["selected_indices"].unsqueeze(0),
        "max_num_objects": num_objects,
    }


def _seed_worker(worker_id):
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
        num_max_scenes=config.get("num_max_scenes"),
        augment=config.get("augment", True) if split == "train" else False,
        depth_noise_std=config.get("depth_noise_std", 0.0),
        camera_translation_noise_std=config.get("camera_translation_noise_std", 0.0),
        camera_rotation_noise_std=config.get("camera_rotation_noise_std", 0.0),
        perception_translation_noise_std=config.get("perception_translation_noise_std", 0.0),
        perception_rotation_noise_std=config.get("perception_rotation_noise_std", 0.0),
        perception_scale_noise_std=config.get("perception_scale_noise_std", 0.0),
        perception_instance_id_recall=config.get("perception_instance_id_recall", 1.0),
        perception_instance_id_recall_background_prob=config.get(
            "perception_instance_id_recall_background_prob", 0.25
        ),
        perception_instance_id_corruption_prob=config.get(
            "perception_instance_id_corruption_prob", 0.0
        ),
        perception_instance_id_corruption_background_prob=config.get(
            "perception_instance_id_corruption_background_prob", 0.25
        ),
        perception_instance_id_corruption_near_radius=config.get(
            "perception_instance_id_corruption_near_radius", 2.0
        ),
        perception_instance_id_recall_scale_ratio_thresh=config.get(
            "perception_instance_id_recall_scale_ratio_thresh", 0.5
        ),
        frame_subsample=config.get("frame_subsample", False),
        sample_full_frame_prob=config.get("sample_full_frame_prob", 0.1),
        scene_data_weight=config.get("scene_data_weight", 1),
        object_data_weight=config.get("object_data_weight", 3),
        num_object_data_per_sample=config.get("num_object_data_per_sample", 64),
        object_data_sources=config.get("object_data_sources"),
        object_data_grid_cols=config.get("object_data_grid_cols", 8),
        object_data_spacing=config.get("object_data_spacing", 1.2),
        object_data_scale_min=config.get("object_data_scale_min", 0.8),
        object_data_scale_max=config.get("object_data_scale_max", 1.2),
        object_shape_trellis2_rotation=config.get("object_shape_trellis2_rotation", ROTATIONS),
        scene_shape_trellis2_rotation=config.get("scene_shape_trellis2_rotation", ROTATIONS),
        train_num_max_objects=config.get("train_num_max_objects", 48) if split == "train" else None,
        split_manifest_dir=config.get("split_manifest_dir", DEFAULT_SPLIT_DIR),
        split_manifest_prefix=config.get("split_manifest_prefix", DEFAULT_SPLIT_PREFIX),
        use_split_manifests=config.get("use_split_manifests", True),
        strict_split_manifests=config.get("strict_split_manifests", True),
        dino_upsample=config.get("dino_upsample", 1),
        ss_x2_channels=config.get("ss_x2_channels", 8),
        ss_x2_resolution=config.get("ss_x2_resolution", 2),
        ss_input_resolution=config.get("ss_input_resolution", 8),
        ss_latent_key=config.get("ss_x2_latent_key", "ss_vae_x2_encoding"),
        require_ss_provenance=config.get("require_ss_provenance", True),
        expected_ss_artifact_id=config.get("expected_ss_artifact_id"),
        expected_ss_step=config.get("expected_ss_step"),
        expected_ss_encoder_sha256=config.get("expected_ss_encoder_sha256"),
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
        shuffle=split == "train",
        num_workers=config.get("num_workers", 4),
        collate_fn=ss_x2_collate_fn,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
