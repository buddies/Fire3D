"""Object-only RGB + Shape-X2 -> PBR-X2 sparse-flow dataset.

This loader reuses the modern Shape-X2 object projection/layout implementation,
but pairs one raw LC64 Shape-X2 latent with one raw LC64 PBR-X2 latent on an
identical sparse support.  Scene records are intentionally excluded from the
training loader.
"""

from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.datasets.scene_objects_dataset_shape_x2_offline_with_objects import (
    VoxelizationDataset as OfflineShapeX2Dataset,
)
from training.datasets.scene_objects_dataset_shape_x2_online_with_objects import (
    DEFAULT_SPLIT_DIR,
    DEFAULT_SPLIT_PREFIX,
)
from training.datasets.split_manifest_utils import (
    load_object_split,
    reorder_object_items_from_split,
    stable_uint32,
)
from utils.constants import POS_MAX
from utils.object_render_augmentation import augment_object_render


class VoxelizationDataset(OfflineShapeX2Dataset):
    """Paired Shape/PBR LC64 dataset backed by rotation-specific NPZ files."""

    def __init__(
        self,
        *args,
        shape_x2_channels=64,
        pbr_x2_channels=64,
        object_render_augment=None,
        use_split_manifests=True,
        strict_split_manifests=True,
        split_manifest_dir=DEFAULT_SPLIT_DIR,
        split_manifest_prefix=DEFAULT_SPLIT_PREFIX,
        **kwargs,
    ):
        self.shape_condition_channels = int(shape_x2_channels)
        self.pbr_target_channels = int(pbr_x2_channels)
        if self.shape_condition_channels <= 0 or self.pbr_target_channels <= 0:
            raise ValueError("Shape/PBR X2 channel counts must be positive")
        self.object_render_augment_config = dict(object_render_augment or {})
        self._requested_split_manifests = bool(use_split_manifests)
        self._requested_strict_split_manifests = bool(strict_split_manifests)
        self._requested_split_manifest_dir = split_manifest_dir
        self._requested_split_manifest_prefix = split_manifest_prefix
        self._pbr_path_by_shape_path = {}
        self._inventory_is_split_manifest = False

        # Let the mature base class build projection/layout utilities and the
        # complete paired object inventory, but bypass its scene/object
        # interleaved materializer.  We install an object-only epoch below.
        super().__init__(
            *args,
            shape_x2_channels=self.shape_condition_channels + self.pbr_target_channels,
            use_split_manifests=False,
            strict_split_manifests=False,
            split_manifest_dir=split_manifest_dir,
            split_manifest_prefix=split_manifest_prefix,
            **kwargs,
        )
        self._install_object_only_split()

    def _build_object_data_list(self, object_data_info):
        if object_data_info is None:
            raise ValueError("PBR-X2 flow requires data_dir_info.object_data")
        data_dir = object_data_info.get("data_dir")
        if data_dir is None:
            raise ValueError("data_dir_info.object_data must contain data_dir")

        paired_manifest_path = object_data_info.get("paired_manifest_path")
        if paired_manifest_path:
            return self._build_object_data_list_from_manifest(paired_manifest_path)

        render_root = object_data_info.get("render_dir", os.path.join(data_dir, "render"))
        shape_root = object_data_info.get(
            "shape_x2_latents_dir", os.path.join(data_dir, "shape_hcvae_latents")
        )
        pbr_root = object_data_info.get(
            "pbr_x2_latents_dir", os.path.join(data_dir, "pbr_hcvae_latents")
        )
        shape_key = object_data_info.get("shape_x2_latent_key", "trellis2_shape_x2_encoding")
        pbr_key = object_data_info.get("pbr_x2_latent_key", "trellis2_pbr_x2_encoding")
        sources = object_data_info.get("sources", self.object_data_sources)
        if sources is None:
            sources = sorted(os.listdir(render_root))
        elif isinstance(sources, str):
            sources = [sources]

        object_items = []
        per_source = {}
        for source in sources:
            source_render = os.path.join(render_root, source)
            source_shape = os.path.join(shape_root, source, "latents", shape_key)
            source_pbr = os.path.join(pbr_root, source, "latents", pbr_key)
            if not all(os.path.isdir(path) for path in (source_render, source_shape, source_pbr)):
                print(
                    f"[pbr_x2_object_data] Skipping {source}: missing render, Shape-X2, or PBR-X2 root"
                )
                continue

            source_items = []
            for object_id in sorted(os.listdir(source_render)):
                object_render_dir = os.path.join(source_render, object_id)
                shape_paths = {
                    rotation: os.path.join(source_shape, f"{object_id}__rot{rotation}.npz")
                    for rotation in self.object_shape_trellis2_rotations
                }
                pbr_paths = {
                    rotation: os.path.join(source_pbr, f"{object_id}__rot{rotation}.npz")
                    for rotation in self.object_shape_trellis2_rotations
                }
                common_rotations = [
                    rotation
                    for rotation in self.object_shape_trellis2_rotations
                    if os.path.isfile(shape_paths[rotation]) and os.path.isfile(pbr_paths[rotation])
                ]
                if not common_rotations:
                    continue
                first_rotation = common_rotations[0]
                item = {
                    "sample_type": "object_item",
                    "dataset_name": source,
                    "object_id": object_id,
                    "data_name": f"{source}_{object_id}",
                    "camera_path": os.path.join(object_render_dir, "cameras.json"),
                    "frames_dir": os.path.join(object_render_dir, "frames"),
                    "depth_dir": os.path.join(object_render_dir, "depth"),
                    "masks_dir": os.path.join(object_render_dir, "mask"),
                    "shape_latent_path": shape_paths[first_rotation],
                    "shape_latent_paths": {rotation: shape_paths[rotation] for rotation in common_rotations},
                    "pbr_latent_path": pbr_paths[first_rotation],
                    "pbr_latent_paths": {rotation: pbr_paths[rotation] for rotation in common_rotations},
                    "shape_latent_rotations": common_rotations,
                    "shape_latent_rotation": first_rotation,
                }
                if (
                    os.path.isfile(item["camera_path"])
                    and os.path.isdir(item["frames_dir"])
                    and os.path.isdir(item["depth_dir"])
                    and os.path.isdir(item["masks_dir"])
                ):
                    source_items.append(item)
                    for rotation in common_rotations:
                        self._pbr_path_by_shape_path[shape_paths[rotation]] = pbr_paths[rotation]
            per_source[source] = len(source_items)
            object_items.extend(source_items)

        print(
            f"[pbr_x2_object_data] Found {len(object_items)} paired render assets; "
            f"per-source={per_source}"
        )
        return object_items

    def _build_object_data_list_from_manifest(self, manifest_path):
        manifest_path = os.fspath(manifest_path)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Missing paired PBR-X2 flow manifest: {manifest_path}")
        grouped = {}
        with open(manifest_path, "r") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                required = {
                    "data_name",
                    "source",
                    "asset_id",
                    "rotation",
                    "camera_path",
                    "frames_dir",
                    "depth_dir",
                    "mask_dir",
                    "shape_x2_path",
                    "pbr_x2_path",
                }
                missing = required.difference(record)
                if missing:
                    raise ValueError(
                        f"{manifest_path}:{line_number} missing fields {sorted(missing)}"
                    )
                item = grouped.setdefault(
                    record["data_name"],
                    {
                        "sample_type": "object_item",
                        "dataset_name": record["source"],
                        "object_id": record["asset_id"],
                        "data_name": record["data_name"],
                        "camera_path": record["camera_path"],
                        "frames_dir": record["frames_dir"],
                        "depth_dir": record["depth_dir"],
                        "masks_dir": record["mask_dir"],
                        "shape_latent_paths": {},
                        "pbr_latent_paths": {},
                    },
                )
                rotation = str(record["rotation"])
                item["shape_latent_paths"][rotation] = record["shape_x2_path"]
                item["pbr_latent_paths"][rotation] = record["pbr_x2_path"]
                self._pbr_path_by_shape_path[record["shape_x2_path"]] = record["pbr_x2_path"]

        items = []
        for data_name in sorted(grouped):
            item = grouped[data_name]
            rotations = [
                rotation
                for rotation in self.object_shape_trellis2_rotations
                if rotation in item["shape_latent_paths"] and rotation in item["pbr_latent_paths"]
            ]
            if not rotations:
                continue
            first = rotations[0]
            item["shape_latent_rotations"] = rotations
            item["shape_latent_rotation"] = first
            item["shape_latent_path"] = item["shape_latent_paths"][first]
            item["pbr_latent_path"] = item["pbr_latent_paths"][first]
            items.append(item)
        self._inventory_is_split_manifest = True
        print(
            f"[pbr_x2_object_data] Loaded {len(items)} paired assets directly from {manifest_path}"
        )
        return items

    def _install_object_only_split(self):
        if self._inventory_is_split_manifest:
            # The manifest builder already applied the frozen base-asset split.
            pass
        elif self._requested_split_manifests:
            split_names = load_object_split(
                self.split,
                split_dir=self._requested_split_manifest_dir,
                prefix=self._requested_split_manifest_prefix,
            )
            object_items, missing = reorder_object_items_from_split(
                self.object_data_list,
                split_names,
                strict=self._requested_strict_split_manifests,
            )
            if missing:
                print(f"[pbr_x2_split] Missing {len(missing)} paired assets from {self.split}")
            self.object_data_list = object_items
        else:
            # Debug-only fallback.  Production always uses frozen manifests.
            total = len(self.object_data_list)
            train_end = max(int(total * self.object_train_split_ratio), 1)
            if self.split == "train":
                self.object_data_list = self.object_data_list[:train_end]
            elif self.split == "trainval":
                self.object_data_list = self.object_data_list[: max(int(train_end * 0.1), 1)]
            elif self.split == "val":
                self.object_data_list = self.object_data_list[train_end:]
            else:
                raise ValueError(f"Invalid split: {self.split}")

        if not self.object_data_list:
            raise RuntimeError(f"No paired PBR-X2 object records for split={self.split}")
        self.use_split_manifests = self._requested_split_manifests
        self.strict_split_manifests = self._requested_strict_split_manifests
        self.object_data_by_name = {item["data_name"]: item for item in self.object_data_list}
        self.split_object_names = [item["data_name"] for item in self.object_data_list]
        self.scene_data_list = []
        self.data_list = []
        num_groups = int(math.ceil(len(self.object_data_list) / self.num_object_data_per_sample))
        for group_idx in range(num_groups):
            object_start = group_idx * self.num_object_data_per_sample
            self.data_list.append(
                {
                    "sample_type": "object",
                    "data_name": f"pbr_x2_object_{self.split}_{group_idx:08d}",
                    "object_start": object_start,
                    "num_objects": min(
                        self.num_object_data_per_sample,
                        len(self.object_data_list) - object_start,
                    ),
                    "seed": stable_uint32("pbr_x2", self.split, group_idx),
                }
            )
        print(
            f"[pbr_x2_split] split={self.split} assets={len(self.object_data_list)} "
            f"object-only batches={len(self.data_list)}"
        )

    def _object_item_with_selected_shape_rotation(self, object_item, frame_seed):
        selected = super()._object_item_with_selected_shape_rotation(object_item, frame_seed)
        selected = dict(selected)
        rotation = selected["shape_latent_rotation"]
        pbr_paths = selected.get("pbr_latent_paths", object_item.get("pbr_latent_paths", {}))
        if rotation not in pbr_paths:
            raise KeyError(f"Missing paired PBR-X2 rotation {rotation} for {selected['data_name']}")
        selected["pbr_latent_path"] = pbr_paths[rotation]
        return selected

    def _read_selected_object_render(self, args):
        out_idx, object_ref, frame_seed = args
        object_item = object_ref if isinstance(object_ref, dict) else self.object_data_list[int(object_ref)]
        object_item = self._object_item_with_selected_shape_rotation(object_item, frame_seed)
        with open(object_item["camera_path"], "r") as f:
            camera_data = json.load(f)

        frame_rng = np.random.default_rng(int(frame_seed))
        frame_idx = int(frame_rng.integers(0, len(camera_data["frames"])))
        intrinsics, c2ws, height, width = self._read_object_camera_frame_from_data(camera_data, frame_idx)
        rgb, depth, mask = self._read_object_render_frame_arrays(object_item, frame_idx)
        augment_rng = np.random.default_rng(stable_uint32(int(frame_seed), "pbr_render_augment"))
        render_aug = augment_object_render(
            rgb,
            depth,
            mask.astype(bool, copy=False),
            intrinsics[0],
            augment_rng,
            self.object_render_augment_config if self.split == "train" else {"enabled": False},
        )
        selected_item = dict(object_item)
        selected_item["render_augmentation"] = render_aug.metadata
        return {
            "out_idx": out_idx,
            "object_item": selected_item,
            "intrinsics": render_aug.intrinsics,
            "c2ws": c2ws[0],
            "height": int(render_aug.rgb.shape[0]),
            "width": int(render_aug.rgb.shape[1]),
            "rgb": render_aug.rgb,
            "depth": render_aug.depth,
            "mask": render_aug.mask.astype(np.int32),
        }

    def _load_object_shape_latent(self, object_item):
        with np.load(object_item["shape_latent_path"]) as archive:
            shape_feats = archive["feats"].astype(np.float32, copy=False)
            shape_coords = archive["coords"].astype(np.int32, copy=False)
        with np.load(object_item["pbr_latent_path"]) as archive:
            pbr_feats = archive["feats"].astype(np.float32, copy=False)
            pbr_coords = archive["coords"].astype(np.int32, copy=False)
        if shape_feats.ndim != 2 or shape_feats.shape[1] != self.shape_condition_channels:
            raise ValueError(f"Invalid Shape-X2 features {shape_feats.shape}: {object_item['shape_latent_path']}")
        if pbr_feats.ndim != 2 or pbr_feats.shape[1] != self.pbr_target_channels:
            raise ValueError(f"Invalid PBR-X2 features {pbr_feats.shape}: {object_item['pbr_latent_path']}")
        if shape_coords.shape != pbr_coords.shape or not np.array_equal(shape_coords, pbr_coords):
            raise ValueError(
                "Shape/PBR X2 sparse coordinates differ for paired record: "
                f"{object_item['shape_latent_path']} vs {object_item['pbr_latent_path']}"
            )
        if not np.isfinite(shape_feats).all() or not np.isfinite(pbr_feats).all():
            raise ValueError(f"Non-finite paired latent features for {object_item['data_name']}")
        return {
            "feats": np.concatenate([shape_feats, pbr_feats], axis=1),
            "coords": shape_coords,
        }

    def _load_object_sample(self, data_dict):
        sample = super()._load_object_sample(data_dict)
        paired = sample.pop("object_feats")
        expected = self.shape_condition_channels + self.pbr_target_channels
        if paired.ndim != 2 or paired.shape[1] != expected:
            raise ValueError(f"Expected paired features [N,{expected}], got {tuple(paired.shape)}")
        sample["object_shape_x2_feats"] = paired[:, : self.shape_condition_channels].contiguous()
        sample["object_pbr_x2_feats"] = paired[:, self.shape_condition_channels :].contiguous()
        # Keep the established name as a Shape alias for evaluation utilities
        # that only inspect geometry conditioning.
        sample["object_feats"] = sample["object_shape_x2_feats"]
        sample["object_pbr_x2_latent_paths"] = [
            self._pbr_path_by_shape_path[path] for path in sample.get("object_shape_latent_paths", [])
        ]
        return sample

    def _load_scene_sample(self, data_dict):
        raise RuntimeError("PBR-X2 production training is object-only")


def sparse_collate_fn(batch):
    if len(batch) != 1:
        raise NotImplementedError("PBR-X2 flow currently requires batch size 1 per GPU")
    item = batch[0]
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
        "num_objects": item["num_objects"],
        "object_feats": item["object_shape_x2_feats"],
        "object_shape_x2_feats": item["object_shape_x2_feats"],
        "object_pbr_x2_feats": item["object_pbr_x2_feats"],
        "object_coords": item["object_coords"],
        "object_transforms": item["object_transforms"].unsqueeze(0),
        "selected_indices": item["selected_indices"].unsqueeze(0),
        "max_num_objects": item["num_objects"],
        "object_shape_latent_rotations": item.get("object_shape_latent_rotations"),
        "object_shape_latent_paths": item.get("object_shape_latent_paths"),
        "object_pbr_x2_latent_paths": item.get("object_pbr_x2_latent_paths"),
    }


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config, split):
    data_dir_info = dict(config["data_dir_info"])
    data_dir_info["scene_data"] = {}
    object_data_info = dict(data_dir_info["object_data"])
    manifest_paths = config.get("paired_manifest_paths") or {}
    if split in manifest_paths:
        object_data_info["paired_manifest_path"] = manifest_paths[split]
    data_dir_info["object_data"] = object_data_info
    dataset = VoxelizationDataset(
        data_dir_info=data_dir_info,
        scene_scale=POS_MAX,
        split=split,
        train_split_ratio=config.get("train_split_ratio", 0.9),
        train_val_split=config.get("train_val_split", True),
        num_max_scenes=config.get("num_max_scenes"),
        augment=config.get("augment", False) if split == "train" else False,
        depth_noise_std=config.get("depth_noise_std", 0.0),
        camera_translation_noise_std=config.get("camera_translation_noise_std", 0.0),
        camera_rotation_noise_std=config.get("camera_rotation_noise_std", 0.0),
        perception_translation_noise_std=config.get("perception_translation_noise_std", 0.0),
        perception_rotation_noise_std=config.get("perception_rotation_noise_std", 0.0),
        perception_scale_noise_std=config.get("perception_scale_noise_std", 0.0),
        perception_instance_id_recall=config.get("perception_instance_id_recall", 1.0),
        perception_instance_id_recall_background_prob=config.get("perception_instance_id_recall_background_prob", 0.0),
        perception_instance_id_corruption_prob=config.get("perception_instance_id_corruption_prob", 0.0),
        perception_instance_id_corruption_background_prob=config.get("perception_instance_id_corruption_background_prob", 0.0),
        perception_instance_id_corruption_near_radius=config.get("perception_instance_id_corruption_near_radius", 1.0),
        perception_instance_id_recall_scale_ratio_thresh=config.get("perception_instance_id_recall_scale_ratio_thresh", 0.5),
        frame_subsample=False,
        sample_full_frame_prob=1.0,
        scene_data_weight=0,
        object_data_weight=1,
        num_object_data_per_sample=config.get("num_object_data_per_sample", 48),
        object_data_sources=config.get("object_data_sources"),
        object_data_grid_cols=config.get("object_data_grid_cols", 8),
        object_data_spacing=config.get("object_data_spacing", 1.2),
        object_data_scale_min=config.get("object_data_scale_min", 0.8),
        object_data_scale_max=config.get("object_data_scale_max", 1.2),
        object_shape_trellis2_rotation=config.get("object_shape_trellis2_rotation", ["000", "090", "180", "270"]),
        scene_shape_trellis2_rotation=None,
        train_num_max_objects=config.get("train_num_max_objects", 48) if split == "train" else None,
        split_manifest_dir=config.get("split_manifest_dir", DEFAULT_SPLIT_DIR),
        split_manifest_prefix=config.get("split_manifest_prefix", DEFAULT_SPLIT_PREFIX),
        use_split_manifests=config.get("use_split_manifests", True),
        strict_split_manifests=config.get("strict_split_manifests", True),
        dino_upsample=config.get("dino_upsample", 4),
        shape_x2_channels=config.get("shape_x2_channels", 64),
        pbr_x2_channels=config.get("pbr_x2_channels", 64),
        object_render_augment=config.get("object_render_augment", {}) if split == "train" else {"enabled": False},
    )
    loader_seed = config.get("loader_seed")
    generator = None
    worker_init_fn = None
    if loader_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(stable_uint32(int(loader_seed), split))
        worker_init_fn = _seed_worker
    workers = int(config.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1) if split == "train" else 1,
        shuffle=(split == "train"),
        num_workers=workers,
        collate_fn=sparse_collate_fn,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
        prefetch_factor=4 if workers > 0 else None,
        persistent_workers=workers > 0,
    )
