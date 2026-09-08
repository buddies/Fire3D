import os
import json
from typing import *
import numpy as np
import torch
from .. import models
from .components import StandardDatasetBase, ImageConditionedMixin
from ..modules.sparse import SparseTensor, sparse_cat
from ..representations import MeshWithVoxel
from ..utils.data_utils import load_balanced_group_indices
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics

class SLatPbrOnly(StandardDatasetBase):
    """
    structured latent for sparse voxel pbr dataset

    Args:
        roots (str): path to the dataset
        latent_key (str): key of the latent to be used
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        resolution (int): resolution of decoded sparse voxel
        attrs (list): attributes to be decoded
        pretained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        full_pbr: bool = False,
        latent_key: str = 'tex_enc_next_dc_f16c32_fp16_512',
        pbr_slat_normalization: Optional[dict] = None,
        attrs: list[str] = ['base_color', 'metallic', 'roughness', 'emissive', 'alpha'],
        **kwargs
    ):
        self.resolution = resolution
        self.pbr_slat_normalization = pbr_slat_normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.full_pbr = full_pbr
        self.latent_key = latent_key
        self.value_range = (-1, 1)

        super().__init__(
            roots,
            **kwargs
        )

        self.loads = [32 for _, sha256 in self.instances]

        if self.pbr_slat_normalization is not None:
            self.pbr_slat_mean = torch.tensor(self.pbr_slat_normalization['mean']).reshape(1, -1)
            self.pbr_slat_std = torch.tensor(self.pbr_slat_normalization['std']).reshape(1, -1)

        self.attrs = attrs
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]

    def filter_metadata(self, metadata):
        stats = {}
        encoded_col = f'latent_{self.latent_key}'
        if encoded_col in metadata.columns:
            metadata = metadata[metadata[encoded_col] == True]
        elif 'pbr_latent_encoded' in metadata.columns:
            metadata = metadata[metadata['pbr_latent_encoded'] == True]
        else:
            raise KeyError(f'Metadata missing PBR latent flag column: expected {encoded_col} or pbr_latent_encoded')
        stats['With PBR latent'] = len(metadata)

        tokens_col = f'{self.latent_key}_tokens'
        if tokens_col in metadata.columns:
            metadata = metadata[metadata[tokens_col] <= self.max_tokens]
            stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        elif 'pbr_latent_tokens' in metadata.columns:
            metadata = metadata[metadata['pbr_latent_tokens'] <= self.max_tokens]
            stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)

        if self.full_pbr:
            for col in ('num_basecolor_tex', 'num_metallic_tex', 'num_roughness_tex'):
                if col not in metadata.columns:
                    raise KeyError(f'full_pbr=True requires metadata column {col}')
            metadata = metadata[metadata['num_basecolor_tex'] > 0]
            metadata = metadata[metadata['num_metallic_tex'] > 0]
            metadata = metadata[metadata['num_roughness_tex'] > 0]
            stats['Full PBR'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        # PBR latent
        latent_path = os.path.join(root, 'latents', self.latent_key, f'{instance}.npz')
        if not os.path.exists(latent_path):
            latent_path = os.path.join(root, 'pbr_latents', self.latent_key, f'{instance}.npz')
        data = np.load(latent_path)
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.pbr_slat_normalization is not None:
            feats = (feats - self.pbr_slat_mean) / self.pbr_slat_std
        pbr_z = SparseTensor(feats, coords)

        return {
            'x_0': pbr_z,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['x_0'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], SparseTensor):
                    pack[k] = sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs
