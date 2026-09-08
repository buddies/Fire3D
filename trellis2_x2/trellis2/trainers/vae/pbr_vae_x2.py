from typing import *
import os
import copy
import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...modules import sparse as sp
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from ...utils.loss_utils import l1_loss, l2_loss


class PbrVaeX2Trainer(BasicTrainer):
    """
    Trainer for Shape VAE X2

    A VAE trainer for sparse structured latent compression.
    The encoder takes a sparse tensor, downsamples 2x twice (32 -> 64 -> 128),
    and outputs 16-dim latent (mean and variance).
    The decoder reverses this process.

    Args:
        models (dict[str, nn.Module]): Models to train. Should contain 'encoder' and 'decoder'.
        dataset (torch.utils.data.Dataset): Dataset that provides x_0 as SparseTensor.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        loss_type (str): Type of reconstruction loss. Options: 'l1', 'l2'.
        lambda_kl (float): KL divergence loss weight.
        lambda_subdiv (float): Subdivision/occupancy prediction loss weight.
    """

    def __init__(
        self,
        *args,
        loss_type: str = 'l2',
        lambda_kl: float = 1e-6,
        lambda_subdiv: float = 0.1,
        num_workers: Optional[int] = None,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        **kwargs
    ):
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.lambda_subdiv = lambda_subdiv
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        super().__init__(*args, **kwargs)

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        num_workers = self.num_workers
        if num_workers is None:
            num_workers = int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        dataloader_kwargs = {}
        if num_workers > 0:
            dataloader_kwargs.update(
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
            )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
            **dataloader_kwargs,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x_0: sp.SparseTensor
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            x_0 (SparseTensor): Input sparse tensor to encode and reconstruct.

        Returns:
            Tuple of (loss_dict, status_dict) where loss_dict contains 'loss' and other terms.
        """
        z, mean, logvar = self.training_models['encoder'](x_0, sample_posterior=True, return_raw=True)
        decoder = self.training_models['decoder']

        # Check if decoder predicts subdivision (occupancy)
        pred_subdiv = getattr(decoder, 'pred_subdiv', False)

        if pred_subdiv:
            # Decoder returns (output, subs_gt, subs) during training
            y, subs_gt, subs = decoder(z)
        else:
            y = decoder(z)

        terms = edict(loss = 0.0)

        # direct regression
        if self.loss_type == 'l1':
            terms["l1"] = l1_loss(x_0.feats, y.feats)
            terms["loss"] = terms["loss"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(x_0.feats, y.feats)
            terms["loss"] = terms["loss"] + terms["l2"]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

        # Subdivision/occupancy prediction loss
        # Penalizes differences in voxel existence between input and reconstruction
        # Uses binary cross entropy to enforce that the decoder predicts
        # the correct voxel occupancy at each upsampling stage
        if pred_subdiv and self.lambda_subdiv > 0:
            for i, (sub_gt, sub) in enumerate(zip(subs_gt, subs)):
                terms[f"bce_sub{i}"] = F.binary_cross_entropy_with_logits(
                    sub.feats, sub_gt.float()
                )
                terms["loss"] = terms["loss"] + self.lambda_subdiv * terms[f"bce_sub{i}"]

        # KL regularization
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        gts = []
        recons = []
        self.models['encoder'].eval()
        self.models['decoder'].eval()
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device)
            x_0 = args['x_0']
            z = self.models['encoder'](x_0)
            y = self.models['decoder'](z)
            gts.append(x_0)
            recons.append(y)
        self.models['encoder'].train()
        self.models['decoder'].train()

        # Concatenate sparse tensors for visualization
        gt_concat = sp.SparseTensor(
            coords=torch.cat([g.coords for g in gts], dim=0),
            feats=torch.cat([g.feats for g in gts], dim=0),
        )
        recon_concat = sp.SparseTensor(
            coords=torch.cat([r.coords for r in recons], dim=0),
            feats=torch.cat([r.feats for r in recons], dim=0),
        )

        sample_dict = {
            'gt': {'value': gt_concat, 'type': 'sample'},
            'recon': {'value': recon_concat, 'type': 'sample'},
        }

        return sample_dict
