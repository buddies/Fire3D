from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_sparse_structure_dataset_builds_dense_occupancy(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "trellis2_x2"))
    from trellis2.datasets.ss import SparseStructure

    object_id = "a" * 64
    coords = np.asarray([[0, 1, 2], [7, 6, 5], [0, 1, 2]], dtype=np.uint8)
    np.savez_compressed(tmp_path / f"{object_id}.npz", coords=coords)

    dataset = SparseStructure(str(tmp_path), resolution=8, rotations=[0])
    occupancy = dataset[0]["ss"]
    assert occupancy.shape == (1, 8, 8, 8)
    assert occupancy.dtype == torch.float32
    assert occupancy.sum().item() == 2
    assert occupancy[0, 0, 1, 2].item() == 1
    assert occupancy[0, 7, 6, 5].item() == 1


def test_ssvae_joint_loss_updates_both_models(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "trellis2_x2"))
    from trellis2.models import SparseStructureDecoder, SparseStructureEncoder
    from trellis2.trainers.vae.ss_x2 import SSX2Trainer

    trainer = object.__new__(SSX2Trainer)
    trainer.lambda_kl = 1e-6
    trainer.training_models = {
        "encoder": SparseStructureEncoder(
            in_channels=1,
            latent_channels=2,
            num_res_blocks=1,
            num_res_blocks_middle=1,
            channels=[4, 8, 16],
            use_fp16=False,
        ),
        "decoder": SparseStructureDecoder(
            out_channels=1,
            latent_channels=2,
            num_res_blocks=1,
            num_res_blocks_middle=1,
            channels=[16, 8, 4],
            use_fp16=False,
        ),
    }
    target = (torch.rand(2, 1, 8, 8, 8) > 0.8).float()
    terms, status = trainer.training_losses(target)
    terms["loss"].backward()

    assert terms["loss"].isfinite()
    assert {"bce", "kl"} <= set(terms)
    assert {"iou_05", "dice_05", "exact_support_05"} <= set(status)
    for model in trainer.training_models.values():
        assert any(parameter.grad is not None for parameter in model.parameters())
