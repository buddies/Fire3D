import argparse
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "trellis2_x2"
for path in (REPO_ROOT, TRELLIS_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.append(path)

import numpy as np
import o_voxel
import torch
from tqdm import tqdm

import trellis2.models as models
import trellis2.modules.sparse as sp
from trellis2.modules.sparse.basic import sparse_cat, sparse_unbind
from data_processing.stages.latents.checkpoints import resolve_vae_checkpoints

torch.set_grad_enabled(False)


OBJECT_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_ROOT", REPO_ROOT / "data/training_objects")
)
POSTPROCESS_ROOT = Path(
    os.environ.get("FIRE3D_OBJECT_POSTPROCESS_ROOT", OBJECT_ROOT / "postprocess")
)
DATASET_NAMES = {
    "3d-future": "3D-FUTURE",
    "3D-FUTURE": "3D-FUTURE",
    "abo": "ABO",
    "ABO": "ABO",
    "hssd": "HSSD",
    "HSSD": "HSSD",
    "objaversexl_github": "ObjaverseXL_github",
    "ObjaverseXL_github": "ObjaverseXL_github",
    "objaversexl_sketchfab": "ObjaverseXL_sketchfab",
    "ObjaverseXL_sketchfab": "ObjaverseXL_sketchfab",
}


def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def load_pbr_vae_x2(pbr_vae_x2_ckpt, use_ema=True, step=None):
    config, encoder_ckpt_path, decoder_ckpt_path, label = resolve_vae_checkpoints(
        pbr_vae_x2_ckpt, use_ema=use_ema, step=step
    )
    print(f"Loading PBR HC-VAE {label}...")
    print(f"  Encoder: {encoder_ckpt_path}")
    print(f"  Decoder: {decoder_ckpt_path}")

    encoder_config = config["models"]["encoder"]
    encoder_x2 = getattr(models, encoder_config["name"])(**encoder_config.get("args", {}))
    encoder_x2.load_state_dict(torch.load(encoder_ckpt_path, map_location="cuda", weights_only=True))
    encoder_x2.cuda().eval()

    decoder_config = config["models"]["decoder"]
    decoder_x2 = getattr(models, decoder_config["name"])(**decoder_config.get("args", {}))
    decoder_x2.load_state_dict(torch.load(decoder_ckpt_path, map_location="cuda", weights_only=True))
    decoder_x2.cuda().eval()

    normalization_config = config["dataset"]["args"].get("pbr_slat_normalization")
    if normalization_config is None:
        raise ValueError(f"{pbr_vae_x2_ckpt} config missing dataset.args.pbr_slat_normalization")
    normalization = {
        "mean": torch.tensor(normalization_config["mean"], dtype=torch.float32).cuda(),
        "std": torch.tensor(normalization_config["std"], dtype=torch.float32).cuda(),
    }
    return encoder_x2, decoder_x2, normalization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_NAMES.keys()))
    parser.add_argument(
        "--enc_pretrained",
        type=str,
        default="microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16",
        help="Pretrained PBR encoder model",
    )
    parser.add_argument(
        "--pbr_vae_x2_ckpt",
        type=str,
        default=str(REPO_ROOT / "checkpoints/Fire3D/reconstruction/vae/pbr"),
        help="PBR VAE x2 checkpoint",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_loader_workers", type=int, default=8)
    parser.add_argument("--num_saver_workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=32,
                        help="Number of voxel load tasks to keep in-flight")
    return parser.parse_args()


PBR_ATTRS = ["base_color", "metallic", "roughness", "alpha"]


def load_voxel_item(voxel_path, voxel_name):
    try:
        coords, attr = o_voxel.io.read_vxz(str(voxel_path), num_threads=2)
        feats = torch.concat([attr[k] for k in PBR_ATTRS], dim=-1) / 255.0 * 2 - 1
        voxels = sp.SparseTensor(
            feats.float(),
            torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
        )
        if not is_valid_sparse_tensor(voxels):
            print(f"[Skip] {voxel_name}: NaN/Inf in input")
            return None
        return voxel_name, voxels
    except Exception as e:
        print(f"[Skip] {voxel_name}: load error: {e}")
        return None


def save_latent(save_path, feats_np, coords_np):
    try:
        np.savez_compressed(save_path, feats=feats_np, coords=coords_np)
        print(f"Saved latent to {save_path}")
    except Exception as e:
        print(f"[Error] failed to save {save_path}: {e}")


def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PBR latent encoding.")

    dataset_name = DATASET_NAMES[opt.dataset_name]
    pbr_voxel_save_dir = POSTPROCESS_ROOT / "pbr_ovoxels" / dataset_name
    pbr_latent_save_dir = POSTPROCESS_ROOT / "pbr_latents" / dataset_name
    pbr_latent_save_dir.mkdir(parents=True, exist_ok=True)

    encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    encoder_x2, _, normalization = load_pbr_vae_x2(
        opt.pbr_vae_x2_ckpt,
        use_ema=not opt.no_ema,
        step=opt.step,
    )

    voxel_names = [fname for fname in sorted(os.listdir(pbr_voxel_save_dir)) if fname.endswith(".vxz")]
    print(f"Full: Encoding {len(voxel_names)} {dataset_name} PBR ovoxels")
    start = len(voxel_names) * opt.rank // opt.world_size
    end = len(voxel_names) * (opt.rank + 1) // opt.world_size
    voxel_names = voxel_names[start:end]
    print(f"Rank {opt.rank} / {opt.world_size}: Encoding {len(voxel_names)} objects from {start} to {end}")

    if not opt.overwrite:
        voxel_names = [
            voxel_name for voxel_name in voxel_names
            if not (pbr_latent_save_dir / voxel_name.replace(".vxz", ".npz")).exists()
        ]
        print(f"After skipping existing latents: {len(voxel_names)} objects")

    loader_pool = ThreadPoolExecutor(max_workers=opt.num_loader_workers, thread_name_prefix="loader")
    saver_pool = ThreadPoolExecutor(max_workers=opt.num_saver_workers, thread_name_prefix="saver")

    pending = deque()
    next_idx = 0
    total = len(voxel_names)
    prefetch = max(opt.prefetch, opt.batch_size * 2)

    def submit_next():
        nonlocal next_idx
        while next_idx < total and len(pending) < prefetch:
            voxel_name = voxel_names[next_idx]
            voxel_path = pbr_voxel_save_dir / voxel_name
            pending.append(loader_pool.submit(load_voxel_item, voxel_path, voxel_name))
            next_idx += 1

    save_futures = deque()

    try:
        submit_next()
        pbar = tqdm(total=total, desc="Encoding PBR objects")
        while pending:
            voxels_batch = []
            valid_voxel_names = []

            while len(valid_voxel_names) < opt.batch_size and pending:
                fut = pending.popleft()
                result = fut.result()
                pbar.update(1)
                submit_next()
                if result is None:
                    continue
                voxel_name, voxels = result
                voxels_batch.append(voxels)
                valid_voxel_names.append(voxel_name)

            if not valid_voxel_names:
                continue

            voxels_cat = sparse_cat(voxels_batch, dim=0)
            pbr_slat = encoder(voxels_cat.cuda())
            if not torch.isfinite(pbr_slat.feats).all():
                print(f"[Skip] {valid_voxel_names}: Non-finite latent in pbr_slat.feats")
                clear_cuda_error()
                continue

            pbr_slat_normalized = sp.SparseTensor(
                feats=(pbr_slat.feats - normalization["mean"]) / normalization["std"],
                coords=pbr_slat.coords,
            )
            pbr_z = encoder_x2(pbr_slat_normalized, sample_posterior=False)
            if not torch.isfinite(pbr_z.feats).all():
                print(f"[Skip] {valid_voxel_names}: Non-finite latent in x2 pbr_z.feats")
                clear_cuda_error()
                continue

            pbr_z_list = sparse_unbind(pbr_z, dim=0)
            for pbr_z_item, voxel_name in zip(pbr_z_list, valid_voxel_names):
                feats_np = pbr_z_item.feats.detach().cpu().numpy().astype(np.float32)
                coords_np = pbr_z_item.coords[:, 1:].detach().cpu().numpy().astype(np.uint8)
                save_latent_path = pbr_latent_save_dir / voxel_name.replace(".vxz", ".npz")
                save_futures.append(saver_pool.submit(save_latent, save_latent_path, feats_np, coords_np))

            while save_futures and save_futures[0].done():
                save_futures.popleft().result()

        pbar.close()
        for fut in save_futures:
            fut.result()
    finally:
        loader_pool.shutdown(wait=True)
        saver_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
