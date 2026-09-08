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
import torch
from tqdm import tqdm

import trellis2.models as models
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


def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def load_ss_vae_x2(ss_vae_x2_ckpt, use_ema=True, step=None):
    config, encoder_ckpt_path, decoder_ckpt_path, label = resolve_vae_checkpoints(
        ss_vae_x2_ckpt, use_ema=use_ema, step=step
    )
    print(f"Loading sparse-structure VAE {label}...")
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

    return encoder_x2, decoder_x2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_NAMES.keys()))
    parser.add_argument(
        "--ss_vae_x2_ckpt",
        type=str,
        default=str(REPO_ROOT / "checkpoints/Fire3D/reconstruction/vae/ss"),
        help="SS VAE x2 checkpoint",
    )
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_loader_workers", type=int, default=8)
    parser.add_argument("--num_saver_workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=256,
                        help="Number of shape latent load tasks to keep in-flight")
    return parser.parse_args()


def load_shape_latent_item(shape_latent_path, latent_name, resolution):
    try:
        with np.load(shape_latent_path) as shape_latent:
            coords = shape_latent["coords"]
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"expected coords with shape [N, 3], got {coords.shape}")
        if coords.size > 0 and (coords.min() < 0 or coords.max() >= resolution):
            raise ValueError(f"coords out of range for resolution {resolution}")

        ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.float32)
        coords = torch.from_numpy(coords.astype(np.int64, copy=False)).long()
        if coords.numel() > 0:
            ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        return latent_name, ss
    except Exception as e:
        print(f"[Skip] {latent_name}: load error: {e}")
        return None


def save_latent(save_path, ss_latent_np):
    try:
        np.savez_compressed(save_path, ss_latent=ss_latent_np)
        print(f"Saved SS latent to {save_path}")
    except Exception as e:
        print(f"[Error] failed to save {save_path}: {e}")


def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SS latent encoding.")

    dataset_name = DATASET_NAMES[opt.dataset_name]
    shape_latent_dir = POSTPROCESS_ROOT / "shape_latents" / dataset_name
    ss_latent_save_dir = POSTPROCESS_ROOT / "ss_latents" / dataset_name
    ss_latent_save_dir.mkdir(parents=True, exist_ok=True)

    encoder_x2, _ = load_ss_vae_x2(
        opt.ss_vae_x2_ckpt,
        use_ema=not opt.no_ema,
        step=opt.step,
    )

    latent_names = [fname for fname in sorted(os.listdir(shape_latent_dir)) if fname.endswith(".npz")]
    print(f"Full: Encoding {len(latent_names)} {dataset_name} SS latents from shape latents")
    start = len(latent_names) * opt.rank // opt.world_size
    end = len(latent_names) * (opt.rank + 1) // opt.world_size
    latent_names = latent_names[start:end]
    print(f"Rank {opt.rank} / {opt.world_size}: Encoding {len(latent_names)} objects from {start} to {end}")

    if not opt.overwrite:
        latent_names = [
            latent_name for latent_name in latent_names
            if not (ss_latent_save_dir / latent_name).exists()
        ]
        print(f"After skipping existing SS latents: {len(latent_names)} objects")

    loader_pool = ThreadPoolExecutor(max_workers=opt.num_loader_workers, thread_name_prefix="loader")
    saver_pool = ThreadPoolExecutor(max_workers=opt.num_saver_workers, thread_name_prefix="saver")

    pending = deque()
    next_idx = 0
    total = len(latent_names)
    prefetch = max(opt.prefetch, opt.batch_size * 2)

    def submit_next():
        nonlocal next_idx
        while next_idx < total and len(pending) < prefetch:
            latent_name = latent_names[next_idx]
            shape_latent_path = shape_latent_dir / latent_name
            pending.append(loader_pool.submit(load_shape_latent_item, shape_latent_path, latent_name, opt.resolution))
            next_idx += 1

    save_futures = deque()

    try:
        submit_next()
        pbar = tqdm(total=total, desc="Encoding SS objects")
        while pending:
            ss_batch = []
            valid_latent_names = []

            while len(valid_latent_names) < opt.batch_size and pending:
                fut = pending.popleft()
                result = fut.result()
                pbar.update(1)
                submit_next()
                if result is None:
                    continue
                latent_name, ss = result
                ss_batch.append(ss)
                valid_latent_names.append(latent_name)

            if not valid_latent_names:
                continue

            ss = torch.stack(ss_batch, dim=0).cuda()
            ss_latent = encoder_x2(ss, sample_posterior=False)
            if not torch.isfinite(ss_latent).all():
                print(f"[Skip] {valid_latent_names}: Non-finite SS latent")
                clear_cuda_error()
                continue

            ss_latent = ss_latent.detach().cpu().numpy().astype(np.float32)
            for latent_name, ss_latent_item in zip(valid_latent_names, ss_latent):
                save_latent_path = ss_latent_save_dir / latent_name
                save_futures.append(saver_pool.submit(save_latent, save_latent_path, ss_latent_item))

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
