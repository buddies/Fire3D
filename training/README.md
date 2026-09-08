# Training

The public recipes expose the model and data contracts used for Fire3D's
perception model, three cascaded flow-matching models, and shape/PBR HC-VAEs.
The release includes training logic and data-processing reference code, but it
does not redistribute the full training datasets.

## Paths

Set these roots before launching perception or flow training:

```bash
export FIRE3D_TRAINING_ROOT=/path/to/processed/scene_datasets
export FIRE3D_OBJECT_ROOT=/path/to/processed/object_datasets
export FIRE3D_MODEL_ROOT=$PWD/checkpoints/Fire3D
export FIRE3D_MANIFEST_ROOT=/path/to/training/manifests
export FIRE3D_DINOV3_ROOT=$PWD/third_party/dinov3
```

HC-VAE configs accept a comma-separated list or a JSON source mapping through
`FIRE3D_HCVAE_ROOTS`. Every root follows the TRELLIS.2 metadata/latent layout
described by `trellis2_x2/trellis2/datasets/components.py`.

## Launch

```bash
# Scene perception
accelerate launch --num_processes 2 -m training.perception.train \
  --config configs/training/perception/default.yaml \
  --output_dir outputs/perception --exp_name fire3d_perception

# Cascaded flow models
accelerate launch --num_processes 8 -m training.flow_matching.train_ss \
  --config configs/training/flow_matching/ss.yaml \
  --output_dir outputs/flow --exp_name sparse_structure
accelerate launch --num_processes 8 -m training.flow_matching.train_shape \
  --config configs/training/flow_matching/shape.yaml \
  --output_dir outputs/flow --exp_name shape
accelerate launch --num_processes 8 -m training.flow_matching.train_pbr \
  --config configs/training/flow_matching/pbr.yaml \
  --output_dir outputs/flow --exp_name pbr

# Shape and PBR HC-VAEs (one GPU each)
CUDA_VISIBLE_DEVICES=0 python -m training.hcvae.train \
  --config configs/training/hcvae/shape.yaml
CUDA_VISIBLE_DEVICES=0 python -m training.hcvae.train \
  --config configs/training/hcvae/pbr.yaml
```

The released HC-VAE recipes are trained with one process on one GPU. The
`batch_size_per_gpu` values in their configs are therefore also the effective
global batch sizes.

Perception preserves the released point-normalized `[0,24]` position-token
contract, local-up 90-degree yaw invariance in Hungarian matching and rotation
loss, and background pose/segmentation as instance zero. Its public recipe uses
discrete scene yaw, sensor noise, instance yaw/deletion/transform, internal
copy-paste, and external object paste at full intensity from the first step.

The sparse-structure flow checks an artifact ID and encoder SHA-256 before
loading targets. This prevents silently mixing incompatible latent caches while
keeping private training iteration numbers out of public paths.

## Evaluation

Perception validation uses the same deterministic mini-validation path as
training:

```bash
python -m eval.perception.evaluate --help
```

HC-VAE latent reconstruction can be evaluated with the released encoder and
decoder pairs:

```bash
python -m training.hcvae.evaluate --kind shape \
  --encoder checkpoints/Fire3D/reconstruction/vae/shape/ckpts/encoder.pt \
  --decoder checkpoints/Fire3D/reconstruction/vae/shape/ckpts/decoder.pt \
  --output results/hcvae_shape.json
```

Install training dependencies with `pip install -e '.[training,evaluation]'`.
