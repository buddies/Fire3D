# Training

The public recipes expose the model and data contracts used for Fire3D's
perception model, three cascaded flow-matching models, sparse-structure VAE,
and shape/PBR HC-VAEs. The release includes training logic and data-processing
reference code, but it does not redistribute the training corpora. All source
datasets are publicly available and retain their original licenses and access
terms. See the [source-download
table](../data_processing/README.md#source-downloads). For object training data,
start from
[TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K)
rather than downloading each constituent object collection independently.

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

The sparse-structure VAE accepts comma-separated object roots through
`FIRE3D_SSVAE_ROOTS`. Each root is a directory of object NPZ files produced by
the shape HC-VAE encoding stage, and each file contains its `8^3` support in a
`coords` array.

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

# Sparse-structure VAE encoder and decoder (one GPU)
CUDA_VISIBLE_DEVICES=0 python -m training.ssvae.train \
  --config configs/training/ssvae/default.yaml

# Shape and PBR HC-VAEs (one GPU each)
CUDA_VISIBLE_DEVICES=0 python -m training.hcvae.train \
  --config configs/training/hcvae/shape.yaml
CUDA_VISIBLE_DEVICES=0 python -m training.hcvae.train \
  --config configs/training/hcvae/pbr.yaml
```

The released VAE recipes use one process on one GPU. The
`batch_size_per_gpu` values in their configs are therefore also the effective
global batch sizes.

## Sparse-Structure VAE

The standard SS-VAE path trains the encoder and decoder jointly from random
initialization. It does not freeze either model. Each training target is the
binary support of an `8^3` shape HC-VAE latent. Four exact quarter-yaw rotations
provide augmentation without interpolation. The objective is
binary-cross-entropy with logits plus a `1e-6` KL penalty. The released config
uses an `8 x 2 x 2 x 2` latent bottleneck, EMA `0.9999`, and one GPU.

Prepare the shape HC-VAE latents first, then point the trainer to one or more
processed roots:

```bash
python data_processing/stages/latents/shape_enc_seq.py --help

export FIRE3D_SSVAE_ROOTS=/path/to/source-a,/path/to/source-b
CUDA_VISIBLE_DEVICES=0 python -m training.ssvae.train \
  --config configs/training/ssvae/default.yaml
```

To resume from the latest complete encoder, decoder, and optimizer state:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.ssvae.train \
  --config configs/training/ssvae/default.yaml \
  --resume outputs/ss_vae
```

Use a matched encoder and decoder from the same joint run. Before training a
new sparse-structure flow model, encode every target with that selected EMA
encoder, recompute its latent statistics, and update the artifact ID and
SHA-256 contract in the flow config. The released pretrained flow remains
pinned to its original latent contract; swapping only its encoder checksum is
not valid.

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
