---
license: other
library_name: pytorch
pipeline_tag: image-to-3d
tags:
  - 3d-scene-reconstruction
  - rgb-d
  - single-image
  - flow-matching
  - textured-mesh
---

# FIRE3D: Feed-forward Interactive 3D Scene Reconstruction Within A Minute

Fire3D is a unified feed-forward framework that transforms a single RGB image
or casual RGB video into simulation-ready 3D scene assets. It predicts a
compositional scene representation with object-level 6-DoF pose, bounding box,
mesh geometry, and texture, without test-time optimization.

This release provides the inference code, model checkpoints, processed example
inputs, and frozen protocols needed to reproduce Fire3D results on iTHOR,
Imaginarium, ScanNet++, and single-image scenes.

The inference code and installation instructions are maintained at
[xiahongchi/Fire3D](https://github.com/xiahongchi/Fire3D). Processed release
inputs are hosted at
[datasets/hongchi/Fire3D](https://huggingface.co/datasets/hongchi/Fire3D).

## Overview

![Fire3D reconstructs compositional, simulation-ready 3D scenes from RGB-D observations.](assets/teaser.png)

**Teaser.** Fire3D reconstructs a compositional 3D scene from a single RGB
image or casual RGB video. The output contains object-level pose, geometry,
and material assets that can be rendered, edited, and used in simulation.

![Fire3D end-to-end perception and reconstruction method.](assets/method_overview.png)

**Method overview.** Fire3D lifts RGB-D observations into a shared 3D scene
representation, predicts object instances and 6-DoF poses, and reconstructs
the sparse structure, shape, and PBR fields with three cascaded flow-matching
models. Batched decoders and mesh post-processing produce the final textured
scene assets.

![Fire3D Hierarchical Compression VAE architecture.](assets/hcvae.png)

**HC-VAE.** The Hierarchical Compression VAE reduces each sparse SC-VAE shape
or material field from a `32^3 x 32` representation to an `8^3 x 64` latent.
This 32x reduction makes scene-level batched flow sampling practical while
retaining the frozen TRELLIS.2 decoding path.

## Model Bundle

The repository contains the scene perception checkpoint, the three
flow-matching checkpoints, sparse VAE decoders, latent normalization
statistics, and the TRELLIS.2 shape/PBR decoders used by the frozen release
protocols. `manifest.json` records the byte size and SHA-256 digest of every
inference file; `checksums.sha256` provides the same values in standard form.

Fire3D's Hierarchical Compression VAE (HC-VAE) compresses each sparse SC-VAE
shape and material field from a `32^3` representation with 32 channels into an
`8^3` representation with 64 channels. This 32x reduction enables batched flow
sampling and VAE decoding across many scene instances while retaining the
frozen TRELLIS.2 SC-VAE decoding path.

| Component | Bundle path |
|---|---|
| Scene perception | `perception/model.pt` |
| Sparse-structure flow | `reconstruction/flows/ss/model.pt` |
| Shape flow | `reconstruction/flows/shape/model.pt` |
| PBR flow | `reconstruction/flows/pbr/model.pt` |
| Sparse-structure VAE | `reconstruction/vae/ss/ckpts/decoder.pt` |
| Shape HC-VAE | `reconstruction/vae/shape/ckpts/decoder.pt` |
| PBR HC-VAE | `reconstruction/vae/pbr/ckpts/decoder.pt` |
| DINOv3 encoder | `external/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| TRELLIS.2 shape decoder | `external/trellis2/shape_dec_next_dc_f16c32_fp16.safetensors` |
| TRELLIS.2 PBR decoder | `external/trellis2/tex_dec_next_dc_f16c32_fp16.safetensors` |

Public filenames intentionally do not encode private training iteration
numbers.

The default protocols use classifier-free guidance 3, flow and VAE batches of
16, and the batch CuMesh postprocessor. The iTHOR, Imaginarium, and ScanNet++
protocols enable the validated background room-box fit before background
reconstruction.

## Use

```bash
git clone https://github.com/xiahongchi/Fire3D.git
cd Fire3D
bash scripts/install.sh
conda activate fire3d
bash scripts/install_blender.sh

fire3d download --models
fire3d download --data --dataset ithor --scene-id iTHOR_FloorPlan312_physics
bash scripts/run_ithor_example.sh
```

The downloader places this repository under `checkpoints/Fire3D/` and verifies
the release manifest before inference.

## Scope

The released checkpoints are intended for inference through the frozen Fire3D
protocols. Performance can degrade for scenes with inaccurate camera poses or
depth, severe occlusion, unusual scale, non-room backgrounds, or objects far
outside the training distribution. The generated meshes should be reviewed
before use in safety-critical or physically deployed applications.

## Licenses

Original Fire3D code is MIT licensed. This model bundle includes DINOv3 and
selected TRELLIS.2 files, which retain their upstream licenses. Review the
included `licenses/` directory and the repository's
`THIRD_PARTY_NOTICES.md` before use or redistribution.

## Citation

```bibtex
@misc{xia2026fire3d,
  title={FIRE3D: Feed-forward Interactive 3D Scene Reconstruction Within A Minute},
  author={Hongchi Xia and Tianhang Cheng and Wei-Chiu Ma and Shenlong Wang},
  year={2026}
}
```
