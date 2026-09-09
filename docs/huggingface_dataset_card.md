---
license: other
task_categories:
  - image-to-3d
tags:
  - 3d-scene-reconstruction
  - rgb-d
  - ithor
  - imaginarium
  - scannetpp
  - single-image
---

# Fire3D Processed Inference Data

[Paper](https://arxiv.org/pdf/2609.08848) |
[Code](https://github.com/xiahongchi/Fire3D) |
[Models](https://huggingface.co/hongchi/Fire3D)

Fire3D is a unified feed-forward framework that transforms a single RGB image
or casual RGB video into simulation-ready 3D scene assets. It predicts a
compositional scene representation with object-level 6-DoF pose, bounding box,
mesh geometry, and texture, without test-time optimization.

This release provides the inference code, model checkpoints, processed example
inputs, and frozen protocols needed to reproduce Fire3D results on iTHOR,
Imaginarium, ScanNet++, and single-image scenes.

This dataset contains only processed, inference-ready inputs selected by the
Fire3D release whitelists. Model files are hosted at
[hongchi/Fire3D](https://huggingface.co/hongchi/Fire3D), and code is maintained
at [xiahongchi/Fire3D](https://github.com/xiahongchi/Fire3D).

This repository is not a training-data release. Fire3D does not redistribute
its training corpora; all training sources remain available from their public
projects under their original terms. For object training data, use
[TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K).
Official scene-data links and Fire3D preprocessing entry points are listed in
the [code release](https://github.com/xiahongchi/Fire3D/tree/main/data_processing).

## Overview

![Fire3D reconstructs compositional, simulation-ready 3D scenes from RGB-D observations.](assets/teaser.png)

**Teaser.** Fire3D reconstructs a compositional 3D scene from a single RGB
image or casual RGB video. The output contains object-level pose, geometry,
and material assets that can be rendered, edited, and used in simulation.

[![Fire3D end-to-end perception and reconstruction method.](assets/method_overview.png)](assets/method_overview.pdf)

**Method overview.** Fire3D lifts RGB-D observations into a shared 3D scene
representation, predicts object instances and 6-DoF poses, and reconstructs
the sparse structure, shape, and PBR fields with three cascaded flow-matching
models. Batched decoders and mesh post-processing produce the final textured
scene assets.

[![Fire3D Hierarchical Compression VAE architecture.](assets/hcvae.png)](assets/hcvae.pdf)

**HC-VAE.** The Hierarchical Compression VAE reduces each sparse SC-VAE shape
or material field from a `32^3 x 32` representation to an `8^3 x 64` latent.
This 32x reduction makes scene-level batched flow sampling practical while
retaining the frozen TRELLIS.2 decoding path.

## Contents

| Dataset adapter | Whitelisted scenes | Published input |
|---|---:|---|
| iTHOR | 67 | 60-view RGB-D sequence, masks, camera, transform, updated exact-camera RGB |
| Imaginarium | 120 | 60-view RGB-D sequence, masks, camera, transform, updated exact-camera RGB |
| ScanNet++ | 165 | up to 300 RGB frames, cameras, Pi3 depth and confidence |
| Single image | 20 | RGB, aligned point cloud, compact native/reconstruction camera record |

The root `manifest.json` maps every dataset and scene to a deterministic TAR
archive and records its SHA-256 digest. Per-scene archives keep the Hugging Face
repository below its practical file-count limit while allowing selective
downloads. The Fire3D downloader verifies and safely extracts each archive.

## Download

```bash
git clone https://github.com/xiahongchi/Fire3D.git
cd Fire3D
bash scripts/install.sh
conda activate fire3d

fire3d download --data --dataset ithor --scene-id iTHOR_FloorPlan312_physics
fire3d download --data --dataset imaginarium --scene-id bedroom_01
fire3d download --data --dataset scannetpp --scene-id 09bced689e
fire3d download --data --dataset single_image --scene-id 003025
fire3d download --evaluation shaper
```

Omit `--scene-id` to download all published scenes for one adapter. Data is
extracted under `data/{ithor,Imaginarium,scannetpp,single_image}`.

The optional ShapeR evaluation bundle is extracted under
`data/evaluation/shaper/`. It is a compact, pickle-free derivative containing
the meshes, bounds, transforms, and condition points required by Fire3D's
geometry evaluator. It retains the upstream ShapeR evaluation-data terms and
includes the source license.

## Processing

iTHOR and Imaginarium use the exact-camera RGB rerenders validated by the
release protocol. ScanNet++ uses Pi3 depth with retained confidence and drops
pixels below confidence 0.6 rather than filling invalid depth. The single-image
adapter uses the source camera pose and an aligned point cloud. Fire3D room-box
fitting is an inference operation and does not modify the published source
points.

## Sources And Terms

The archives are processed subsets of iTHOR/AI2-THOR, Imaginarium, ScanNet++,
and `TianhangCheng7/Fire3DSingleImageData`. They remain subject to their source
datasets' licenses, access conditions, and citation requirements. This dataset
card does not replace or broaden those terms. Users are responsible for
checking the upstream terms before downloading, redistributing, or using a
subset.

## Citation

Please cite Fire3D and the source dataset corresponding to each subset used.

```bibtex
@article{xia2026fire3d,
  title={{FIRE3D}: Feed-forward Interactive 3D Scene Reconstruction Within A Minute},
  author={Xia, Hongchi and Cheng, Tianhang and Ma, Wei-Chiu and Wang, Shenlong},
  journal={arXiv preprint arXiv:2609.08848},
  year={2026},
  url={https://arxiv.org/pdf/2609.08848}
}
```
