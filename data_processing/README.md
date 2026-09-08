# Data Processing Reference

This directory contains the dataset adapters and shared stages used to build
Fire3D training records. It is a reproducibility reference, not a copy of the
source datasets. Obtain each dataset under its original terms and configure its
local root through an environment variable or command-line argument.

## Processing Graph

```text
scene source -> canonical assets + object transforms
             -> RGB/depth/instance masks/cameras
             -> object and background O-Voxels
             -> sparse-structure, shape, and PBR latent caches

object source -> canonical object adapter
              -> multi-view RGB/depth/mask render + point cloud
              -> O-Voxel
              -> sparse-structure, shape, and PBR latent caches
```

`registry.json` is the machine-readable index of every released adapter and
stage. Validate it before starting a processing job:

```bash
python -m data_processing.validate_registry
```

## Supported Sources

| Source | Canonical export | Render | O-Voxel | Latent encoding |
|---|---:|---:|---:|---:|
| SAGE-10K | yes | Isaac Sim | object + background | shared |
| InternScenes | yes | USD/Omniverse | object + background | shared |
| MansionWorld | yes | native renderer | object + background | shared |
| iTHOR | yes | AI2-THOR | scene export | shared |
| ProcTHOR | yes | AI2-THOR | object + background | shared |
| SceneSmith | yes | native toolkit | object + background | shared |
| Imaginarium | yes | Blender | scene export | shared |
| 3D-FUTURE | object adapter | Blender | shared | shared |
| ABO | object adapter | Blender | shared | shared |
| HSSD | object adapter | Blender | shared | shared |
| Objaverse-XL GitHub | object adapter | Blender | shared | shared |
| Objaverse-XL Sketchfab | object adapter | Blender | shared | shared |

The table means the implementation is included. It does not mean source data,
source licenses, proprietary render runtimes, or third-party checkpoints are
redistributed by Fire3D.

## Source Downloads

Fire3D does not redistribute its training corpora. Download each source from
its official project and review its current license and access requirements
before processing it.

| Training source | Official project or download |
|---|---|
| SAGE-10K | [Hugging Face dataset](https://huggingface.co/datasets/nvidia/SAGE-10k) |
| InternScenes | [Repository and data guide](https://github.com/InternRobotics/InternScenes) |
| MansionWorld | [Hugging Face dataset](https://huggingface.co/datasets/superbigsaw/MansionWorld) |
| iTHOR | [AI2-THOR iTHOR documentation](https://ai2thor.allenai.org/ithor/documentation/) |
| ProcTHOR | [ProcTHOR-10K dataset](https://github.com/allenai/procthor-10k) |
| SceneSmith | [Repository and data setup](https://github.com/nepfaff/scenesmith) |
| Imaginarium | [Hugging Face dataset](https://huggingface.co/datasets/HiHiAllen/Imaginarium-Dataset) |
| Object collections | [TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K) and its [preparation guide](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md) |

TRELLIS-500K is the canonical starting point for the object path. Its metadata
and preparation tools cover the ObjaverseXL GitHub and Sketchfab subsets, ABO,
3D-FUTURE, and HSSD represented by the released adapters. Fire3D supplies the
subsequent normalization, rendering, O-Voxel, and latent-encoding stages.

## Paths

Shared stages use these roots:

```bash
export FIRE3D_TRAINING_ROOT=/path/to/processed/scene_datasets
export FIRE3D_OBJECT_ROOT=/path/to/processed/object_datasets
export FIRE3D_MODEL_ROOT=$PWD/checkpoints/Fire3D
export FIRE3D_MANIFEST_ROOT=/path/to/training/manifests
export FIRE3D_DEPENDENCY_ROOT=/path/to/external/toolkits
export FIRE3D_BLENDER=$PWD/blender/blender
```

Dataset-specific variables include `FIRE3D_SAGE10K_ROOT`,
`FIRE3D_INTERNSCENES_ROOT`, `FIRE3D_MANSIONWORLD_ROOT`,
`FIRE3D_ITHOR_ROOT`, `FIRE3D_PROCTHOR_ROOT`, `FIRE3D_SCENESMITH_ROOT`, and
`FIRE3D_IMAGINARIUM_ROOT`. Object adapters similarly accept
`FIRE3D_3D_FUTURE_ROOT`, `FIRE3D_ABO_ROOT`, `FIRE3D_HSSD_ROOT`,
`FIRE3D_OBJAVERSE_GITHUB_ROOT`, and `FIRE3D_OBJAVERSE_SKETCHFAB_ROOT`.
Command-line arguments take precedence where a script exposes both forms.

## External Toolkits

SceneSmith export imports helpers from a pinned upstream checkout. Fetch it
into the Git-ignored processing workspace with:

```bash
python -m data_processing.fetch scenesmith
```

The exact repository and revision are recorded in `registry.json`. Other native
environments must be installed according to their source projects: Isaac Sim
for SAGE-10K, USD/Omniverse for InternScenes, AI2-THOR for iTHOR and ProcTHOR,
and Blender for Imaginarium and object conversion. The shared O-Voxel and
latent stages run in the Fire3D environment.

## Object Path

Each object adapter exposes the same four-function contract:
`list_all_model_paths`, `build_metadata_mapping`, `save_metadata_mapping`, and
`load_model`. For example:

```bash
python data_processing/objects/3d_future/adapter.py --help
python data_processing/objects/abo/adapter.py --help
python data_processing/stages/render/render_object.py --help
python data_processing/stages/render/generate_pcd.py --help
```

Adapters apply the same axis conversion, center the source bounds, and scale
the longest extent just below one. Metadata and path indexes are written below
`outputs/data_processing/` by default and are not tracked by Git.

## Scene Path

Use the paths in `registry.json` rather than copying one-off launch scripts.
Every dataset entry names its canonical export/transform and render stages.
The five scene corpora used for flow training additionally provide separate
object and background O-Voxel stages.

```bash
python data_processing/datasets/ai2thor/ithor/export_scene_final.py --help
python data_processing/datasets/scenesmith/export_all.py --help
python data_processing/stages/ovoxel/voxelize_procthor_objects.py --help
python data_processing/stages/ovoxel/voxelize_procthor_scene_bg.py --help
```

Long-running launch wrappers, scheduler files, logs, extracted source kits, and
generated payloads belong under `temp_scripts/`, `outputs/`, `data/`, or
`data_processing/_upstream/`; all are ignored by Git.

## Latent Path

The shared encoders consume normalized O-Voxels or shape support and write the
cache contracts consumed by the three flow trainers:

```bash
python data_processing/stages/latents/ovoxel_seq.py --help
python data_processing/stages/latents/shape_enc_seq.py --help
python data_processing/stages/latents/shape_enc_rot_aug_seq.py --help
python data_processing/stages/latents/pbr_enc_seq.py --help
python data_processing/stages/latents/pbr_enc_rot_aug_seq.py --help
python data_processing/stages/latents/ss_enc_seq.py --help
```

Shape and PBR HC-VAE encoders resolve the stable public
`ckpts/{encoder,decoder}.pt` aliases. Legacy numbered training checkpoints are
accepted only when explicitly requested. Sparse-structure targets retain their
artifact ID and encoder SHA-256 contract so incompatible latent caches fail
before training.

## Validation Boundary

The release test suite checks registry coverage, pinned upstream revisions,
adapter imports, synthetic mesh discovery and normalization, stable checkpoint
resolution, CLI construction, and Python compilation. Dataset-scale rendering
still requires legal access to each source corpus and its native runtime; those
jobs are intentionally not treated as unit tests.
