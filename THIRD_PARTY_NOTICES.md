# Third-Party Notices

Fire3D's MIT license applies only to original Fire3D code. The following
components retain their upstream licenses and terms.

| Component | Use in Fire3D | Source | License |
|---|---|---|---|
| DINOv3 | image features | https://github.com/facebookresearch/dinov3 | DINOv3 License (`licenses/DINOV3_LICENSE.md`) |
| AnyUp | feature upsampling | https://github.com/wimmerth/anyup | upstream license (`third_party/anyup/LICENSE`) |
| TRELLIS.2 | sparse decoders/runtime | https://github.com/microsoft/TRELLIS.2 | MIT (`licenses/TRELLIS2_LICENSE`) |
| CuMesh | mesh postprocessing | https://github.com/JeffreyXiang/CuMesh | MIT (`trellis2_x2/CuMesh/LICENSE`) |
| nvdiffrast | differentiable rasterization | https://github.com/NVlabs/nvdiffrast | NVIDIA Source Code License |
| PyTorch3D | 3D operators | https://github.com/facebookresearch/pytorch3d | BSD-3-Clause |
| SceneSmith | optional scene export toolkit | https://github.com/nepfaff/scenesmith | upstream terms at pinned revision |
| AI2-THOR / ProcTHOR | optional scene generation and rendering | https://github.com/allenai/ai2thor | Apache-2.0 |
| Blender | scene and asset conversion/rendering | https://www.blender.org/ | GPL and Blender asset terms |
| ShapeR preprocessing subset | camera and object-evaluation input conversion | https://github.com/xiahongchi/shaper | CC BY-NC 4.0 (`baselines/shaper/LICENSE`) |
| BoxeR loader adapter | baseline RGB-D input conversion | https://github.com/facebookresearch/boxer | CC BY-NC 4.0 (`baselines/boxer_trellis2/LICENSE`) |

The Blender-side O-Voxel export helpers in `data_processing/blender/` are
adapted from the pinned TRELLIS.2 source and retain its MIT license in that
directory. Baseline adapters under `baselines/` are Fire3D integration code;
the baseline implementations are fetched separately at the revisions in
`baselines/registry.json` and retain their own licenses.

The model repository includes selected TRELLIS.2 decoder weights and the
DINOv3 backbone required by the frozen inference protocol. Those files remain
subject to their original licenses. Dataset subsets and processing adapters do
not broaden the terms of SAGE-10K, InternScenes, MansionWorld, iTHOR/ProcTHOR,
SceneSmith, Imaginarium, 3D-FUTURE, ABO, HSSD, or Objaverse. Consult each source
and the Fire3D dataset card before use or redistribution.
