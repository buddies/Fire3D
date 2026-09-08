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

The model repository includes selected TRELLIS.2 decoder weights and the
DINOv3 backbone required by the frozen inference protocol. Those files remain
subject to their original licenses. Dataset subsets retain the terms of their
source datasets; consult the dataset card before use or redistribution.
