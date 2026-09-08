# Release validation

Fire3D is released only after the public commands pass inference, camera
sampling, and rendering checks on each supported dataset family. Generated
artifacts and model weights are intentionally excluded from Git.

## Validation matrix

| Dataset | Validation scenes | Protocol | Inference | Rendering |
| --- | --- | --- | --- | --- |
| iTHOR | `iTHOR_FloorPlan24_physics`, `iTHOR_FloorPlan306_physics`, `iTHOR_FloorPlan312_physics`, `iTHOR_FloorPlan328_physics` | `fire3d_video_v1` | Passed | Passed, 16 views |
| Imaginarium | `bedroom_01`, `bedroom_17`, `bedroom_35`, `computer_room_03` | `fire3d_video_v1` | Passed | Passed, 16 views |
| ScanNet++ | `09bced689e` | `fire3d_scannetpp_v1` | Passed | Passed |
| Single image | `003025`, `003084` | `fire3d_single_image_v1` | Passed | Passed at the input camera |

The iTHOR, Imaginarium, and ScanNet++ protocols enable the background room-box
prior, unit-box pruning, and `room_box_isotropic_enclose` canonical transform.
The scene audit written beside each reconstruction must report
`background_room_box_prior.canonical_transform.applied: true` and zero points
outside the canonical unit box. Single-image inference keeps its independent
perception-OBB background transform because a room cannot be estimated reliably
from every input photograph.

## Required gates

1. `fire3d --help` and every public subcommand help page run without imports failing.
2. `ruff check .` and `pytest -q` pass in the release workspace.
3. Every validation scene completes through the public `fire3d infer` command.
4. Camera sampling and rendering complete through the public interfaces.
5. Rendered images are nonblank and receive a visual geometry/alignment audit.
6. A clean environment installation and clean checkout reproduce the commands.
7. Hugging Face archives pass checksum, whitelist, extraction, and smoke tests.
8. Model bundle paths use stable aliases and do not expose private iteration
   identifiers in filenames, protocols, manifests, or model-card text.

The authoritative validation is performed on `sl-gpu-02`; the local workspace
is a mirror of the committed release state.

## ScanNet++ release-example audit

The advertised ScanNet++ scene, `09bced689e`, was rerun on September 8, 2026
through the public `fire3d infer` command with `fire3d_scannetpp_v1`. The run
completed perception, batched reconstruction, camera sampling, and rendering
for all 30 predicted instances. Its background audit records
`canonical_transform.applied: true`, reduces 63,674 previously out-of-box
points to zero, and bounds the retained 369,344 background points within
`[-0.49, 0.49]`. The authoritative generated output is
`results/release_validation_scannetpp_09bced689e_20260908/` on `sl-gpu-02`;
it remains excluded from Git.

## Published artifact audit

The public artifacts were verified on September 7, 2026:

- Model repository revision:
  `da94067a2e6e3a4ed18936b80c0c9549c3258fcf`.
- Dataset repository revision:
  `29b23029058dd828d08f6f8483ba1e796a4a76f2`.
- All 376 dataset archives match the byte sizes recorded in `manifest.json`;
  the remote manifest is byte-for-byte identical to the staged manifest.
- The dataset inventory contains exactly 67 iTHOR, 120 Imaginarium, 165
  ScanNet++, and 20 single-image scenes selected by the release whitelists.
- Every published iTHOR and Imaginarium scene includes 60 exact-camera RGB
  rerenders. All 11,220 archived images across 187 scenes match the
  authoritative `renders_updated` images byte-for-byte.
- Both Hugging Face repository cards include the current teaser, method
  overview, and HC-VAE figures under stable asset names.
