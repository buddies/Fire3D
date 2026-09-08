"""Alignment for imported Fire3DReconBaseline results (Gen3DSR, MIDI-3D).

Two separate coordinate problems, and only one of them needs a fit.

**Frames.** Gen3DSR and the GT plys live in the **OpenCV camera frame** at
metric scale (x right, y down, z forward -- the frame you get by unprojecting
`depth_*.npy` with `camera_intrinsics`), so upstream renders them at identity.
Our single_image pipeline instead works in the dataset's **z-up world frame**,
so an imported mesh needs the camera-to-world matrix built from the scene
annotation. That is exact, not fitted: `camera_pose_tran` / `camera_pose_rot`
are in the annotation, the latter in an OpenGL-style basis (x right, y up, z
back), so the OpenCV camera-to-world is `R @ diag(1,-1,-1)` with `t` unchanged.

**MIDI-3D needs a fit.** Its `output.glb` is normalised to roughly a unit cube,
neither metric nor camera-frame, so it takes a per-scene 7-DoF similarity. This
module ports the upstream `--align robust` protocol (`render_rgb.py`) rather
than inventing one, so our renders show the same placement their figures do:

* a **closed-form init** from instance centroids -- MIDI is *given* the
  segmentation, so its i-th output node corresponds to the i-th non-zero id in
  `seg.png`, and that mask plus the metric GT depth says where the instance
  belongs in the camera frame. Umeyama on those few centroid pairs is a direct
  solve.
* a **trimmed similarity ICP** robust to partial overlap: keep the best
  `trim_fwd` (0.9) of pred->GT pairs (the prediction really should lie on the
  GT) and only the best `trim_bwd` (0.5) of GT->pred pairs (unreconstructed GT
  is expected, but dropping the term entirely lets the prediction collapse onto
  one dense GT region). Coarse-to-fine: refine on 8k samples, then 40k.
* both KD-trees built **once**. The backward direction would naively need a tree
  over the moving prediction each iteration; instead the GT is pulled into the
  source frame with `T^-1` against a static tree, and the returned distances are
  rescaled by the similarity's scale -- a similarity preserves
  nearest-neighbour order and scales all distances by exactly `s`, so this
  reproduces the world-frame residual and the same trimming for free.

`sample_surface(seed=)` is passed explicitly: trimesh >= 4 samples from its own
Generator, so `np.random.seed` does not pin it and a re-fit would otherwise
drift run to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The GT world *is* the OpenCV camera frame, so its up axis is -y.
GT_UP = np.array([0.0, -1.0, 0.0])
# Fixed MIDI->GT up alignment; azimuth about GT_UP is the only free rotation.
MIDI_TO_GT_UP = np.diag([1.0, -1.0, -1.0])
# OpenGL-style camera basis (x right, y up, z back) -> OpenCV (x right, y down, z fwd)
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def single_image_camera_to_world(annotation_path: Path) -> np.ndarray:
    """Exact 4x4 camera(OpenCV)->world for a single_image scene. No fitting."""

    data = json.loads(Path(annotation_path).read_text())
    c2w = np.eye(4)
    c2w[:3, :3] = np.asarray(data["camera_pose_rot"], float) @ GL_TO_CV
    c2w[:3, 3] = np.asarray(data["camera_pose_tran"], float)
    return c2w


def scene_intrinsics(annotation_path: Path) -> np.ndarray:
    return np.asarray(
        json.loads(Path(annotation_path).read_text())["camera_intrinsics"], float
    )


def _similarity_from_pairs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama similarity (uniform scale + rotation + translation) src->dst."""

    from trimesh.registration import procrustes

    transform, _, _ = procrustes(
        src, dst, reflection=False, translation=True, scale=True
    )
    return transform


def _rot_about_axis(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def _azimuth_candidates(n: int, base=None, up_axis=GT_UP) -> list[np.ndarray]:
    """`base` composed with n yaw rotations about the GT vertical.

    Only azimuth is free: a single-image method has a fixed camera->pred and the
    GT shares an up axis, so searching full 3D orientation would let a partial
    reconstruction cheat by flipping onto a dense GT region.
    """

    base = MIDI_TO_GT_UP if base is None else base
    return [
        _rot_about_axis(up_axis, angle) @ base
        for angle in np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ]


def _sim_init(src_pts, dst_pts, rotation) -> np.ndarray:
    """Centre, rotate, rescale to the dst rms radius, translate to dst centroid."""

    src_c, dst_c = src_pts.mean(0), dst_pts.mean(0)
    src_r = np.sqrt(np.mean(np.sum((src_pts - src_c) ** 2, axis=1)))
    dst_r = np.sqrt(np.mean(np.sum((dst_pts - dst_c) ** 2, axis=1)))
    scale = dst_r / max(src_r, 1e-12)
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = dst_c - (scale * rotation) @ src_c
    return transform


def _trimmed_similarity_icp(
    src, dst, initial, *, max_iter=60, trim_fwd=0.9, trim_bwd=0.5, tol=1e-9,
    workers=-1,
):
    """Similarity ICP robust to partial overlap. Both KD-trees built once."""

    import trimesh
    from scipy.spatial import cKDTree

    transform = np.asarray(initial, float).copy()
    dst_tree, src_tree = cKDTree(dst), cKDTree(src)
    previous = None
    cost = float("inf")
    for _ in range(max_iter):
        moved = trimesh.transform_points(src, transform)
        d_fwd, i_fwd = dst_tree.query(moved, k=1, workers=workers)
        k_fwd = max(3, int(len(d_fwd) * trim_fwd))
        keep_fwd = np.argpartition(d_fwd, k_fwd - 1)[:k_fwd]

        scale = float(abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0))
        dst_in_src = trimesh.transform_points(dst, np.linalg.inv(transform))
        d_bwd, i_bwd = src_tree.query(dst_in_src, k=1, workers=workers)
        d_bwd = d_bwd * scale  # back into world/GT units
        k_bwd = max(3, int(len(d_bwd) * trim_bwd))
        keep_bwd = np.argpartition(d_bwd, k_bwd - 1)[:k_bwd]

        pairs_src = np.vstack([src[keep_fwd], src[i_bwd[keep_bwd]]])
        pairs_dst = np.vstack([dst[i_fwd[keep_fwd]], dst[keep_bwd]])
        cost = 0.5 * (d_fwd[keep_fwd].mean() + d_bwd[keep_bwd].mean())
        candidate = _similarity_from_pairs(pairs_src, pairs_dst)
        if previous is not None and abs(previous - cost) < tol:
            break
        previous, transform = cost, candidate
    return transform, float(cost)


def sample_mesh_det(mesh, count: int, seed: int = 0) -> np.ndarray:
    """Deterministic surface sampling. `seed=` is what actually pins trimesh>=4."""

    import trimesh

    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, np.float64)


def _lift_depth(depth, K, mask) -> np.ndarray:
    """Unproject masked z-depth pixels into the OpenCV camera frame."""

    rows, cols = np.nonzero(mask)
    z = depth[rows, cols]
    x = (cols - K[0, 2]) * z / K[0, 0]
    y = (rows - K[1, 2]) * z / K[1, 1]
    return np.c_[x, y, z]


def midi_instance_init(midi_glb: Path, seg_png: Path, depth_npy: Path, K):
    """Closed-form similarity init from per-instance centroid pairs.

    Returns (4x4, n_pairs), or (None, 0) when the node/mask mapping is unclear --
    upstream reports 19 of 20 scenes solved from this init, with 3431 falling
    back to the azimuth search.
    """

    import trimesh
    from PIL import Image

    if not (Path(seg_png).is_file() and Path(midi_glb).is_file()):
        return None, 0
    depth = np.load(depth_npy)
    seg = np.asarray(Image.open(seg_png).convert("L"))
    if seg.shape != depth.shape:
        seg = np.asarray(
            Image.fromarray(seg).resize(
                (depth.shape[1], depth.shape[0]), Image.NEAREST
            )
        )
    scene = trimesh.load(str(midi_glb), process=False)
    if not isinstance(scene, trimesh.Scene):
        return None, 0
    nodes = sorted(scene.graph.nodes_geometry, key=lambda n: (len(n), n))
    ids = [int(i) for i in np.unique(seg) if i != 0]
    if len(nodes) != len(ids) or len(ids) < 3:
        return None, 0

    src, dst = [], []
    for node, ident in zip(nodes, ids):
        mask = (seg == ident) & (depth > 0) & (depth < 100)
        if mask.sum() < 50:
            continue
        node_transform, geom_name = scene.graph.get(node)
        points = trimesh.transform_points(
            scene.geometry[geom_name].vertices, node_transform
        )
        src.append(points.mean(0))
        dst.append(_lift_depth(depth, K, mask).mean(0))
    if len(src) < 3:
        return None, 0
    return _similarity_from_pairs(np.asarray(src), np.asarray(dst)), len(src)


def fit_similarity_robust(
    pred_pts, gt_pts, *, seed=0, n_azimuth=24, base_rotation=None,
    n_coarse=8000, n_fine=40000, trim_fwd=0.9, trim_bwd=0.5, init=None,
):
    """Coarse-to-fine trimmed similarity fit; `init` skips the yaw search."""

    rng = np.random.RandomState(seed)

    def subsample(points, count):
        return points[rng.choice(len(points), min(count, len(points)), replace=False)]

    coarse_pred, coarse_gt = subsample(pred_pts, n_coarse), subsample(gt_pts, n_coarse)
    kwargs = dict(trim_fwd=trim_fwd, trim_bwd=trim_bwd)

    candidates = []
    if init is not None:
        candidates.append(
            _trimmed_similarity_icp(coarse_pred, coarse_gt, init, max_iter=40, **kwargs)
        )
    # keep a few yaw inits even with an init, so a bad init cannot win
    searched = 4 if init is not None else n_azimuth
    for rotation in _azimuth_candidates(searched, base=base_rotation):
        candidates.append(
            _trimmed_similarity_icp(
                coarse_pred, coarse_gt,
                _sim_init(coarse_pred, coarse_gt, rotation),
                max_iter=40, **kwargs,
            )
        )
    best = min(candidates, key=lambda c: c[1])[0]
    fine_pred, fine_gt = subsample(pred_pts, n_fine), subsample(gt_pts, n_fine)
    return _trimmed_similarity_icp(fine_pred, fine_gt, best, max_iter=80, **kwargs)


def load_concatenated(path: Path):
    """Load a mesh/scene file as one Trimesh with node transforms baked in."""

    import trimesh

    obj = trimesh.load(str(path), process=False)
    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            raise ValueError(f"{path} has no geometry")
        obj = obj.to_mesh() if hasattr(obj, "to_mesh") else obj.dump(concatenate=True)
    if len(obj.vertices) == 0 or len(obj.faces) == 0:
        raise ValueError(f"{path} has no triangles")
    return obj


def fit_midi_to_camera_frame(
    *, midi_glb: Path, gt_ply: Path, seg_png: Path, depth_npy: Path,
    annotation: Path, n_points: int = 200000, seed: int = 0,
    trim_fwd: float = 0.9, trim_bwd: float = 0.5, use_instances: bool = True,
) -> tuple[np.ndarray, dict]:
    """4x4 mapping MIDI's normalised output into the GT (OpenCV camera) frame."""

    K = scene_intrinsics(annotation)
    gt_pts = sample_mesh_det(load_concatenated(gt_ply), n_points, seed=seed)
    pred_pts = sample_mesh_det(load_concatenated(midi_glb), n_points, seed=seed)
    init, n_pairs = (
        midi_instance_init(midi_glb, seg_png, depth_npy, K)
        if use_instances
        else (None, 0)
    )
    transform, cost = fit_similarity_robust(
        pred_pts, gt_pts, seed=seed, n_azimuth=24, base_rotation=None,
        trim_fwd=trim_fwd, trim_bwd=trim_bwd, init=init,
    )
    scale = float(abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0))
    return transform, {
        "mode": "robust",
        "init": f"instances({n_pairs})" if n_pairs else "azimuth_search",
        "n_instance_pairs": int(n_pairs),
        "scale": scale,
        "trimmed_residual": float(cost),
        "n_points": int(n_points),
        "seed": int(seed),
        "trim_fwd": trim_fwd,
        "trim_bwd": trim_bwd,
    }
