import argparse
from ast import arg
import os
import sys
import json
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.interpolate import CubicHermiteSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from contextlib import contextmanager
import bpy
import imageio
from tqdm import tqdm
import mathutils
import OpenEXR
import Imath
import random

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

from tex_utils_local import (
    dict_to_floor_plan,
    export_layout_to_mesh_dict_list_v2
)


def _murmur3_32(data, seed=0):
    """MurmurHash3 32-bit (x86) - matches Blender's util_murmur_hash3 for Cryptomatte.
    Returns unsigned 32-bit hash. `data` is bytes or str (encoded as utf-8).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    length = len(data)
    nblocks = length // 4
    h1 = seed
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    for i in range(nblocks):
        k1 = int.from_bytes(data[i * 4 : (i + 1) * 4], "little") & 0xFFFFFFFF
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF
    tail = data[nblocks * 4 :]
    k1 = 0
    if len(tail) >= 3:
        k1 ^= tail[2] << 16
    if len(tail) >= 2:
        k1 ^= tail[1] << 8
    if len(tail) >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
    h1 ^= length
    h1 ^= (h1 >> 16) & 0xFFFFFFFF
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= (h1 >> 13) & 0xFFFFFFFF
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= (h1 >> 16) & 0xFFFFFFFF
    return h1


def _cryptomatte_manifest_from_exr_header(header):
    """Parse Cryptomatte manifest from EXR header if present.
    Returns dict mapping object name (str) -> hash (uint32), or None if not found.
    """
    # Blender / Cryptomatte spec: manifest is a string attribute, often under a key like
    # "cryptomatte/0/manifest" or "cryptomatte/manifest" (layer 0 = Object).
    manifest_str = None
    try:
        keys = list(header.keys()) if hasattr(header, "keys") else []
    except Exception:
        keys = []
    for key in keys:
        if "manifest" in key.lower() and "cryptomatte" in key.lower():
            val = header[key]
            if hasattr(val, "value"):
                val = val.value
            if isinstance(val, str):
                manifest_str = val
                break
    if manifest_str is None and hasattr(header, "get"):
        for attr in ("cryptomatte/0/manifest", "cryptomatte/manifest", "cryptomatte/1/manifest"):
            try:
                val = header.get(attr)
            except Exception:
                continue
            if val is not None:
                if hasattr(val, "value"):
                    val = val.value
                if isinstance(val, str):
                    manifest_str = val
                    break
    if not manifest_str:
        return None
    try:
        # Manifest format: JSON object {"Object Name": "hex8", ...}
        manifest = json.loads(manifest_str)
        out = {}
        for name, hex_id in manifest.items():
            try:
                out[name] = int(hex_id, 16) & 0xFFFFFFFF
            except (ValueError, TypeError):
                continue
        return out if out else None
    except (json.JSONDecodeError, TypeError):
        return None


def _build_cryptomatte_hash_to_pass_index(room, manifest_name_to_hash=None):
    """Build mapping from Cryptomatte hash (uint32) to pass_index (1-based) for scene objects.
    If manifest_name_to_hash is provided (from EXR), use it; else compute hashes via MurmurHash3.
    """
    object_ids = sorted([obj.id for obj in room.objects])
    obj_id_to_pass_index = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    hash_to_pass = {0: 0}
    if manifest_name_to_hash:
        for name, h in manifest_name_to_hash.items():
            if name in obj_id_to_pass_index:
                hash_to_pass[h] = obj_id_to_pass_index[name]
        return hash_to_pass
    for mesh_id in object_ids:
        h = _murmur3_32(mesh_id)
        hash_to_pass[h] = obj_id_to_pass_index[mesh_id]
    return hash_to_pass


# Constants
MIN_DIST_OBSTACLE = 0.5  # User requirement
CAMERA_RADIUS = 0.5      # Safety margin for camera body
WORLD_UP = np.array([0, 0, 1])

# Step limits for adaptive sampling
MAX_TRANS_STEP = 0.05
MAX_ROT_STEP = np.radians(0.5)

class CameraPlannerEnv:
    def __init__(self, room_bounds, all_meshes, interest_meshes=None):
        """
        room_bounds: list/array [x_min, y_min, z_min, x_max, y_max, z_max]
        all_meshes: list of trimesh.Trimesh objects (walls, floor, objects)
        interest_meshes: list of trimesh.Trimesh objects (only objects to focus on)
        """
        self.bounds = np.array(room_bounds)

        # 1. MERGE MESHES
        if all_meshes:
            self.scene_mesh = trimesh.util.concatenate(all_meshes)
        else:
            self.scene_mesh = trimesh.Trimesh() # Empty mesh if no objects

        print("Finished merging meshes")

        if interest_meshes:
            self.interest_mesh = trimesh.util.concatenate(interest_meshes)
        else:
            self.interest_mesh = trimesh.Trimesh()

        print("Finished merging interest meshes")

        # 2. BUILD COLLISION ENGINE (All Meshes)
        self.use_kdtree = True

        if len(self.scene_mesh.faces) > 0:
            try:
                # Sample points from the surface (fast) for collision
                self.collision_points, _ = trimesh.sample.sample_surface(self.scene_mesh, 100000)
                from scipy.spatial import cKDTree
                print(f"Sampled {len(self.collision_points)} collision points, shape: {self.collision_points.shape}")
                self.collision_kdtree = cKDTree(self.collision_points, balanced_tree=False)
                print("Finished building collision kdtree")
            except Exception as e:
                print(f"Warning: Collision point sampling failed ({e}).")
                self.use_kdtree = False
        else:
            self.use_kdtree = False

        print("Finished building collision engine")

        # 3. RAY INTERSECTOR
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector
            self.intersector = RayMeshIntersector(self.scene_mesh)
        except ImportError:
            from trimesh.ray.ray_triangle import RayMeshIntersector
            self.intersector = RayMeshIntersector(self.scene_mesh)

        print("Finished building ray intersector")

    def is_valid_location(self, point, min_dist=0.1):
        """
        Check if the camera position is inside the room AND
        far enough from obstacles.
        """
        # A. Room Bounds Check (Simple AABB)
        if not (np.all(point > self.bounds[:3]) and np.all(point < self.bounds[3:])):
            return False

        if not self.use_kdtree:
            return True

        # B. Obstacle Distance Check (Approximate using KDTree of ALL surface points)
        dist, _ = self.collision_kdtree.query(point, k=1)

        if dist < min_dist:
             return False

        return True

    def is_view_clear(self, origin, target, min_view_dist=0.1):
        """
        Check if the 'forward' ray hits an object too early.
        """
        direction = np.array(target) - np.array(origin)
        dist_to_target = np.linalg.norm(direction)

        if dist_to_target < 1e-6: return False # Target too close to origin

        direction = direction / dist_to_target

        if len(self.scene_mesh.faces) == 0:
            return True

        # Cast a single ray
        hit_points, _, _ = self.intersector.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )

        if len(hit_points) == 0:
            return True

        # Check distance to the closest hit
        closest_hit_dist = np.min(np.linalg.norm(hit_points - origin, axis=1))

        # If the ray hits something closer than the target (with some margin), view is blocked
        # Ideally we want to see the target, so if hit < dist_to_target, it's blocked.
        # However, target is on surface, so we might hit target itself.
        if closest_hit_dist < dist_to_target - 0.1:
            return False

        return True

    def ray_hits_mesh(self, origin, direction):
        """
        Check if a ray from origin in direction hits the mesh.
        """
        if len(self.scene_mesh.faces) == 0:
            return False

        hit_points, _, _ = self.intersector.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )

        return len(hit_points) > 0

def check_view_up(forward, threshold=0.2):
    # Check if up vector satisfies constraint: abs(up . world_up) > 0.2
    # forward: (3,)
    right = np.cross(forward, WORLD_UP)
    if np.linalg.norm(right) < 1e-6:
        return False
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return abs(np.dot(up, WORLD_UP)) > threshold

def slerp_vector(v0, v1, t_array):
    # v0, v1: (3,) unit vectors
    # t_array: (N,) or float, 0..1
    # Returns (N, 3)

    if np.isscalar(t_array):
        t_array = np.array([t_array])

    dot = np.dot(v0, v1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    if abs(sin_theta) < 1e-6:
        # Linear interpolation if parallel
        res = (1 - t_array)[:, np.newaxis] * v0 + t_array[:, np.newaxis] * v1
        norms = np.linalg.norm(res, axis=1)
        # Avoid division by zero
        norms[norms < 1e-6] = 1.0
        return res / norms[:, np.newaxis]

    w0 = np.sin((1 - t_array) * theta) / sin_theta
    w1 = np.sin(t_array * theta) / sin_theta

    res = w0[:, np.newaxis] * v0 + w1[:, np.newaxis] * v1
    return res

def get_interpolated_forwards(eval_dists, cum_dist, forwards):
    """
    Interpolate forward vectors at given distances along the path.
    """
    new_F = []
    current_seg = 0

    # Ensure eval_dists are within range
    eval_dists = np.clip(eval_dists, cum_dist[0], cum_dist[-1])

    for d in eval_dists:
        # Find segment
        while current_seg < len(cum_dist) - 2 and d > cum_dist[current_seg+1]:
            current_seg += 1

        t_segment_start = cum_dist[current_seg]
        t_segment_end = cum_dist[current_seg+1]

        segment_len = t_segment_end - t_segment_start
        if segment_len < 1e-6:
            t = 0.0
        else:
            t = (d - t_segment_start) / segment_len
            t = np.clip(t, 0.0, 1.0)

        f0 = forwards[current_seg]
        f1 = forwards[current_seg+1]

        res = slerp_vector(f0, f1, t)
        new_F.append(res[0])

    return np.array(new_F)

def get_object_view_candidates(env, obj_mesh, num_samples=300):
    """
    Sample candidate views (position, target) around a specific object mesh.
    """
    candidates = []

    if obj_mesh is None or obj_mesh.is_empty:
        return candidates

    center = obj_mesh.centroid
    # max_extent = np.max(obj_mesh.bounding_box.extents)
    extent_x = obj_mesh.bounding_box.extents[0] * 0.5
    extent_y = obj_mesh.bounding_box.extents[1] * 0.5
    extent_z = obj_mesh.bounding_box.extents[2] * 0.5
    max_extent = max(extent_x, extent_y, extent_z)

    # Sample points on spheres at different radii
    # radii = [1.5 * max_extent, 2.0 * max_extent, 2.5 * max_extent, 3.0 * max_extent]

    for _ in range(num_samples):
        # Random direction on sphere
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(0, 60.0 * np.pi / 180.0)

        direction = np.array([
            np.cos(azimuth) * np.sin(elevation),
            np.sin(azimuth) * np.sin(elevation),
            np.cos(elevation)
        ])

        r_direction = direction * max_extent
        scale_r = np.max(np.array([r_direction[0] / extent_x, r_direction[1] / extent_y, r_direction[2] / extent_z]))
        r_direction = r_direction / scale_r

        for _ in range(10):
            r_x = np.random.uniform(1.0, 3.0)
            r_y = np.random.uniform(1.0, 3.0)
            r_z = np.random.uniform(1.0, 3.0)
            r = np.array([r_x, r_y, r_z])
            cam_pos = center + direction * r

            # Check if position is valid
            if not env.is_valid_location(cam_pos, min_dist=CAMERA_RADIUS):
                # print(f"Invalid location, too close to obstacles")
                continue

            # Vector from camera to object center
            forward = center - cam_pos
            f_norm = np.linalg.norm(forward)
            if f_norm < 1e-6: continue
            forward /= f_norm

            # Check constraints
            if not check_view_up(forward):
                # print(f"Invalid view up")
                continue

            # if not env.is_view_clear(cam_pos, center):
            #     print(f"Invalid view clear")
            #     continue

            candidates.append({
                'position': cam_pos,
                'target': center,
                'forward': forward
            })

    return candidates

def sample_rectangle_edges(xy_mins, xy_maxs, num_samples=1):
    x_min, y_min = xy_mins
    x_max, y_max = xy_maxs

    width = x_max - x_min
    height = y_max - y_min
    perimeter = 2 * (width + height)

    # 1. Generate random distances along the total perimeter length
    distances = np.random.uniform(0, perimeter, num_samples)

    points = np.zeros((num_samples, 2))

    for i, d in enumerate(distances):
        if d < width:
            # Bottom edge: x varies, y is min
            points[i] = [x_min + d, y_min]
        elif d < (width + height):
            # Right edge: x is max, y varies
            points[i] = [x_max, y_min + (d - width)]
        elif d < (2 * width + height):
            # Top edge: x varies, y is max
            points[i] = [x_max - (d - width - height), y_max]
        else:
            # Left edge: x is min, y varies
            points[i] = [x_min, y_max - (d - 2 * width - height)]

    return points

def generate_random_anchor(env):
    pos = np.random.uniform(env.bounds[:3], env.bounds[3:])
    if not env.is_valid_location(pos, min_dist=CAMERA_RADIUS):
        return None
    # Look at top random point
    # target = np.random.uniform(env.bounds[:3], env.bounds[3:])
    # target[2] = env.bounds[5] * np.random.uniform(0.75, 1.0)

    target = np.zeros_like(pos)
    target[:2] = sample_rectangle_edges(env.bounds[0:2], env.bounds[3:5], num_samples=1).reshape(-1)
    target[2] = env.bounds[5] * np.random.uniform(0.75, 1.0)

    forward = target - pos

    f_norm = np.linalg.norm(forward)
    if f_norm < 1e-6:
        return None
    forward /= f_norm

    # Check constraints
    if not check_view_up(forward, threshold=0.4):
        # print(f"Invalid view up")
        return None

    return {
        'position': pos,
        'target': target,
        'forward': forward,
        'weight': 3.0
    }


def generate_anchors(env, room, mesh_dict, max_anchors=20):
    """
    Generate anchor points based on object importance logic.
    """

    # 1. Classify objects
    wall_ids = set(w.id for w in room.walls)

    # Count occurrences of objects being placed on other objects
    place_counts = Counter()
    children_info = {}
    for obj in room.objects:
        place_counts[obj.place_id] += 1
        children_info[obj.place_id] = children_info.get(obj.place_id, []) + [obj.id]

    all_candidates = []

    # Weight Constants
    BASE_SCORE_FLOOR = 2.0
    BASE_SCORE_WALL = 1.0
    BONUS_PER_CHILD = 1.0

    processed_count = 0

    for obj in room.objects:
        # Check if mesh exists
        if obj.id not in mesh_dict:
            continue

        weight = 0.0
        is_target = False

        if obj.place_id == 'floor':
            weight = BASE_SCORE_FLOOR
            is_target = True
        elif obj.place_id == 'wall':
            weight = BASE_SCORE_WALL
            is_target = True

        if is_target:
            # Add bonus for children objects (objects on top)
            weight += place_counts[obj.id] * BONUS_PER_CHILD


            # Generate candidates
            mesh_info = mesh_dict.get(obj.id)

            target_mesh = mesh_info['mesh']
            children_meshes = []
            for child_id in children_info.get(obj.id, []):
                child_mesh = mesh_dict.get(child_id)['mesh']
                children_meshes.append(child_mesh)

            target_mesh = trimesh.util.concatenate([target_mesh] + children_meshes)

            cands = get_object_view_candidates(env, target_mesh)
            if len(cands) > 2:
                cands = cands[:2]
            for c in cands:
                c['weight'] = weight
                c['obj_id'] = obj.id
                all_candidates.append(c)
            processed_count += 1

    print(f"Processed {processed_count} objects for anchors with weighted scoring.")

    selected_anchors = []

    if len(all_candidates) > 0:

        weights = np.array([c['weight'] for c in all_candidates])
        weights /= np.sum(weights)

        # Use indices
        indices = np.arange(len(all_candidates))

        num_select = min(int(max_anchors * 0.8), len(all_candidates))
        selected_indices = np.random.choice(indices, size=num_select, replace=False, p=weights)

        for idx in selected_indices:
            selected_anchors.append(all_candidates[idx])

    # add some random anchors
    num_random_anchors_added = 0
    random_anchors = []
    print("env bounds: ", env.bounds)
    for _ in range(max(max_anchors, len(all_candidates)) * 10):
        cand = generate_random_anchor(env)
        if cand is not None:
            random_anchors.append(cand)
            num_random_anchors_added += 1
    print(f"# random anchors added: {num_random_anchors_added}")

    if len(random_anchors) > 0:
        weights = np.array([c['weight'] for c in random_anchors])
        weights /= np.sum(weights)

        # Use indices
        indices = np.arange(len(random_anchors))

        num_select = min(max_anchors - len(selected_anchors), len(random_anchors))
        selected_indices = np.random.choice(indices, size=num_select, replace=False, p=weights)

        for idx in selected_indices:
            selected_anchors.append(random_anchors[idx])

    print(f"# selected anchors: {len(selected_anchors)}; max_anchors: {max_anchors}")

    return selected_anchors

def connect_anchors_tsp(anchors, start_idx=0):
    """
    Connect anchors using a greedy nearest neighbor approach, considering both translation and rotation.
    """
    if not anchors:
        return []

    path = [anchors[start_idx]]
    remaining = anchors[:start_idx] + anchors[start_idx+1:]

    current = anchors[start_idx]

    # Weights for distance metric
    W_TRANS = 1.0
    W_ROT = 1.5 # 1.5 meter equivalent per radian of rotation

    while remaining:
        # Find closest
        best_cost = float('inf')
        best_idx = -1

        curr_pos = current['position']
        curr_fwd = current['forward']

        for i, cand in enumerate(remaining):
            # Translation distance
            dist_trans = np.linalg.norm(cand['position'] - curr_pos)

            # Rotation distance (angle between forward vectors)
            dot = np.dot(curr_fwd, cand['forward'])
            dot = np.clip(dot, -1.0, 1.0)
            dist_rot = np.arccos(dot) # Radians [0, pi]

            cost = W_TRANS * dist_trans + W_ROT * dist_rot

            if cost < best_cost:
                best_cost = cost
                best_idx = i

        current = remaining.pop(best_idx)
        path.append(current)

    return path

def generate_smooth_path(anchors, num_frames=300, env=None):
    """
    Interpolate smoothly between anchors using Cubic Hermite Spline.
    """
    if len(anchors) < 2:
        return np.array([anchors[0]['position']]*num_frames), np.array([anchors[0]['forward']]*num_frames)

    positions = np.array([a['position'] for a in anchors])
    forwards = np.array([a['forward'] for a in anchors])

    # Add tangents for Catmull-Rom style or just heuristic
    # tangent[i] ~ (P[i+1] - P[i-1]) / 2
    # For start/end, use difference
    n = len(positions)
    tangents = np.zeros_like(positions)

    for i in range(n):
        prev_p = positions[max(0, i-1)]
        next_p = positions[min(n-1, i+1)]
        tangents[i] = (next_p - prev_p) * 0.5 # tension

    # Create spline
    # Parametrize by cumulative distance
    dists = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]

    if total_dist < 1e-6:
        return np.resize(positions, (num_frames, 3)), np.resize(forwards, (num_frames, 3))

    spline = CubicHermiteSpline(cum_dist, positions, tangents)

    # --- Adaptive Sampling Strategy ---
    # 1. Sample densely to estimate complexity
    # Use a high enough resolution to capture curvature
    num_dense = max(num_frames * 10, 2000)
    dense_dists = np.linspace(0, total_dist, num_dense)
    dense_P = spline(dense_dists)

    # Interpolate forwards at dense points
    dense_F = get_interpolated_forwards(dense_dists, cum_dist, forwards)

    # 2. Compute costs per segment
    # Translation cost
    delta_trans = np.linalg.norm(dense_P[1:] - dense_P[:-1], axis=1)

    # Rotation cost
    dot_prods = np.sum(dense_F[1:] * dense_F[:-1], axis=1)
    dot_prods = np.clip(dot_prods, -1.0, 1.0)
    delta_rot = np.arccos(dot_prods)

    # Combined cost (normalized by limits)
    # We want step <= LIMIT, so cost = step / LIMIT
    step_costs = np.maximum(delta_trans / MAX_TRANS_STEP, delta_rot / MAX_ROT_STEP)

    # Integrate cost to get "effort" coordinate
    cum_effort = np.concatenate(([0], np.cumsum(step_costs)))
    total_effort = cum_effort[-1]

    # 3. Generate intermediate high-res path based on effort
    # We want enough frames so that each step is small (<= limits)
    # total_effort is roughly the number of steps needed at limit.
    # Add safety factor and ensure at least num_frames
    ideal_num_frames = int(np.ceil(total_effort * 1.2))
    intermediate_num = max(num_frames, ideal_num_frames)

    # Distribute points uniformly in effort space
    target_effort = np.linspace(0, total_effort, intermediate_num)

    # Map target effort back to distance
    # interp1d(x=cum_effort, y=dense_dists)
    dist_mapper = interp1d(cum_effort, dense_dists, kind='linear')
    eval_dists = dist_mapper(target_effort)

    # Initial intermediate positions
    inter_P = spline(eval_dists)

    # --- Collision Avoidance and Smoothing (on intermediate path) ---
    if env is not None and env.use_kdtree:
        # Check if we have collision info
        # Increase iterations for better convergence with smoothing
        for iteration in range(15):
            # 1. Check collisions
            dists_to_obs, indices = env.collision_kdtree.query(inter_P)

            # Identify violating points
            # Use a slightly larger margin for the path than for static anchors to be safe
            safe_margin = CAMERA_RADIUS + 0.1
            violations = dists_to_obs < safe_margin

            # If no violations and we have done at least one smoothing pass (except if perfectly clean initially)
            if not np.any(violations) and iteration > 0:
                break

            # 2. Push points away
            if np.any(violations):
                near_obs_pts = env.collision_points[indices[violations]]
                cam_pts = inter_P[violations]

                push_dirs = cam_pts - near_obs_pts
                dirs_norm = np.linalg.norm(push_dirs, axis=1)

                # Handle concentric case (rare)
                safe_mask = dirs_norm > 1e-6
                push_dirs[~safe_mask] = np.random.normal(size=(np.sum(~safe_mask), 3))
                push_dirs[~safe_mask] /= np.linalg.norm(push_dirs[~safe_mask], axis=1)[:, np.newaxis]
                dirs_norm[~safe_mask] = 1.0

                push_dirs = push_dirs / dirs_norm[:, np.newaxis]

                # Push amount: how much deeper are we than safe_margin?
                needed_push = safe_margin - dists_to_obs[violations]

                # Add a small buffer to push slightly further to account for smoothing pulling it back
                inter_P[violations] += push_dirs * (needed_push[:, np.newaxis] + 0.05)

            # 3. Smooth the path to avoid jaggedness
            # Use Gaussian smoothing for higher quality results
            if len(inter_P) > 5:
                # Apply smoothing
                # Use sigma=2.0 for reasonable smoothness.
                # Since inter_P is dense (small steps), sigma=2.0 is a local smoothing.
                smoothed_P = gaussian_filter1d(inter_P, sigma=2.0, axis=0, mode='nearest')

                # Anchor constraints: keep start/end fixed
                smoothed_P[0] = inter_P[0]
                smoothed_P[-1] = inter_P[-1]

                inter_P = smoothed_P

            # 4. Enforce room bounds
            inter_P = np.maximum(inter_P, env.bounds[:3] + CAMERA_RADIUS)
            inter_P = np.minimum(inter_P, env.bounds[3:] - CAMERA_RADIUS)

    # Calculate intermediate orientations
    inter_F = get_interpolated_forwards(eval_dists, cum_dist, forwards)

    # --- 4. Resample to final num_frames ---
    if intermediate_num == num_frames:
        return inter_P, inter_F

    # Resample
    t_inter = np.linspace(0, 1, intermediate_num)
    t_final = np.linspace(0, 1, num_frames)

    # Linear interpolation for positions
    resampler_P = interp1d(t_inter, inter_P, axis=0, kind='linear')
    final_P = resampler_P(t_final)

    # Linear interpolation for forwards (safe because dense)
    resampler_F = interp1d(t_inter, inter_F, axis=0, kind='linear')
    final_F = resampler_F(t_final)

    # Normalize forwards
    norms = np.linalg.norm(final_F, axis=1)
    norms[norms < 1e-6] = 1.0
    final_F = final_F / norms[:, np.newaxis]

    return final_P, final_F

def setup_camera_look_at(camera, camera_pos, lookat_pos):
    """Position camera and make it look at target position"""
    # Set camera position
    camera.location = camera_pos

    # Calculate direction vector
    direction = mathutils.Vector(lookat_pos) - mathutils.Vector(camera_pos)

    # Point camera to look at target
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def generate_camera_trajectory(
    room_bounds,
    all_meshes,
    num_frames=300,
    complexity=10,
    env=None,
    room=None,
    mesh_dict=None
):
    # 0. Preprocessing
    if env is None:
        print("preprocessing: Building environment...")
        env = CameraPlannerEnv(room_bounds, all_meshes)

    # 1. Generate Anchors
    anchors = generate_anchors(env, room, mesh_dict, max_anchors=complexity)

    # 2. Connect Anchors
    # Start with a random one as the first point
    start_idx = np.random.randint(0, len(anchors))
    sorted_anchors = connect_anchors_tsp(anchors, start_idx)

    # 3. Generate Smooth Path
    trajectory_P, trajectory_F = generate_smooth_path(sorted_anchors, num_frames, env=env)

    trajectory_poses = []

    for i in range(num_frames):
        curr_P = trajectory_P[i]
        curr_F = trajectory_F[i]

        # 4. Compute Orientation (LookAt)
        forward = curr_F
        dist = np.linalg.norm(forward)
        if dist < 1e-6: forward = np.array([1, 0, 0])
        else: forward = forward / dist

        right = np.cross(forward, WORLD_UP)
        if np.linalg.norm(right) < 1e-6:
             right = np.array([1, 0, 0])

        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        R_mat = np.column_stack([right, up, -forward])

        # Calculate lookat target from forward vector
        target_pt = curr_P + forward * 2.0

        trajectory_poses.append({
            'position': curr_P,
            'rotation': R_mat,
            'target': target_pt
        })

    return trajectory_poses

def get_room_meshes(layout, layout_dir):
    mesh_info_dict = export_layout_to_mesh_dict_list_v2(layout, layout_dir)
    all_meshes = []
    interest_meshes = []

    # Identify object IDs
    object_ids = set()
    for r in layout.rooms:
        for obj in r.objects:
            object_ids.add(obj.id)

    for mesh_id, mesh_info in mesh_info_dict.items():
        if "mesh" in mesh_info:
            m = mesh_info["mesh"]
            all_meshes.append(m)
            # Check if this mesh corresponds to an object
            if mesh_id in object_ids:
                interest_meshes.append(m)

    print(f"Found {len(all_meshes)} meshes, {len(interest_meshes)} object meshes")

    return all_meshes, interest_meshes, mesh_info_dict

def setup_scene_lighting(scene, room_size_dict):
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_nodes.clear()
    world_bg = world_nodes.new(type='ShaderNodeBackground')
    world_bg.inputs[0].default_value = (1, 1, 1, 1)
    world_bg.inputs[1].default_value = 1.0
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(world_output.inputs['Surface'], world_bg.outputs['Background'])

    # add ceiling grid lights with 1.0m spacing
    offset = 1.0
    grid_size = 2.5
    ceiling_z = room_size_dict['height'] - offset

    # Grid spans from offset to width/length - offset
    x_start = offset
    x_end = room_size_dict['width'] - offset
    y_start = offset
    y_end = room_size_dict['length'] - offset

    x_positions = np.linspace(x_start, x_end, abs(int((x_end - x_start) / grid_size)) + 2)
    y_positions = np.linspace(y_start, y_end, abs(int((y_end - y_start) / grid_size)) + 2)

    for xi, x in enumerate(x_positions):
        for yi, y in enumerate(y_positions):
            bpy.ops.object.light_add(type='POINT', location=(x, y, ceiling_z))
            light = bpy.context.active_object
            light.name = f"CeilingGridLight_{xi}_{yi}"
            light.data.energy = 100.0
            light.data.color = (0.9, 0.8, 0.7)
            light.data.shadow_soft_size = 0.1

def get_or_create_collection(collection_name):
    """Get or create a collection"""
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]

    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def clear_blender_scene():
    """Clear all objects from Blender scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear all collections except the default Scene Collection
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def load_scene_meshes_into_blender(room, mesh_info_dict):
    """Load room layout meshes from a precomputed mesh_info_dict into Blender"""

    # Clear all existing Blender assets before loading new ones
    clear_blender_scene()

    # Create collection for scene objects
    scene_collection = get_or_create_collection("scene_objects")

    # Identify object IDs for pass index assignment
    # Sort for determinism
    object_ids = sorted([obj.id for obj in room.objects])
    obj_id_to_pass_index = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    print(f"Assigned pass indices for {len(object_ids)} objects (indices 1-{len(object_ids)})")

    # Import each mesh
    for mesh_id, mesh_info in mesh_info_dict.items():
        # if mesh_id.startswith("door") or mesh_id.startswith("window"):
        #     continue

        trimesh_mesh = mesh_info["mesh"]

        # Convert trimesh to Blender mesh
        vertices = trimesh_mesh.vertices
        faces = trimesh_mesh.faces

        # Create new mesh data
        mesh_data = bpy.data.meshes.new(name=f"mesh_{mesh_id}")
        mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh_data.update()

        # Create object from mesh
        obj = bpy.data.objects.new(mesh_id, mesh_data)

        # Assign pass index
        if mesh_id in obj_id_to_pass_index:
            obj.pass_index = obj_id_to_pass_index[mesh_id]
        else:
            obj.pass_index = 0

        scene_collection.objects.link(obj)

        # Load and apply texture if available
        texture_info = mesh_info.get("texture")
        if texture_info and texture_info.get("texture_map_path"):
            texture_path = texture_info["texture_map_path"]
            if os.path.exists(texture_path):
                # Create material with texture
                mat = bpy.data.materials.new(name=f"mat_{mesh_id}")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()

                # Create shader nodes
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                tex_image = nodes.new(type='ShaderNodeTexImage')
                output = nodes.new(type='ShaderNodeOutputMaterial')

                # Load texture image
                tex_image.image = bpy.data.images.load(texture_path)
                tex_image.image.colorspace_settings.name = 'sRGB'  # Ensure correct color space

                # Configure BSDF for clean, slightly glossy appearance like reference images
                bsdf.inputs['Roughness'].default_value = texture_info.get("roughness_factor", 1.0)  # Slight gloss
                # Blender 4.0+ replaced "Specular" with "IOR Level" (controls amount of specular reflection)
                specular_val = texture_info.get("metallic_factor", 0.03)  # Subtle specularity
                if 'Specular' in bsdf.inputs:
                    bsdf.inputs['Specular'].default_value = specular_val
                elif 'IOR Level' in bsdf.inputs:
                    bsdf.inputs['IOR Level'].default_value = specular_val
                # Sheen Tint: Blender 3.x = value (0–1), Blender 4.x = color (RGBA)
                if 'Sheen Tint' in bsdf.inputs:
                    st = bsdf.inputs['Sheen Tint']
                    if st.type == 'VALUE':
                        st.default_value = 0.0  # No sheen (3.x)
                    else:
                        st.default_value = (1.0, 1.0, 1.0, 1.0)  # White = no tint (4.x)

                # Connect nodes
                mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
                mat.node_tree.links.new(output.inputs['Surface'], bsdf.outputs['BSDF'])

                # Apply material to object
                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)

                # Set UV coordinates if available
                vts = texture_info.get("vts")
                fts = texture_info.get("fts")
                if vts is not None and fts is not None:
                    # Create UV layer
                    uv_layer = obj.data.uv_layers.new(name="UVMap")
                    for face_idx, face in enumerate(fts):
                        for vert_idx in range(len(face)):
                            loop_idx = face_idx * len(face) + vert_idx
                            if loop_idx < len(uv_layer.data):
                                uv = vts[face[vert_idx]]
                                uv_layer.data[loop_idx].uv = (uv[0], uv[1])

    print(f"Loaded {len(mesh_info_dict)} meshes into Blender scene")


def setup_blender_scene_for_rendering(room, mesh_info_dict, output_dir, resolution, render_depth=True):
    """
    Load scene meshes and configure Blender for rendering (engine, passes, compositor).
    Call once when rendering multiple samples; then use render_trajectory_only() per sample.
    """
    print("Loading scene meshes into Blender...")
    load_scene_meshes_into_blender(room, mesh_info_dict)

    scene = bpy.context.scene

    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    scene.camera = camera

    camera.data.type = 'PERSP'
    camera.data.sensor_fit = 'VERTICAL'
    # camera.data.angle set per sample in render_trajectory_only (fov can vary)

    # --- ENGINE SETUP ---
    if args.engine == 'CYCLES':
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 4
        scene.cycles.use_denoising = True
        scene.cycles.device = 'GPU'
        if hasattr(scene.cycles, 'tile_size'):
            scene.cycles.tile_size = min(4096, max(2048, resolution))
            print(f"Tile size: {scene.cycles.tile_size}")
        if hasattr(scene.cycles, 'use_auto_tile'):
            scene.cycles.use_auto_tile = True
            print(f"Use auto tile: {scene.cycles.use_auto_tile}")
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01
        scene.cycles.max_bounces = 2
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 1
        scene.cycles.transmission_bounces = 1
        scene.cycles.volume_bounces = 0
        scene.cycles.transparent_max_bounces = 1
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = "OPTIX"
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            if device.type != 'CPU':
                device.use = True
            else:
                device.use = False
        scene.render.use_persistent_data = True
    else:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        scene.eevee.taa_render_samples = 16
        scene.eevee.use_gtao = True
        scene.eevee.use_shadows = True

    room_size_dict = {
        'width': room.dimensions.width,
        'length': room.dimensions.length,
        'height': room.dimensions.height
    }
    setup_scene_lighting(scene, room_size_dict)

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = 95
    scene.render.film_transparent = False

    # --- PASS SETUP ---
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_combined = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    if args.engine == 'CYCLES':
        scene.view_layers["ViewLayer"].use_pass_object_index = True
        scene.view_layers["ViewLayer"].use_pass_cryptomatte_object = False
    else:
        scene.view_layers["ViewLayer"].use_pass_object_index = False
        scene.view_layers["ViewLayer"].use_pass_cryptomatte_object = True
    bpy.context.view_layer.update()

    # --- COMPOSITOR (placeholder paths; overwritten per sample in render_trajectory_only) ---
    placeholder_dir = os.path.join(os.path.abspath(output_dir), "_blender_setup")
    frames_placeholder = os.path.join(placeholder_dir, "frames")
    masks_placeholder = os.path.join(placeholder_dir, "masks")
    depth_placeholder = os.path.join(placeholder_dir, "depth")
    os.makedirs(masks_placeholder, exist_ok=True)
    if render_depth:
        os.makedirs(depth_placeholder, exist_ok=True)

    scene.render.use_compositing = True
    tree = scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl_node = tree.nodes.new('CompositorNodeRLayers')
    comp_node = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(rl_node.outputs['Image'], comp_node.inputs['Image'])
    # Viewer Node receives the beauty pass so we can read pixels in Python (Render Result often has no pixel data in bpy)
    viewer_node = tree.nodes.new('CompositorNodeViewer')
    viewer_node.name = "RGBViewer"
    tree.links.new(rl_node.outputs['Image'], viewer_node.inputs['Image'])

    mask_output_node = tree.nodes.new('CompositorNodeOutputFile')
    mask_output_node.name = "MaskOutput"
    mask_output_node.base_path = masks_placeholder + "/"
    if args.engine == 'CYCLES':
        mask_output_node.format.file_format = 'OPEN_EXR'
        mask_output_node.format.color_depth = '32'
        mask_output_node.format.color_mode = 'RGB'
        mask_output_node.file_slots[0].path = "mask_"
        mask_output_node.file_slots[0].use_node_format = True
        math_node = tree.nodes.new('CompositorNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = 1.0
        tree.links.new(rl_node.outputs['IndexOB'], math_node.inputs[0])
        tree.links.new(math_node.outputs[0], mask_output_node.inputs[0])
    else:
        mask_output_node.format.file_format = 'OPEN_EXR_MULTILAYER'
        mask_output_node.format.color_depth = '32'
        mask_output_node.file_slots.clear()
        mask_output_node.file_slots.new('CryptoObject00')
        mask_output_node.file_slots.new('CryptoObject01')
        mask_output_node.file_slots.new('CryptoObject02')
        tree.links.new(rl_node.outputs['CryptoObject00'], mask_output_node.inputs['CryptoObject00'])
        tree.links.new(rl_node.outputs['CryptoObject01'], mask_output_node.inputs['CryptoObject01'])
        tree.links.new(rl_node.outputs['CryptoObject02'], mask_output_node.inputs['CryptoObject02'])

    if render_depth:
        depth_output_node = tree.nodes.new('CompositorNodeOutputFile')
        depth_output_node.name = "DepthOutput"
        depth_output_node.base_path = depth_placeholder + "/"
        depth_output_node.format.file_format = 'OPEN_EXR'
        depth_output_node.format.color_depth = '32'
        depth_output_node.format.color_mode = 'RGB'
        depth_output_node.file_slots[0].path = "depth_"
        depth_output_node.file_slots[0].use_node_format = True
        tree.links.new(rl_node.outputs['Depth'], depth_output_node.inputs[0])

    print("Blender scene setup complete (reuse for each trajectory).")


def _read_render_result_to_array():
    """
    Read the compositor beauty pass into a numpy array (RGB uint8).
    Prefer 'Viewer Node' then 'Render Result' (both are often empty in background mode).
    Returns (H, W, 3) C-contiguous uint8 array.
    """
    render_image = bpy.data.images.get("Viewer Node") or bpy.data.images.get("Render Result")
    if render_image is None:
        raise RuntimeError("Neither 'Viewer Node' nor 'Render Result' found.")
    w, h = render_image.size
    if w <= 0 or h <= 0:
        scene = bpy.context.scene
        w, h = scene.render.resolution_x, scene.render.resolution_y
        if w <= 0 or h <= 0:
            raise RuntimeError("Render/Viewer image has zero size and scene resolution is invalid.")
        return np.zeros((h, w, 3), dtype=np.uint8, order='C')
    n = w * h * 4
    pixels = np.empty(n, dtype=np.float32)
    render_image.pixels.foreach_get(pixels)
    pixels = pixels.reshape((h, w, 4))
    rgb = (pixels[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.flipud(rgb))


def _read_frame_from_file(path):
    """Load a single rendered image file (e.g. PNG/JPEG) into RGB uint8 array (H, W, 3)."""
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img.astype(np.uint8))


def _process_mask_exr_to_arrays(room, masks_dir):
    """
    Read mask EXR files from masks_dir, convert to int16, return stacked array (N, H, W).
    Used when saving to a single npz; does not write per-frame npz or remove EXRs (caller cleans up).
    """
    if not os.path.exists(masks_dir):
        return None
    exr_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.exr')])
    if not exr_files:
        return None
    if args.engine == 'CYCLES':
        first_path = os.path.join(masks_dir, exr_files[0])
        img = bpy.data.images.load(first_path)
        width, height = img.size
        bpy.data.images.remove(img)
        masks_list = []
        for exr_fname in tqdm(exr_files, desc="Loading masks to memory"):
            exr_path = os.path.join(masks_dir, exr_fname)
            img = bpy.data.images.load(exr_path)
            pixels = np.empty(width * height * 4, dtype=np.float32)
            img.pixels.foreach_get(pixels)
            pixels = pixels.reshape((height, width, 4))
            mask = np.flipud(pixels[:, :, 0]).astype(np.int16)
            masks_list.append(mask)
            bpy.data.images.remove(img)
        return np.stack(masks_list, axis=0)
    else:
        first_path = os.path.join(masks_dir, exr_files[0])
        hash_to_pass_index = {0: 0}
        try:
            first_exr = OpenEXR.InputFile(first_path)
            header = first_exr.header()
            manifest_name_to_hash = _cryptomatte_manifest_from_exr_header(header)
            first_exr.close()
            hash_to_pass_index = _build_cryptomatte_hash_to_pass_index(room, manifest_name_to_hash)
        except Exception:
            hash_to_pass_index = _build_cryptomatte_hash_to_pass_index(room, None)
        masks_list = []
        for exr_fname in tqdm(exr_files, desc="Loading masks to memory (Eevee)"):
            exr_path = os.path.join(masks_dir, exr_fname)
            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            channels = header['channels'].keys()
            if 'CryptoObject00.R' in channels:
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                raw_bytes = exr_file.channel('CryptoObject00.R', pt)
                mask_float = np.frombuffer(raw_bytes, dtype=np.float32)
                mask_uint32 = mask_float.view(np.uint32).reshape((height, width))
                mask = np.zeros_like(mask_uint32, dtype=np.int16)
                for h in np.unique(mask_uint32):
                    mask[mask_uint32 == h] = hash_to_pass_index.get(h, 0)
            else:
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                raw_bytes = exr_file.channel('R', pt)
                mask_float = np.frombuffer(raw_bytes, dtype=np.float32).reshape((height, width))
                mask = mask_float.astype(np.int16)
            masks_list.append(mask)
            exr_file.close()
        return np.stack(masks_list, axis=0)


def _process_depth_exr_to_arrays(depth_dir):
    """
    Read depth EXR files from depth_dir, convert to float16, return stacked array (N, H, W).
    Used when saving to a single npz; does not write per-frame npz or remove EXRs (caller cleans up).
    """
    if not depth_dir or not os.path.exists(depth_dir):
        return None
    exr_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.exr')])
    if not exr_files:
        return None
    depth_list = []
    for exr_fname in tqdm(exr_files, desc="Loading depths to memory"):
        exr_path = os.path.join(depth_dir, exr_fname)
        img = bpy.data.images.load(exr_path)
        width, height = img.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        img.pixels.foreach_get(pixels)
        pixels = pixels.reshape((height, width, 4))
        depth = np.flipud(pixels[:, :, 0]).astype(np.float16)
        depth_list.append(depth)
        bpy.data.images.remove(img)
    return np.stack(depth_list, axis=0)


def _get_ramdisk_base():
    """Return a base path for temporary render output (prefer /dev/shm to avoid disk I/O)."""
    if os.path.exists("/dev/shm"):
        return "/dev/shm"
    return None


def render_trajectory_only(room, trajectory, output_path, fov=30.0, resolution=512, fps=30, render_depth=True, use_memory_cache=True):
    """
    Render one trajectory using the already-loaded Blender scene. Updates only camera, output paths, then renders and post-processes.
    Requires setup_blender_scene_for_rendering() to have been called once.

    When use_memory_cache=True (default): RGB is read from bpy.data.images['Render Result'] each frame (no per-frame jpg write).
    Mask and depth EXR are written to a temp dir in /dev/shm (or output_dir if no ramdisk). After all frames, masks and depths
    are packed into a single npz file and the temp dir is removed. This avoids heavy per-frame disk I/O.
    """
    scene = bpy.context.scene
    camera = bpy.data.objects.get("Camera")
    if camera is None:
        raise RuntimeError("Camera not found; call setup_blender_scene_for_rendering() first.")
    tree = scene.node_tree
    mask_output_node = tree.nodes.get("MaskOutput")
    depth_output_node = tree.nodes.get("DepthOutput") if render_depth else None
    if mask_output_node is None:
        raise RuntimeError("MaskOutput node not found; call setup_blender_scene_for_rendering() first.")

    camera.data.angle = np.radians(fov)

    output_dir = os.path.abspath(os.path.dirname(output_path))
    output_basename = os.path.splitext(os.path.basename(output_path))[0]

    if use_memory_cache:
        ramdisk = _get_ramdisk_base()
        cache_base = os.path.join(ramdisk or output_dir, f"blender_render_{os.getpid()}_{output_basename}")
        masks_dir = os.path.join(cache_base, "masks")
        depth_dir = os.path.join(cache_base, "depth") if render_depth and depth_output_node else None
        rgb_temp_dir = os.path.join(cache_base, "rgb")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(rgb_temp_dir, exist_ok=True)
        if depth_dir:
            os.makedirs(depth_dir, exist_ok=True)
        mask_output_node.base_path = masks_dir + "/"
        if depth_output_node is not None:
            depth_output_node.base_path = (depth_dir or "") + "/"
        # In Blender background mode, Viewer Node / Render Result are not updated. Write each frame to a
        # temp file in ramdisk, read into memory, then delete so we get correct RGB without hitting disk.
        frames_list = []
        res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
        rgb_temp_path = os.path.join(rgb_temp_dir, "frame.jpg")
        print(f"Rendering {len(trajectory)} frames (RGB via ramdisk temp, masks/depth in memory)...")
        for i, pose in enumerate(tqdm(trajectory, desc="Rendering frames")):
            scene.frame_set(i)
            camera_pos = pose['position']
            lookat_pos = pose['target']
            setup_camera_look_at(camera, camera_pos, lookat_pos)
            scene.render.filepath = rgb_temp_path
            with suppress_output():
                bpy.ops.render.render(write_still=True)
            if os.path.exists(rgb_temp_path):
                frame = _read_frame_from_file(rgb_temp_path)
                try:
                    os.remove(rgb_temp_path)
                except OSError:
                    pass
                if frame.shape[0] != res_y or frame.shape[1] != res_x:
                    frame = np.zeros((res_y, res_x, 3), dtype=np.uint8, order='C')
            else:
                frame = np.zeros((res_y, res_x, 3), dtype=np.uint8, order='C')
            frames_list.append(np.ascontiguousarray(frame.astype(np.uint8)))

        print(f"Saving video to {output_path}...")
        if not frames_list:
            print("Warning: No frames to write; skipping video.")
        else:
            with imageio.get_writer(
                output_path, fps=fps, format="FFMPEG", codec="libx264",
                pixelformat="yuv420p"
            ) as writer:
                for frame in frames_list:
                    writer.append_data(frame)

        print("Packing masks, depths, and frames into single npz...")
        masks_arr = _process_mask_exr_to_arrays(room, masks_dir)
        depths_arr = _process_depth_exr_to_arrays(depth_dir) if render_depth and depth_dir else None
        npz_path = os.path.join(output_dir, f"{output_basename}_data.npz")
        save_dict = {}
        if frames_list:
            save_dict["frames"] = np.stack(frames_list, axis=0)
        if masks_arr is not None:
            save_dict["masks"] = masks_arr
        if depths_arr is not None:
            save_dict["depths"] = depths_arr
        if save_dict:
            np.savez_compressed(npz_path, **save_dict)
            print(f"Saved {npz_path} with keys: {list(save_dict.keys())}")

        for d in [masks_dir, depth_dir, rgb_temp_dir]:
            if d is not None and os.path.exists(d):
                for f in os.listdir(d):
                    try:
                        os.remove(os.path.join(d, f))
                    except OSError:
                        pass
        try:
            if os.path.exists(cache_base):
                if os.path.exists(masks_dir):
                    os.rmdir(masks_dir)
                if depth_dir is not None and os.path.exists(depth_dir):
                    os.rmdir(depth_dir)
                if os.path.exists(rgb_temp_dir):
                    os.rmdir(rgb_temp_dir)
                os.rmdir(cache_base)
        except OSError:
            pass
    else:
        frames_dir = os.path.join(output_dir, f"{output_basename}_frames")
        masks_dir = os.path.join(output_dir, f"{output_basename}_masks")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        mask_output_node.base_path = masks_dir + "/"
        if render_depth and depth_output_node is not None:
            depth_dir = os.path.join(output_dir, f"{output_basename}_depth")
            os.makedirs(depth_dir, exist_ok=True)
            depth_output_node.base_path = depth_dir + "/"
        else:
            depth_dir = None

        frame_paths = []
        print(f"Rendering {len(trajectory)} frames...")
        for i, pose in enumerate(tqdm(trajectory, desc="Rendering frames")):
            scene.frame_set(i)
            camera_pos = pose['position']
            lookat_pos = pose['target']
            setup_camera_look_at(camera, camera_pos, lookat_pos)
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            scene.render.filepath = frame_path
            with suppress_output():
                bpy.ops.render.render(write_still=True)
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)

        print(f"Saving video to {output_path}...")
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in frame_paths:
                frame_img = imageio.imread(frame_path)
                writer.append_data(frame_img)

        _process_mask_exr_files(room, masks_dir)
        if render_depth and depth_dir and os.path.exists(depth_dir):
            _process_depth_exr_files(depth_dir)
    print("Done!")


def _process_mask_exr_files(room, masks_dir):
    """Convert mask EXR files in masks_dir to int16 npz; remove EXRs. Shared by full and trajectory-only render."""
    if not os.path.exists(masks_dir):
        return
    if args.engine == 'CYCLES':
        print("Processing mask EXR files using fast foreach_get...")
        exr_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.exr')])
        for exr_fname in tqdm(exr_files, desc="Converting masks to int16 npz"):
            exr_path = os.path.join(masks_dir, exr_fname)
            img = bpy.data.images.load(exr_path)
            width, height = img.size
            pixels = np.empty(width * height * 4, dtype=np.float32)
            img.pixels.foreach_get(pixels)
            pixels = pixels.reshape((height, width, 4))
            mask = pixels[:, :, 0]
            mask = np.flipud(mask)
            frame_num = int(exr_fname.split('_')[-1].split('.')[0])
            npz_path = os.path.join(masks_dir, f"mask_{frame_num:04d}.npz")
            np.savez_compressed(npz_path, mask.astype(np.int16))
            bpy.data.images.remove(img)
            os.remove(exr_path)
    else:
        print("Processing mask EXR files (Eevee Cryptomatte)...")
        exr_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.exr')])
        hash_to_pass_index = {0: 0}
        if exr_files:
            first_path = os.path.join(masks_dir, exr_files[0])
            try:
                first_exr = OpenEXR.InputFile(first_path)
                header = first_exr.header()
                manifest_name_to_hash = _cryptomatte_manifest_from_exr_header(header)
                first_exr.close()
                hash_to_pass_index = _build_cryptomatte_hash_to_pass_index(room, manifest_name_to_hash)
                if manifest_name_to_hash:
                    print("Using Cryptomatte manifest from EXR for hash -> pass index.")
                else:
                    print("No manifest in EXR; using MurmurHash3 of object names for hash -> pass index.")
            except Exception as e:
                print(f"Could not read manifest from EXR ({e}); using MurmurHash3 of object names.")
                hash_to_pass_index = _build_cryptomatte_hash_to_pass_index(room, None)
        else:
            hash_to_pass_index = _build_cryptomatte_hash_to_pass_index(room, None)
        for exr_fname in tqdm(exr_files, desc="Converting masks to int16 npz"):
            exr_path = os.path.join(masks_dir, exr_fname)
            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            channels = header['channels'].keys()
            if 'CryptoObject00.R' in channels:
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                raw_bytes = exr_file.channel('CryptoObject00.R', pt)
                mask_float = np.frombuffer(raw_bytes, dtype=np.float32)
                mask_uint32 = mask_float.view(np.uint32).reshape((height, width))
                mask = np.zeros_like(mask_uint32, dtype=np.int16)
                for h in np.unique(mask_uint32):
                    mask[mask_uint32 == h] = hash_to_pass_index.get(h, 0)
            else:
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                raw_bytes = exr_file.channel('R', pt)
                mask_float = np.frombuffer(raw_bytes, dtype=np.float32).reshape((height, width))
                mask = mask_float.astype(np.int16)
            frame_num = int(exr_fname.split('_')[-1].split('.')[0])
            npz_path = os.path.join(masks_dir, f"mask_{frame_num:04d}.npz")
            np.savez_compressed(npz_path, mask)
            exr_file.close()
            os.remove(exr_path)


def _process_depth_exr_files(depth_dir):
    """Convert depth EXR files to float16 npz; remove EXRs. Shared by full and trajectory-only render."""
    print("Processing depth EXR files using fast foreach_get...")
    exr_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.exr')])
    for exr_fname in tqdm(exr_files, desc="Reading depth EXR files"):
        exr_path = os.path.join(depth_dir, exr_fname)
        img = bpy.data.images.load(exr_path)
        width, height = img.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        img.pixels.foreach_get(pixels)
        pixels = pixels.reshape((height, width, 4))
        depth = pixels[:, :, 0]
        depth = np.flipud(depth)
        frame_num = int(exr_fname.split('_')[-1].split('.')[0])
        npz_path = os.path.join(depth_dir, f"depth_{frame_num:04d}.npz")
        np.savez_compressed(npz_path, depth.astype(np.float16))
        bpy.data.images.remove(img)
        os.remove(exr_path)


def render_trajectory_video(room, mesh_info_dict, trajectory, output_path, fov=30.0, resolution=512, fps=30, render_depth=True):
    """
    Full setup + render (load meshes, configure engine/compositor, render trajectory, post-process).
    Use this for a single trajectory. For multiple samples, use setup_blender_scene_for_rendering() once
    then render_trajectory_only() per sample.
    """
    output_dir = os.path.abspath(os.path.dirname(output_path))
    setup_blender_scene_for_rendering(room, mesh_info_dict, output_dir, resolution, render_depth)
    render_trajectory_only(room, trajectory, output_path, fov=fov, resolution=resolution, fps=fps, render_depth=render_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate camera trajectory for a room")
    parser.add_argument("layout_dir", type=str, help="Layout ID to visualize")
    parser.add_argument("--room_id", type=str, help="Room ID (optional, defaults to first room)")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames")
    parser.add_argument("--complexity", type=int, default=25, help="Number of anchors")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--render", action="store_true", help="Render video using Blender")
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"], help="Render engine")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution width (if rendering)")
    parser.add_argument("--fps", type=int, default=30, help="FPS (if rendering)")
    parser.add_argument("--no_memory_cache", action="store_true", help="Disable memory cache: write each frame to disk (legacy behavior)")

    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    try:
        # layout_dir = os.path.dirname(args.layout_path)
        layout_dir = args.layout_dir
        scene_id = os.path.basename(layout_dir)
        layout_id = scene_id[-len("layout_xxxxxxxx"):]
        json_path = os.path.join(layout_dir, f"{layout_id}.json")

        if not os.path.exists(json_path):
             print(f"Error: Layout file not found at {json_path}")
             sys.exit(1)

        print(f"Loading layout from {json_path}...")
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        layout = dict_to_floor_plan(layout_data)

        if args.room_id:
            room = next((r for r in layout.rooms if r.id == args.room_id), None)
            if room is None:
                print(f"Error: Room {args.room_id} not found")
                sys.exit(1)
        else:
            room = layout.rooms[0]
            print(f"Selected room: {room.id}")

        bounds = [
            room.position.x, room.position.y, room.position.z,
            room.position.x + room.dimensions.width,
            room.position.y + room.dimensions.length,
            room.position.z + room.dimensions.height
        ]

        print("Extracting meshes...")
        all_meshes, interest_meshes, mesh_info_dict = get_room_meshes(layout, layout_dir)

        print("Building environment...")
        env = CameraPlannerEnv(bounds, all_meshes, interest_meshes)

        # base_output, ext = os.path.splitext(args.output)
        # os.makedirs(os.path.dirname(args.output), exist_ok=True)
        # if ext == '': ext = '.json'

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        ext = '.json'

        # When rendering multiple samples: load scene and setup Blender once, then only change camera per sample
        if args.render and args.num_samples > 0:
            setup_blender_scene_for_rendering(
                room, mesh_info_dict, output_dir,
                resolution=args.resolution,
                render_depth=True
            )

        for sample_idx in range(args.num_samples):
            print(f"\n--- Generating sample {sample_idx} / {args.num_samples} ---")
            trajectory = generate_camera_trajectory(
                bounds,
                all_meshes,
                num_frames=args.frames,
                complexity=args.complexity,
                env=env,
                room=room,
                mesh_dict=mesh_info_dict
            )

            # Compute camera intrinsic matrix K
            args.fov = np.random.uniform(50.0, 80.0)
            fov_y = np.radians(args.fov)  # Vertical FOV from camera settings
            width = args.resolution
            height = width  # Aspect ratio from render settings

            fy = height / (2 * np.tan(fov_y / 2))
            fx = fy  # Square pixels
            cx = width / 2.0
            cy = height / 2.0

            K = [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ]

            frames_data = []
            for pose in trajectory:
                # up vector is column 1 of rotation matrix (R_mat = [right, up, -forward])
                up_vector = pose['rotation'][:, 1]

                frames_data.append({
                    'eye': pose['position'].tolist(),
                    'lookat': pose['target'].tolist(),
                    'up': up_vector.tolist()
                })

            output_data = {
                'K': K,
                'width': width,
                'height': height,
                'fov_y_deg': args.fov,
                'frames': frames_data
            }

            current_output_path = os.path.join(output_dir, f"{sample_idx}{ext}")

            with open(current_output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Trajectory saved to {current_output_path} with {len(frames_data)} frames")

            if args.render:
                print(f"Rendering video for sample {sample_idx}...")
                video_output_path = current_output_path.replace('.json', '.mp4')
                if video_output_path == current_output_path:
                    video_output_path = current_output_path + '.mp4'
                # Scene already loaded and configured; only update camera trajectory and output paths
                render_trajectory_only(
                    room,
                    trajectory,
                    video_output_path,
                    fov=args.fov,
                    resolution=args.resolution,
                    fps=args.fps,
                    render_depth=True,
                    use_memory_cache=not getattr(args, "no_memory_cache", False)
                )

        exe_log_path = os.path.join(output_dir, "exe_log.json")
        with open(exe_log_path, 'w') as f:
            json.dump({
                'scene_id': scene_id,
                'layout_id': layout_id,
                "num_samples": args.num_samples
            }, f, indent=4)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
