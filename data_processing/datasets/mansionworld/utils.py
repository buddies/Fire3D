# Full code to convert the pkl.gz data to an .obj file
import json
import gzip
import pickle
import numpy as np
import os
import difflib
import trimesh
from scipy.spatial.transform import Rotation as R
import colorsys
from PIL import Image
import xatlas
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from shapely.affinity import scale
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = Path(
    os.environ.get("FIRE3D_TRAINING_ROOT", REPO_ROOT / "data/training_scenes")
)
MANSIONWORLD_ROOT = os.environ.get(
    "MANSIONWORLD_ROOT",
    os.environ.get("FIRE3D_MANSIONWORLD_ROOT", str(TRAINING_ROOT / "MansionWorld")),
)
OBJATHOR_VERSION = os.environ.get("OBJATHOR_VERSION", "2023_09_23")
OBJATHOR_ROOT = os.environ.get(
    "OBJATHOR_ROOT",
    os.path.join(MANSIONWORLD_ROOT, "objathor"),
)
OBJATHOR_ASSET_DIR = os.environ.get(
    "OBJATHOR_ASSET_DIR",
    os.path.join(OBJATHOR_ROOT, OBJATHOR_VERSION, "assets"),
)
AI2THOR_ASSET_DIR = os.environ.get(
    "AI2THOR_ASSET_DIR",
    os.environ.get(
        "FIRE3D_AI2THOR_ASSET_DIR",
        str(TRAINING_ROOT / "ProcTHOR/ai2thor-hab/assets"),
    ),
)
MANSION_PATCH_ASSET_DIR = os.environ.get(
    "MANSION_PATCH_ASSET_DIR",
    os.path.join(MANSIONWORLD_ROOT, "mansion_patch", "asset", "objathor_assets"),
)

MATERIAL_DIR = os.environ.get(
    "MATERIAL_DIR",
    os.path.join(OBJATHOR_ROOT, "holodeck", OBJATHOR_VERSION, "materials", "images"),
)

SCENE_DATA_DIR = os.environ.get(
    "SCENE_DATA_DIR",
    os.path.join(MANSIONWORLD_ROOT, "mansionworld"),
)

def random_cool_color():
    h = np.random.random()
    s = np.random.uniform(0.3, 0.7)
    l = np.random.uniform(0.7, 0.9)

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([r, g, b], dtype=np.float32)

all_texture_names = (
    [os.path.splitext(fname)[0] for fname in sorted(os.listdir(MATERIAL_DIR)) if not fname.startswith("._")]
    if os.path.isdir(MATERIAL_DIR)
    else []
)
def normalize_name(name):
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")
normalized_to_name = {normalize_name(name): name for name in all_texture_names}
def find_texture_path(material_name):
    # search all_texture_names for the material_name, if not found, return the closest match
    if not all_texture_names:
        raise FileNotFoundError(f"No textures found in {MATERIAL_DIR}")

    target = str(material_name)
    if target in all_texture_names:
        return os.path.join(MATERIAL_DIR, target + ".png")

    normalized_target = normalize_name(target)

    if normalized_target in normalized_to_name:
        best_name = normalized_to_name[normalized_target]
        return os.path.join(MATERIAL_DIR, best_name + ".png")

    normalized_names = list(normalized_to_name.keys())
    close_matches = difflib.get_close_matches(normalized_target, normalized_names, n=1, cutoff=0.0)
    if close_matches:
        best_name = normalized_to_name[close_matches[0]]
        return os.path.join(MATERIAL_DIR, best_name + ".png")

    # Very defensive fallback: return the first texture by sorted order.
    return os.path.join(MATERIAL_DIR, all_texture_names[0] + ".png")

def get_texture(material_name):
    albedo_path = find_texture_path(material_name)
    albedo = Image.open(albedo_path).convert("RGBA")
    return albedo


def normalize_mesh(mesh: trimesh.Trimesh):
    """
    Preprocess the input mesh.
    """
    vertices = mesh.vertices
    vertices_min = vertices.min(axis=0)
    vertices_max = vertices.max(axis=0)
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()
    # vertices = (vertices - center) * scale

    mesh_transform = (center, scale)
    # assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
    return mesh_transform


def generate_uvs(vertices, faces):
    # Create a mesh in xatlas format
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    new_vertices = vertices[vmapping]
    return new_vertices, indices, uvs

def load_pkl_gz(file_path):
    """Load a .pkl.gz file."""
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_vertices(vertices_data):
    """Extract vertices into a NumPy array from the given data format."""
    return np.array([[v['x'], v['y'], v['z']] for v in vertices_data])

def extract_vts(vts_data):
    """Extract vertices into a NumPy array from the given data format."""
    return np.array([[v['x'], v['y']] for v in vts_data])

def create_faces(triangles_data):
    """Create faces (triangles) from the given indices."""
    return triangles_data.reshape(-1, 3)

def convert_pkl_gz_to_mesh(input_file_path):
    """Convert a .pkl.gz file to an .obj file."""
    # Load the .pkl.gz file
    data = load_pkl_gz(input_file_path)

    # Extracting vertices and triangles (faces) from the data
    vertices = np.array(data['vertices'])
    triangles = np.array(data['triangles'])
    uvs = np.array(data['uvs'])

    # Process the data
    vertices = extract_vertices(vertices)
    triangles = create_faces(triangles)
    uvs = extract_vts(uvs)

    rotation_matrix = R.from_euler('y', data['yRotOffset'], degrees=True).as_matrix()[:3, :3]
    vertices = vertices @ rotation_matrix.T

    input_file_dir = os.path.dirname(input_file_path)

    albedo_name = os.path.basename(data['albedoTexturePath'])
    albedo_path = os.path.join(input_file_dir, albedo_name)
    emission_name = os.path.basename(data['emissionTexturePath'])
    emission_path = os.path.join(input_file_dir, emission_name)
    normal_name = os.path.basename(data['normalTexturePath'])
    normal_path = os.path.join(input_file_dir, normal_name)

    if os.path.exists(albedo_path):
        albedo = Image.open(albedo_path).convert("RGBA")
    else:
        albedo = None
    if os.path.exists(emission_path):
        emission = Image.open(emission_path).convert("RGB")
    else:
        emission = None
    if os.path.exists(normal_path):
        normal = Image.open(normal_path).convert("RGB")
    else:
        normal = None

    # Build a complete PBR material from available textures.
    # If albedo is missing, use a white 1x1 fallback so exporters still work.
    if albedo is None:
        base_color_texture = Image.fromarray(
            np.array([[[255, 255, 255, 255]]], dtype=np.uint8), mode="RGBA"
        )
        alpha_mode = "OPAQUE"
        texture_height, texture_width = 1, 1
    else:
        base_color_texture = albedo
        albedo_np = np.array(albedo)
        texture_height, texture_width = albedo_np.shape[:2]
        alpha_mode = "BLEND" if np.any(albedo_np[..., 3] < 255) else "OPAQUE"

    # Standard PBR packs metallic in B and roughness in G.
    metallic = np.zeros((texture_height, texture_width, 1), dtype=np.uint8)
    roughness = np.full((texture_height, texture_width, 1), 255, dtype=np.uint8)
    zero_r = np.zeros_like(metallic)
    metallic_roughness_texture = Image.fromarray(
        np.concatenate([zero_r, roughness, metallic], axis=-1),
        mode="RGB",
    )

    material_kwargs = dict(
        baseColorTexture=base_color_texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=metallic_roughness_texture,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )
    if emission is not None:
        material_kwargs["emissiveTexture"] = emission
        material_kwargs["emissiveFactor"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if normal is not None:
        material_kwargs["normalTexture"] = normal

    material = trimesh.visual.material.PBRMaterial(**material_kwargs)


    textured_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=triangles,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
    )

    return textured_mesh

def get_asset_mesh(asset_id):

    # 1. find asset in objathor
    asset_path = os.path.join(OBJATHOR_ASSET_DIR, asset_id, asset_id + ".pkl.gz")
    if os.path.exists(asset_path):
        asset_mesh = convert_pkl_gz_to_mesh(asset_path)
        return asset_mesh

    # 2. find asset in ai2thor
    asset_path = os.path.join(AI2THOR_ASSET_DIR, "objects", asset_id + ".glb")
    if os.path.exists(asset_path):
        asset_mesh = trimesh.load(asset_path)
        return asset_mesh

    # 3. find asset in mansion patch
    # asset_path = os.path.join(MANSION_PATCH_ASSET_DIR, asset_id, asset_id + ".obj")
    # if os.path.exists(asset_path) and 'elevator' not in asset_id:
    #     asset_mesh = trimesh.load(asset_path)
    #     return asset_mesh

    # print(f"Asset {asset_id} not retrieved from objathor or ai2thor")
    return None

def _make_room_polygon_2d(floor_verts):
    """
    Valid Shapely Polygon for the room footprint in the XZ plane (vertices as (x, z)).
    """
    pts = np.asarray(floor_verts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("floor polygon needs at least 3 vertices")
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise ValueError("empty floor polygon")
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    elif poly.geom_type != "Polygon":
        raise ValueError(f"unsupported floor geometry: {poly.geom_type}")
    return poly


def _floor_mesh_from_shapely_polygon(poly, thickness=0.01):
    """
    Build a thin floor slab from a Shapely polygon in the XZ plane (stored as x, y in extrusion).
    extrude_polygon: polygon in XY, extrusion +Z -> (vx, vy, vz) with vz in {0, height}.
    Scene floor: horizontal XZ, thickness along Y from -height to 0 (center -height/2).
    """
    floor_mesh = trimesh.creation.extrude_polygon(poly, height=thickness)
    R = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T = trimesh.transformations.translation_matrix([0.0, -thickness, 0.0])
    floor_mesh.apply_transform(T @ R)
    return trimesh.Trimesh(
        vertices=floor_mesh.vertices,
        faces=floor_mesh.faces,
        process=False,
    )


def _floor_mesh_from_polygon(floor_verts, thickness=0.01):
    """Build floor mesh from raw floor vertices (wrapper around _make_room_polygon_2d)."""
    return _floor_mesh_from_shapely_polygon(_make_room_polygon_2d(floor_verts), thickness)


def _resolve_wall_room_ids(wall_original_room_id, wall_center_xz, room_polygons, boundary_match_tol=0.05):
    """
    Return one or two room ids for a wall from 2D distance to each room footprint boundary.

    If two or more rooms have boundary distance <= boundary_match_tol, return the two closest
    (by distance, then room id) so the wall can be duplicated for shared boundaries.

    If exactly one room is within tolerance, return [that id]. If none are within tolerance,
    return the single room with nearest boundary (same as before).
    """
    if not room_polygons:
        return [wall_original_room_id]
    pt = Point(float(wall_center_xz[0]), float(wall_center_xz[1]))
    candidates = [
        (pt.distance(poly.boundary), rid)
        for rid, poly in room_polygons.items()
        if not poly.is_empty
    ]
    if not candidates:
        return [wall_original_room_id]
    candidates.sort(key=lambda x: (x[0], x[1]))
    within_tol = [(d, rid) for d, rid in candidates if d <= boundary_match_tol]
    if len(within_tol) >= 2:
        return [within_tol[0][1], within_tol[1][1]]
    if len(within_tol) == 1:
        return [within_tol[0][1]]
    return [candidates[0][1]]


def load_floor(floors, wall_height=None):
    floor_mesh_list = []
    room_polygons = {}
    for floor in floors:
        room_id = floor["id"]
        floor_verts = np.array(floor["vertices"]).reshape(-1, 2)
        thickness = 0.01
        try:
            poly = _make_room_polygon_2d(floor_verts)
            floor_mesh = _floor_mesh_from_shapely_polygon(poly, thickness=thickness)
            room_polygons[room_id] = poly
        except Exception:
            print(f"Error creating floor mesh for room {room_id}, using box instead")
            x_min, x_max = floor_verts[:, 0].min(), floor_verts[:, 0].max()
            z_min, z_max = floor_verts[:, 1].min(), floor_verts[:, 1].max()
            floor_mesh = trimesh.primitives.Box(
                extents=[x_max - x_min, thickness, z_max - z_min],
                transform=trimesh.transformations.translation_matrix(
                    [
                        x_min + (x_max - x_min) / 2,
                        -thickness / 2,
                        z_min + (z_max - z_min) / 2,
                    ]
                )
            )
            room_polygons[room_id] = box(x_min, z_min, x_max, z_max)

        vertices, indices, uvs = generate_uvs(floor_mesh.vertices, floor_mesh.faces)
        albedo = get_texture(floor['floorMaterial']['name'])

        floor_material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
            doubleSided=True,
        )
        floor_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=indices,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=floor_material),
        )
        if wall_height is not None:
            ceiling_mesh = floor_mesh.copy()
            ceiling_mesh.apply_transform(trimesh.transformations.translation_matrix([0, wall_height, 0]))
        rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        floor_mesh.apply_transform(rot_x_90)
        if wall_height is not None:
            ceiling_mesh.apply_transform(rot_x_90)
        floor_mesh_list.append({
            "name": room_id,
            "type": "floor",
            "mesh": floor_mesh,
            "room_id": room_id,
        })
        if wall_height is not None:
            floor_mesh_list.append({
                "name": room_id+"_ceiling",
                "type": "ceiling",
                "mesh": ceiling_mesh,
                "room_id": room_id,
            })
    return floor_mesh_list, room_polygons

def load_wall(walls, room_polygons=None, boundary_match_tol=0.05):
    wall_mesh_list = []
    if room_polygons is None:
        room_polygons = {}
    for wall in walls:
        segment = np.array(wall['segment']).reshape(-1, 2)
        wall_center_xz = segment.mean(axis=0)
        room_ids = _resolve_wall_room_ids(
            wall["roomId"], wall_center_xz, room_polygons, boundary_match_tol=boundary_match_tol
        )
        wall_id = wall["id"].replace("|", "_").replace(" ", "_")
        height = wall['height']
        x_min, x_max = segment[:, 0].min(), segment[:, 0].max()
        z_min, z_max = segment[:, 1].min(), segment[:, 1].max()
        extent_x = max(x_max - x_min, 0.001)
        extent_z = max(z_max - z_min, 0.001)
        wall_mesh = trimesh.primitives.Box(
            extents=[extent_x, height, extent_z],
            transform=trimesh.transformations.translation_matrix(
                [x_min + (x_max - x_min) / 2, height / 2, z_min + (z_max - z_min) / 2]
            )
        )


        vertices, indices, uvs = generate_uvs(wall_mesh.vertices, wall_mesh.faces)
        albedo = get_texture(wall['material']['name'])

        # uvs[:, 1] = 1 - uvs[:, 1]  # Flip UV V-coordinate
        wall_material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
            doubleSided=True,
        )
        wall_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=indices,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=wall_material),
        )
        rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        wall_mesh.apply_transform(rot_x_90)
        for i, room_id in enumerate(room_ids):
            mesh_out = wall_mesh.copy() if len(room_ids) > 1 else wall_mesh
            name = wall_id if len(room_ids) == 1 else f"{wall_id}__{room_id.replace('|', '_').replace(' ', '_')}"
            wall_mesh_list.append({
                "name": name,
                "type": "wall",
                "mesh": mesh_out,
                "room_id": room_id,
            })

    return wall_mesh_list

def load_floor_objects(floor_objects):
    floor_objects_list = []
    for floor_object in floor_objects:
        asset_id = floor_object["assetId"]
        room_id = floor_object["roomId"]
        position = floor_object["position"]
        rotation = floor_object["rotation"]
        object_name = floor_object["id"].replace(" ", "_").replace("-", "_")

        asset_mesh = get_asset_mesh(asset_id)
        if asset_mesh is None:
            continue

        orientation_matrix = np.array([
            [-1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

        asset_mesh.apply_transform(orientation_matrix)

        if isinstance(asset_mesh, trimesh.Scene):
            center, scale = normalize_mesh(trimesh.util.concatenate(asset_mesh.dump()))
        else:
            center, scale = normalize_mesh(asset_mesh)

        normalization_center_matrix = np.eye(4)
        normalization_center_matrix[:3, 3] = -center
        normalization_scale_matrix = np.eye(4)
        normalization_scale_matrix[np.diag_indices(3)] = scale
        asset_mesh.apply_transform(normalization_center_matrix)
        asset_mesh.apply_transform(normalization_scale_matrix)

        canonical_mesh = asset_mesh.copy()

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R.from_euler('xyz', (rotation['x'], rotation['y'], rotation['z']), degrees=True).as_matrix()[:3, :3]
        translation_matrix = trimesh.transformations.translation_matrix(np.array([position['x'], 0, position['z']]))
        rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

        added_transform = rot_x_90 @ translation_matrix @ rotation_matrix
        canonical_transform = normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix
        total_transform = added_transform @ np.linalg.inv(canonical_transform)

        final_mesh = canonical_mesh.copy().apply_transform(total_transform)

        floor_objects_list.append({
            "name": object_name,
            "type": "object",
            "mesh": final_mesh,
            "canonical_mesh": canonical_mesh,
            "transform": total_transform,
            "room_id": room_id,
            "asset_id": asset_id
        })

    return floor_objects_list


def load_wall_objects(wall_objects):
    wall_objects_list = []
    for wall_object in wall_objects:
        asset_id = wall_object["assetId"]
        room_id = wall_object["roomId"]
        position = wall_object["position"]
        rotation = wall_object["rotation"]
        object_name = wall_object["id"].replace(" ", "_").replace("-", "_")

        asset_mesh = get_asset_mesh(asset_id)
        if asset_mesh is None:
            continue

        orientation_matrix = np.array([
            [-1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

        asset_mesh.apply_transform(orientation_matrix)

        if isinstance(asset_mesh, trimesh.Scene):
            center, scale = normalize_mesh(trimesh.util.concatenate(asset_mesh.dump()))
        else:
            center, scale = normalize_mesh(asset_mesh)
        normalization_center_matrix = np.eye(4)
        normalization_center_matrix[:3, 3] = -center
        normalization_scale_matrix = np.eye(4)
        normalization_scale_matrix[np.diag_indices(3)] = scale
        asset_mesh.apply_transform(normalization_center_matrix)
        asset_mesh.apply_transform(normalization_scale_matrix)

        canonical_mesh = asset_mesh.copy()

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R.from_euler('xyz', (rotation['x'], rotation['y'], rotation['z']), degrees=True).as_matrix()[:3, :3]
        translation_matrix = trimesh.transformations.translation_matrix(np.array([position['x'], 0, position['z']]))
        rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

        added_transform = rot_x_90 @ translation_matrix @ rotation_matrix
        canonical_transform = normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix
        total_transform = added_transform @ np.linalg.inv(canonical_transform)

        final_mesh = canonical_mesh.copy().apply_transform(total_transform)

        wall_objects_list.append({
            "name": object_name,
            "type": "object",
            "mesh": final_mesh,
            "canonical_mesh": canonical_mesh,
            "transform": total_transform,
            "room_id": room_id,
            "asset_id": asset_id
        })
    return wall_objects_list


def load_small_objects(small_objects):
    small_objects_list = []
    for small_object in small_objects:
        asset_id = small_object["assetId"]
        room_id = small_object["roomId"]
        position = small_object["position"]
        rotation = small_object["rotation"]
        object_name = small_object["id"].replace(" ", "_").replace("-", "_")

        asset_mesh = get_asset_mesh(asset_id)
        if asset_mesh is None:
            continue

        orientation_matrix = np.array([
            [-1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1]
        ])

        asset_mesh.apply_transform(orientation_matrix)

        if isinstance(asset_mesh, trimesh.Scene):
            center, scale = normalize_mesh(trimesh.util.concatenate(asset_mesh.dump()))
        else:
            center, scale = normalize_mesh(asset_mesh)
        normalization_center_matrix = np.eye(4)
        normalization_center_matrix[:3, 3] = -center
        normalization_scale_matrix = np.eye(4)
        normalization_scale_matrix[np.diag_indices(3)] = scale
        asset_mesh.apply_transform(normalization_center_matrix)
        asset_mesh.apply_transform(normalization_scale_matrix)

        canonical_mesh = asset_mesh.copy()

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R.from_euler('xyz', (rotation['x'], rotation['y'], rotation['z']), degrees=True).as_matrix()[:3, :3]
        translation_matrix = trimesh.transformations.translation_matrix(np.array([position['x'], 0, position['z']]))
        rot_x_90 = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

        added_transform = rot_x_90 @ translation_matrix @ rotation_matrix
        canonical_transform = normalization_scale_matrix @ normalization_center_matrix @ orientation_matrix
        total_transform = added_transform @ np.linalg.inv(canonical_transform)

        final_mesh = canonical_mesh.copy().apply_transform(total_transform)

        small_objects_list.append({
            "name": object_name,
            "type": "object",
            "mesh": final_mesh,
            "canonical_mesh": canonical_mesh,
            "transform": total_transform,
            "room_id": room_id,
            "asset_id": asset_id
        })

    return small_objects_list

def save_glb(mesh_or_scene, save_path):
    rot_x_90 = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
    mesh_or_scene_copy = mesh_or_scene.copy()
    mesh_or_scene_copy.apply_transform(rot_x_90)
    trimesh.exchange.export.export_mesh(mesh_or_scene_copy, save_path)


def get_open_room_clusters(open_room_pairs):
    """
    Group room ids into connected clusters from open-room pair links.
    """
    adjacency = {}

    for pair in open_room_pairs:
        if not pair:
            continue
        if len(pair) == 1:
            adjacency.setdefault(pair[0], set())
            continue
        room_a, room_b = pair[0], pair[1]
        adjacency.setdefault(room_a, set()).add(room_b)
        adjacency.setdefault(room_b, set()).add(room_a)

    visited = set()
    clusters = []
    for room_id in adjacency:
        if room_id in visited:
            continue
        stack = [room_id]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        clusters.append(sorted(component))

    # Deterministic output: sort room ids within each cluster, then sort clusters.
    clusters.sort(key=lambda cluster: (cluster[0], len(cluster), "_".join(cluster)))
    return clusters


def _normalize_mesh_room_id(room_id):
    """Strip first path segment; matches room_id handling on mesh_dict in get_room_geoms."""
    if room_id.startswith("F"):
        return "_".join(str(room_id).split("_")[1:])
    return room_id


def _cluster_merge_epsilon(polys):
    """Scale-aware epsilon to bridge tiny gaps between adjacent floor footprints after union."""
    if not polys:
        return 1e-4
    minx = min(p.bounds[0] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    extent = max(maxx - minx, maxy - miny)
    return max(1e-6, min(0.02, 1e-4 * extent))


def _coerce_to_single_polygon(geom):
    """
    Return a single shapely Polygon.

    unary_union of touching rooms is usually already one Polygon. MultiPolygon appears when
    parts are disjoint or separated by floating-point gaps; we try a small buffer merge,
    then convex_hull as a last resort so callers always get one Polygon.
    """
    if geom is None or geom.is_empty:
        return geom
    gt = geom.geom_type
    if gt == "Polygon":
        return geom
    if gt == "MultiPolygon":
        polys = list(geom.geoms)
        eps = _cluster_merge_epsilon(polys)
        merged = unary_union([g.buffer(eps) for g in polys]).buffer(-eps)
        if merged.geom_type == "Polygon" and not merged.is_empty:
            return merged
        if merged.geom_type == "MultiPolygon":
            return merged.convex_hull
        if merged.is_empty:
            return geom.convex_hull
        return _coerce_to_single_polygon(merged)
    if gt == "GeometryCollection":
        polys = [g for g in geom.geoms if getattr(g, "geom_type", None) == "Polygon"]
        if not polys:
            return geom
        if len(polys) == 1:
            return polys[0]
        return _coerce_to_single_polygon(unary_union(polys))
    return geom


def _union_cluster_polygons_to_polygon(polys):
    """Union cluster footprints into one Polygon."""
    if not polys:
        raise ValueError("empty polygon list for cluster")
    if len(polys) == 1:
        return _coerce_to_single_polygon(polys[0])
    return _coerce_to_single_polygon(unary_union(polys))


def merge_room_polygons_with_open_clusters(room_polygons, open_room_clusters):
    """
    Build one shapely Polygon per logical room: union floor footprints for rooms in the same
    open-room cluster (always coerced to a single Polygon); other rooms keep their own
    footprint, also coerced to one Polygon. Keys match open_room_cluster names
    ('_'.join(cluster)) or standalone normalized room ids.
    """
    norm = {}
    for rid, poly in room_polygons.items():
        nrid = _normalize_mesh_room_id(rid)
        norm[nrid] = poly

    clustered_ids = set()
    for cluster in open_room_clusters:
        for r in cluster:
            clustered_ids.add(r)

    merged = {}
    for cluster in open_room_clusters:
        cluster_norm = [r for r in cluster]
        polys = [norm[r] for r in cluster_norm if r in norm]
        if not polys:
            continue
        name = "_".join(cluster)
        merged[name] = _union_cluster_polygons_to_polygon(polys)

    for nrid, poly in norm.items():
        if nrid not in clustered_ids:
            merged[nrid] = _coerce_to_single_polygon(poly)

    return merged


def flip_room_polygon_y(geom, origin=(0.0, 0.0)):
    """
    Reflect the 2D footprint across the x axis (negate the second coordinate).
    Aligns polygon coordinates with conventions where the vertical axis is flipped (e.g. image vs scene z).
    """
    if geom is None or geom.is_empty:
        return geom
    return scale(geom, xfact=1.0, yfact=-1.0, origin=origin)


def room_polygon_to_xy_array(geom):
    """
    Exterior ring as (N, 2) float64 array for matplotlib Path.contains_point.
    Returns None if geom is None, empty, or not a polygonal geometry.
    """
    if geom is None:
        return None
    if not hasattr(geom, "geom_type") or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return np.asarray(geom.exterior.coords, dtype=np.float64)
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda g: g.area)
        return np.asarray(largest.exterior.coords, dtype=np.float64)
    return None


def get_room_geoms(scene_json_path):
    scene_dict = json.load(open(scene_json_path, "r"))

    wall_height = scene_dict["wall_height"]

    floor_mesh_list, room_polygons = load_floor(scene_dict["rooms"], wall_height=wall_height)
    wall_mesh_list = load_wall(scene_dict["walls"], room_polygons=room_polygons)
    floor_objects_mesh_list = load_floor_objects(scene_dict["floor_objects"])
    wall_objects_mesh_list = load_wall_objects(scene_dict["wall_objects"])
    small_objects_mesh_list = load_small_objects(scene_dict["small_objects"])

    all_mesh_list = floor_mesh_list + wall_mesh_list + floor_objects_mesh_list + wall_objects_mesh_list + small_objects_mesh_list

    room_objects_geoms = {}
    room_bg_geoms = {}
    room_num_objects = {}
    open_room_pairs = scene_dict["open_room_pairs"]
    open_room_clusters = get_open_room_clusters(open_room_pairs)
    open_room_cluster_names = ["_".join(cluster) for cluster in open_room_clusters]

    for mesh_dict in all_mesh_list:
        room_id = mesh_dict["room_id"]
        room_id = _normalize_mesh_room_id(room_id)
        for open_room_cluster_name, open_room_cluster in zip(open_room_cluster_names, open_room_clusters):
            if room_id in open_room_cluster:
                room_id = open_room_cluster_name
                break
        if mesh_dict["type"] == "object":
            if room_id not in room_objects_geoms:
                room_objects_geoms[room_id] = []
                room_num_objects[room_id] = 0
            room_num_objects[room_id] += 1
            room_objects_geoms[room_id].append(mesh_dict)
        else:
            if room_id not in room_bg_geoms:
                room_bg_geoms[room_id] = []
            room_bg_geoms[room_id].append(mesh_dict)

    room_geoms = {}

    for room_id, mesh_list in room_objects_geoms.items():
        if room_num_objects[room_id] == 0:
            continue

        room_geoms[room_id] = {}

        # bg
        bg_scene = trimesh.Scene()
        for mesh_dict in room_bg_geoms[room_id]:
            bg_scene.add_geometry(mesh_dict["mesh"])

        center, scale = normalize_mesh(trimesh.util.concatenate(bg_scene.dump()))

        normalization_center_matrix = np.eye(4)
        normalization_center_matrix[:3, 3] = -center
        normalization_scale_matrix = np.eye(4)
        normalization_scale_matrix[np.diag_indices(3)] = scale

        bg_scene.apply_transform(normalization_center_matrix)
        bg_scene.apply_transform(normalization_scale_matrix)

        canonical_mesh = bg_scene.copy()
        canonical_transform = normalization_scale_matrix @ normalization_center_matrix
        total_transform = np.linalg.inv(canonical_transform)

        final_mesh = canonical_mesh.copy().apply_transform(total_transform)

        room_geoms[room_id]["bg"] = {
            "name": "bg",
            "type": "bg",
            "mesh": final_mesh,
            "canonical_mesh": canonical_mesh,
            "transform": total_transform,
            "room_id": room_id,
        }


        # objects
        for mesh_dict in mesh_list:
            mesh_name = mesh_dict["name"]
            room_geoms[room_id][mesh_name] = mesh_dict

    merged_room_polygons = merge_room_polygons_with_open_clusters(
        room_polygons, open_room_clusters
    )
    room_polygons_dict = {
        room_id: flip_room_polygon_y(merged_room_polygons[room_id])
        for room_id in room_geoms
    }

    return room_geoms, room_polygons_dict

def _iter_shapely_polygons(geom):
    """Yield shapely Polygon parts from Polygon, MultiPolygon, or GeometryCollection."""
    if geom is None or geom.is_empty:
        return
    gt = geom.geom_type
    if gt == "Polygon":
        yield geom
    elif gt == "MultiPolygon":
        for g in geom.geoms:
            yield from _iter_shapely_polygons(g)
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            yield from _iter_shapely_polygons(g)


def visualize_room_polygons(room_polygons_dict, save_png_path):
    """
    Draw each room footprint in a distinct color, label by room id, save as PNG.
    Coordinates follow the scene floor plane (x horizontal, z as vertical axis in the plot).
    """
    if not room_polygons_dict:
        return

    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    items = sorted(room_polygons_dict.items(), key=lambda x: x[0])
    n = len(items)

    for i, (room_id, geom) in enumerate(items):
        color = plt.cm.tab20(i % 20) if n else (0.5, 0.5, 0.5, 1.0)
        for poly in _iter_shapely_polygons(geom):
            if poly.is_empty:
                continue
            xy = np.asarray(poly.exterior.coords, dtype=float)
            if xy.shape[0] < 3:
                continue
            patch = MplPolygon(
                xy,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.55,
            )
            ax.add_patch(patch)
        if not geom.is_empty:
            rp = geom.representative_point()
            label = str(room_id)
            if len(label) > 32:
                label = label[:29] + "..."
            ax.text(
                rp.x,
                rp.y,
                label,
                fontsize=5,
                ha="center",
                va="center",
                color="0.15",
                clip_on=True,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.margins(0.06)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Room floor polygons")
    ax.grid(True, alpha=0.25)

    out_dir = os.path.dirname(save_png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_png_path, bbox_inches="tight")
    plt.close(fig)

def get_all_scene_paths():
    all_building_names = sorted(os.listdir(SCENE_DATA_DIR))
    all_scene_paths = []

    for building_name in all_building_names:
        building_dir = os.path.join(SCENE_DATA_DIR, building_name)
        all_floor_json_paths = sorted(glob.glob(os.path.join(building_dir, "floor_*.json")))
        for floor_json_path in all_floor_json_paths:
            all_scene_paths.append(floor_json_path)

    return all_scene_paths



if __name__ == "__main__":
    all_scene_paths = get_all_scene_paths()
    print(len(all_scene_paths))
    print(all_scene_paths[0])
