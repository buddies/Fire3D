
import os
import json
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

def merge_paths(path1, path2):
    """
    merge two Path3D objects, connect their vertices and entities.
    note: the point indices in entities need to be offset according to the number of vertices in the first path.
    """

    vertices = np.vstack([path1.vertices, path2.vertices])
    entities = []

    for ent in path1.entities:
        entities.append(ent.copy())
    offset = len(path1.vertices)

    for ent in path2.entities:
        new_ent = ent.copy()
        new_ent.points = ent.points + offset
        entities.append(new_ent)
    return trimesh.path.Path3D(entities=entities, vertices=vertices)

def create_line_segment(origin, direction, length, thickness):
    """
    create a line segment, use a thin cylinder to represent it.
    args:
      origin: the start point of the line segment [x, y, z]
      direction: the direction of the line segment (no need to normalize)
      length: the length of the line segment
      thickness: the radius of the line segment
    """

    line = trimesh.creation.cylinder(radius=thickness, height=length, sections=32)
    R = trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0])
    line.apply_transform(R)
    line.apply_translation([length / 2, 0, 0])
    target = np.array(direction)
    target = target / np.linalg.norm(target)
    source = np.array([1, 0, 0])
    R_align = trimesh.geometry.align_vectors(source, target)
    line.apply_transform(R_align)
    line.apply_translation(origin)
    return line

def create_bbox_with_arrow(bbox, transform = None, scale = None, use_degrees = False):
    """
    create a bbox with arrow
    """

    pos = np.array(bbox[:3])
    size = np.array(bbox[3:6])
    angles = np.array(bbox[6:9])

    if type(transform) != type(None):
        rotation = transform.copy()
        transform = np.eye(4)
        transform[:3, :3] = rotation[:3, :3]
        transform[:3, 3] = pos

    else:
        # generate transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R.from_euler('ZXY', angles, degrees = use_degrees).as_matrix()
        # set translation part
        transform[:3, 3] = pos


    if type(scale) != type(None):
        # scale
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale[0]
        scale_matrix[1, 1] = scale[1]
        scale_matrix[2, 2] = scale[2]
        transform = np.dot(transform, scale_matrix)

    # create a box with center at origin, size is size
    box = trimesh.path.creation.box_outline(extents=size) #trimesh.creation.box(extents=size)
    random_color = np.random.randint(0, 256, size=3).tolist()
    for ent in box.entities:
        ent.color = random_color
    # box.visual.face_colors = (255, 0, 0, 255)  # red


    # calculate the start and end point of the arrow (line segment):
    # without rotation, take the local coordinate point on the top of the box (sx/2, 0, sz/2)
    local_start = np.array([size[0] / 2, 0, size[2] / 2])
    front_dir = np.array([1, 0, 0])
    arrow_length =  0.25
    local_end = local_start + front_dir * arrow_length

    # construct arrow Path3D
    arrow_vertices = np.array([local_start, local_end])
    arrow_entity = trimesh.path.entities.Line(np.array([0, 1]))
    arrow_path = trimesh.path.Path3D(entities=[arrow_entity], vertices=arrow_vertices)
    for ent in box.entities:
        ent.color = random_color
    # merge the box and arrow into a Path3D object
    merged_path = merge_paths(box, arrow_path)

    # apply transformation to the box
    merged_path.apply_transform(transform)

    # rotate -90 degree around x axis, save the scene as Y-up, so the scene has the correct upward orientation
    rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    merged_path.apply_transform(rotation_matrix)

    return merged_path

def compose_bboxes_scene(bboxes, save_path=None, show_scene=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scene = trimesh.Scene()
    for i, bbox in enumerate(bboxes):
        bbox_model = create_bbox_with_arrow(bbox, use_degrees=False)
        scene.add_geometry(bbox_model,  geom_name=f"{i}-th bbox")

    if save_path is not None:
        scene.export(save_path)
        print(f"scene saved to {save_path}")

    if show_scene:
        scene.show()

if __name__ == '__main__':

    json_path = "./data/Layout_info/scannet/scene0000_00/layout.json"
    save_path = "./examples/bbox_scene.glb"
    instance_infos = json.load(open(json_path, "r"))
    bboxes = []
    for instance_info in instance_infos:
        bboxes.append(instance_info["bbox"])
    compose_bboxes_scene(bboxes, save_path=save_path, show_scene=False)
