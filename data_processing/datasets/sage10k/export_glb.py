import asyncio
import json
import sys
import os
import argparse
import numpy as np
from PIL import Image

# Add the server directory to the Python path to import from layout.py
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, server_dir)

# from tex_utils import export_layout_to_mesh_dict_list
from tex_utils_local import export_layout_to_mesh_dict_list_v2
from glb_utils import (
    create_glb_scene,
    add_textured_mesh_to_glb_scene,
    save_glb_scene
)
from utils import (
    dict_to_floor_plan,
)

def export_glb(layout_file_path: str, output_dir: str):
    """Test loading layout from JSON file"""


    with open(layout_file_path, 'r') as f:
        layout_data = json.load(f)
    current_layout = dict_to_floor_plan(layout_data)

    export_glb_path = os.path.join(output_dir, os.path.basename(layout_file_path).replace(".json", ".glb"))
    mesh_dict_list = export_layout_to_mesh_dict_list_v2(current_layout, os.path.dirname(layout_file_path))
    scene = create_glb_scene()
    for mesh_id, mesh_data in mesh_dict_list.items():
        mesh_data_dict = {
            'vertices': mesh_data['mesh'].vertices,
            'faces': mesh_data['mesh'].faces,
            'vts': mesh_data['texture']['vts'],
            'fts': mesh_data['texture']['fts'],
            'texture_image': np.array(Image.open(mesh_data['texture']['texture_map_path'])),
            'metallic_factor': mesh_data['texture'].get('metallic_factor', 0.0),
            'roughness_factor': mesh_data['texture'].get('roughness_factor', 1.0)
        }
        add_textured_mesh_to_glb_scene(
            mesh_data_dict,
            scene,
            material_name=f"material_{mesh_id}",
            mesh_name=f"mesh_{mesh_id}",
            preserve_coordinate_system=True,
        )
    save_glb_scene(export_glb_path, scene)
    print(f"GLB exported to: {os.path.abspath(export_glb_path)}")



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Export layout to GLB file")
    parser.add_argument("layout_file_path", type=str, help="Layout file path")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    # Run the test
    export_glb(args.layout_file_path, args.output_dir)
