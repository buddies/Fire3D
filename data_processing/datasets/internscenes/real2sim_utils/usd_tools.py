import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Function to compute the bounding box of a USD prim
def compute_prim_bbox(prim: Usd.Prim) -> Gf.Range3d:
    imageable: UsdGeom.Imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default()
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return bound_range


def visiblePrims(prim):
    for child in prim.GetChildren():
        # print(child.GetPath())
        UsdGeom.Imageable(child).MakeVisible()
        visiblePrims(child)

def set_usd_prim_orientation(usd_path, prim_path: str = "/World", euler_xyz_deg: list = [90, 0, 0]):
    stage = Usd.Stage.Open(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    instance_prim = stage.GetPrimAtPath(prim_path)
    attributes = instance_prim.GetAttributes()
    for attr in attributes:
        attr_name = attr.GetName()
        # print(attr_name)
        if 'orient' in attr_name:
            euler_xyz_deg = [90, 0, 0]
            rotation = R.from_euler('xyz', euler_xyz_deg, degrees=True)
            new_quat = rotation.as_quat()
            attr.Set(Gf.Quatf(new_quat[3], new_quat[0], new_quat[1], new_quat[2]))
            print(attr.Get())
    stage.Save()