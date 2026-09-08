import numpy as np
from scipy.spatial.transform import Rotation as R



def generate_intrinsic(width, height, hfov, vfov):
    intrinsic = np.eye(3)
    intrinsic[0][0] = width / (2 * (np.tan(np.deg2rad(hfov)/2)))
    intrinsic[1][1] = height / (2 * (np.tan(np.deg2rad(vfov)/2)))
    intrinsic[0][2] = width / 2
    intrinsic[1][2] = height / 2
    return intrinsic

def build_transformation_mat(translation, rotation):
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector.")
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(R.from_euler('xyz',rotation).as_matrix())
    else:
        raise RuntimeError(f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
                           f"(3,3) or Euler angles of shape (3,) or (3,1).")

    return mat