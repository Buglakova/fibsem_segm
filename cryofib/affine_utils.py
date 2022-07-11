from skimage.transform import AffineTransform
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import z5py
import re
from typing import Dict, List
from transforms3d.euler import euler2mat, mat2euler
from scipy.optimize import minimize


def dict_to_np(parsed_dict):
    """
    Dragonfly has numbering starting from 1 instead of 0, so -1
    For Mobie axis order is zyx
    """
    val = int(parsed_dict["val"])
    coords = [parsed_dict["z"], parsed_dict["y"], parsed_dict["x"]]
    coords = [int(coord) - 1 for coord in coords]
    return coords

def parse_points(dragonfly_output_path: Path, image_names: Dict):
    """
    Parse points saved from Dragonfly
    """
    coords = {}
    with open(dragonfly_output_path, "r") as f:
        text = f.read()
        # print(repr(text))
        for ds in image_names.keys():
            print(ds)
            ds_coords = []
            points = re.compile(r"Image: %s\n  Value: \d*\n  Voxel: x:\d* y:\d* z:\d*"%image_names[ds], re.MULTILINE)
            for point in points.findall(text):
                regexp_coords = re.compile(r"Value: (?P<val>\d*)\n  Voxel: x:(?P<x>\d*) y:(?P<y>\d*) z:(?P<z>\d*)")
                m = regexp_coords.search(point)
                print(m.groupdict())
                ds_coords.append(dict_to_np(m.groupdict()))
            ds_coords = np.array(ds_coords)
            coords[ds] = ds_coords
    return coords


def get_scaling_transform(resolution: List):
    """translation
    Get scaling matrix: 4x4 matrix with scaling factors on diagonal
    resolution: pixel sizes for each axis
    """
    resolution = list(resolution)
    return np.diag(resolution + [1])


def get_translation_transform(translation):
    M = np.identity(4)
    M[0:3, 3] = translation
    return M


def get_affine_transform(translation, rotation, rotation_order="szyx"):
    M = get_translation_transform(translation)
    M[0:3, 0:3] = euler2mat(*rotation, rotation_order)
    return M


def params_to_transform(x):
    """
    Convert optimization parameters vector to affine matrix
    """
    t_x, t_y, t_z = x[0], x[1], x[2]
    phi_x, phi_z = x[3], x[4]
    R = get_affine_transform([t_x, t_y, t_z], [phi_x, 0, phi_z])
    return R

def objective_function(x, ref_points, moving_points):
    """
    x = [t_x, t_y, t_z, phi_y, phi_z]
    """
    R = params_to_transform(x)
    new_points = np.dot(R, moving_points)[0:3, :]
    dist = np.sqrt(np.sum((new_points - ref_points[0:3, :])**2,axis=0))
    print(dist)
    
    return np.sum(dist)  


def get_scaling_transform(resolution: List):
    """
    Get scaling matrix: 4x4 matrix with scaling factors on diagonal
    resolution: pixel sizes for each axis
    """
    resolution = list(resolution)
    return np.diag(resolution + [1])


def native_to_bdv(matrix, resolution=None, invert=True):
    """ Convert affine matrix in native format to
    bdv transformation parameter vector.
    Bdv and elf expect the transformation in the opposite direction.
    So to be directly applied the transformation also needs to be inverted.
    (In addition to changing between the axis conventions.)
    The bdv transformations often also include the transformation from
    voxel space to physical space.
    Arguments:
        matrix [np.ndarray] - native affine transformation matrix
        resolution [listlike] - physical resolution of the data in bdv.
            If given, the transformation will be scaled to voxel sapec (default: None) before inverting!
        invert [bool] - invert the resulting affine matrix.
            This is necessary to apply the affine matrix directly in elf (default: True)
    Returns:
        Vector with transformation parameters
    """
    # TODO include scaling transformation from physical space to voxel space
    if resolution is not None:
        scaling_trafo = get_scaling_transform(resolution)
        matrix = matrix @ scaling_trafo

    if invert:
        matrix = np.linalg.inv(matrix)

    if matrix.shape[0] == 4:
        trafo = [matrix[2, 2], matrix[2, 1], matrix[2, 0], matrix[2, 3],
                 matrix[1, 2], matrix[1, 1], matrix[1, 0], matrix[1, 3],
                 matrix[0, 2], matrix[0, 1], matrix[0, 0], matrix[0, 3]]
    else:
        trafo = [matrix[1, 1], matrix[1, 0], matrix[1, 2],
                 matrix[0, 1], matrix[0, 0], matrix[0, 2]]
    return trafo


def affine_to_parameter_vector(matrix):
    trafo = [matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
             matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
             matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]]

    return trafo


