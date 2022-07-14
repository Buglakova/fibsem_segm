import os
import mobie
import z5py
import numpy as np
from typing import List
from pathlib import Path

MOBIE_PROJECT_FOLDER = "/scratch/buglakova/data/cryofib/registration_fluo/mobie_projects/fibsem-registration-affine-segmentation/data"
DATASET_NAME = "registration"

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
            If given, the transformation will be scaled to voxel sapec (default: None)
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

def add_from_n5(input_file, input_key, source_name, menu_name, transform=None):
    target = "local"
    max_jobs = 16

    f = z5py.File(input_file, "r")
    ds = f[input_key]

    unit = ds.attrs["unit"]
    resolution = ds.attrs["resolution"]
    print()
    print()
    print()
    print("Adding data from", input_file, "with:")
    print("unit:", unit)
    print("resolution:", resolution)
    print()
    print()
    print()
    chunks = (1, 512, 512)
    scale_factors = 4 * [[2, 2, 2]]

    mobie.add_image(
        input_path=input_file,
        input_key=input_key,
        root=MOBIE_PROJECT_FOLDER,
        dataset_name=DATASET_NAME,
        image_name=source_name,
        menu_name=menu_name,
        resolution=resolution,
        chunks=chunks,
        scale_factors=scale_factors,
        transformation=transform,
        target=target,
        max_jobs=max_jobs,
        unit=unit,
    )


def add_segmentation_from_n5(input_file, input_key, source_name, menu_name):
    target = "local"
    max_jobs = 16

    f = z5py.File(input_file, "r")
    ds = f[input_key]

    unit = ds.attrs["unit"]
    resolution = ds.attrs["resolution"]
    print()
    print()
    print()
    print("Adding data from", input_file, "with:")
    print("unit:", unit)
    print("resolution:", resolution)
    print()
    print()
    print()
    chunks = (1, 512, 512)
    scale_factors = 4 * [[2, 2, 2]]

    mobie.add_segmentation(
        input_path=input_file,
        input_key=input_key,
        root=MOBIE_PROJECT_FOLDER,
        dataset_name=DATASET_NAME,
        segmentation_name=source_name,
        menu_name=menu_name,
        resolution=resolution,
        chunks=chunks,
        scale_factors=scale_factors,
        target=target,
        max_jobs=max_jobs,
        unit=unit,
    )


def main():
    dataset_folder = os.path.join(MOBIE_PROJECT_FOLDER, DATASET_NAME)
    ds_metadata = mobie.metadata.read_dataset_metadata(dataset_folder)
    sources = ds_metadata.get("sources", {})

    if "em_segm" not in sources:
        add_segmentation_from_n5(
            "/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem/segm_em_nuclei/em_nuclei.n5",
            "raw", "em_segm", "em_segm"
        )

    if "fluo" not in sources:
        add_from_n5(
            "/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem/fluo/fluo.n5",
            "raw", "fluo", "fluo"
        )
    if "em" not in sources:
        add_from_n5(
            "/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem/em/em.n5",
            "raw", "em", "em"
        )



    # scratch_dir = Path("/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem")
    # em_dir = scratch_dir / "em"
    # em_n5_path = em_dir / "em.n5"
    # f_em = z5py.File(em_n5_path, "r")
    # ds = f_em["raw"]
    # resolution = ds.attrs["resolution"]
    # transform_key = "ref_affine_transform"
    # R = np.array(ds.attrs[transform_key]["R"])
    # print("Affine transform from napari/scipy")
    # print(R)

    # # try with trafo in xyz order
    # trafo_xyz = native_to_bdv(R, invert=False)
    # print(trafo_xyz)

    # # try with scaled transform
    # trafo_scaled = native_to_bdv(R, invert=False, resolution=resolution)
    # print(trafo_scaled)
    
    # if "em_xyz_not_scaled" not in sources:
    #     add_from_n5(
    #         "/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem/em/em.n5",
    #         "raw", "em_xyz_forward", "em_xyz_forward", transform=trafo_xyz
    #     )
    
    # if "em_xyz_scaled" not in sources:
    #     add_from_n5(
    #         "/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem/em/em.n5",
    #         "raw", "em_xyz_scaled", "em_xyz_scaled", transform=trafo_scaled
    #     )

if __name__ == "__main__":
    main()
