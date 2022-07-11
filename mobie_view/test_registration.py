import os
import mobie
import numpy as np
import elf.transformation as trafo
from pathlib import Path
from typing import List
import z5py

from cryofib.affine_utils import native_to_bdv, affine_to_parameter_vector


MOBIE_PROJECT_FOLDER = "/scratch/buglakova/data/cryofib/registration_fluo/mobie_projects/fibsem-registration-affine/data"
DATASET_NAME = "registration"


def test_registration(transform, view_name):
    ds_folder = os.path.join(MOBIE_PROJECT_FOLDER, DATASET_NAME)

    sources = [["fluo"], ["em"]]
    display_settings = [
        mobie.metadata.get_image_display("fluo", sources=["fluo"]),
        mobie.metadata.get_image_display("em", sources=["em"]),
    ]
    source_transforms = [
        mobie.metadata.get_affine_source_transform(["em"], transform)
    ]

    mobie.create_view(ds_folder, view_name, sources, display_settings, source_transforms,
                      overwrite=True, is_exclusive=True)


def main():

    # Read transformations from the n5
    scratch_dir = Path("/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem")
    em_dir = scratch_dir / "em"
    em_n5_path = em_dir / "em.n5"
    f_em = z5py.File(em_n5_path, "r")
    ds = f_em["raw"]
    resolution = ds.attrs["resolution"]
    transform_key = "ref_affine_transform"
    R = np.array(ds.attrs[transform_key]["R"])
    print("Affine transform from napari/scipy")
    print(R)

    # Convert to parameter vector
    R_param = affine_to_parameter_vector(R)

    # add the vanilla transform from Alyona
    test_registration(R_param, "vanilla")


    # try with trafo in xyz order
    trafo_xyz = native_to_bdv(R, invert=False)
    print(trafo_xyz)
    test_registration(trafo_xyz, "xyz_forward")

    # try with inverted trafo in xyz order
    trafo_xyz_inv = native_to_bdv(R, invert=True)
    print(trafo_xyz_inv)
    test_registration(trafo_xyz_inv, "xyz_inv")


if __name__ == "__main__":
    main()
