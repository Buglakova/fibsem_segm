import napari
import numpy as np
import matplotlib.pyplot as plt
import z5py
from pathlib import Path

from cryofib.affine_utils import parse_points
from cryofib.n5_utils import read_volume

DATA_DIR = "/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem"


def main():
    # Set paths

    scratch_dir = Path(DATA_DIR)
    fluo_dir = scratch_dir / "fluo"
    em_dir = scratch_dir / "em"
    segm_em_nuclei_dir = scratch_dir / "segm_em_nuclei"

    fluo_path = fluo_dir / "fluo.n5"
    em_path = em_dir / "em.n5"
    segm_em_nuclei_path = segm_em_nuclei_dir / "em_nuclei.n5"

    alignment_points_path = scratch_dir / "Probe_coordinates_for_Alyona.txt"

    # Parse Dragonfly points
    image_names = {"fluo": "HT6_3Dcrop",
                "em": "F107_a2_ali_crop_from70_bin2_ORS-denoised_YZX"}

    coords = parse_points(alignment_points_path, image_names)

    print("Parsed coordinates")
    print(coords)

    # Add layers
    f_fluo = z5py.File(fluo_path, "r")
    f_em = z5py.File(em_path, "r")
    f_segm_em_nuclei = z5py.File(segm_em_nuclei_path, "r")

    # Get affine transform
    ref_points = coords["fluo"].T
    moving_points = coords["em"].T

    ref_points, moving_points, new_points, x, R = fit_affine_transform(ref_points[:, :-1], moving_points[:, :-1], f_fluo["raw"].attrs["resolution"], f_em["raw"].attrs["resolution"])
    print(f"Parameters (axis order ZYX): t_z = {x[0]}, t_y = {x[1]}, t_x = {x[2]}, phi_z = {x[3]}, phi_x = {x[4]}")
    
    print("Affine transform")
    print(R)
    print(R.tolist())

    affine_transform_attr = {"description": "Reference transformation calculated from manually selected points. Supposed to be applied to already scaled volumes. Parameters are [t_z, t_y, t_x, phi_z, phi_x]",
                            "rotation order": "szyx",
                            "parameters": x.tolist(),
                            "R": R.tolist()}

    # Save transforms to n5 attributes
    for f, key in zip([f_em, f_segm_em_nuclei], ["raw", "raw"]):
        print(f, key)
        ds = f[key]
        attributes = ds.attrs
        attributes["ref_affine_transform"] = affine_transform_attr



if __name__ == "__main__":
    main()