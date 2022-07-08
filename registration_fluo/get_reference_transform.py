import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import z5py
import re
from typing import Dict, List

from transforms3d.euler import euler2mat, mat2euler
from scipy.optimize import minimize

from cryofib.n5_utils import read_volume, tif2n5
from cryofib.affine_utils import parse_points, objective_function, params_to_transform, fit_affine_transform, get_affine_transform, get_translation_transform, get_scaling_transform

DATA_DIR = "/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem"

def fit_affine_transform(ref_points, moving_points, ref_res, moving_res, plots_dir="./plots"):
    ref_points = np.vstack([ref_points, np.ones(ref_points.shape[1])])
    S_ref = get_scaling_transform(ref_res)
    print("Refernce points resolution", ref_res)
    print("Scaled reference points", np.dot(S_ref, ref_points))
    ref_points = np.dot(S_ref, ref_points)

    moving_points = np.vstack([moving_points, np.ones(moving_points.shape[1])])
    S_moving = get_scaling_transform(moving_res)
    moving_points = np.dot(S_moving, moving_points)
    print("Moving points resolution", moving_res)
    print("Scaled moving points")
    print(moving_points)    

    t_0 = np.mean(ref_points, axis=1) - np.mean(moving_points, axis=1)
    print("COM difference t_0", t_0)
    res = minimize(objective_function, [t_0[0], t_0[1], t_0[2], 0, 0], args=(ref_points, moving_points), tol=0.001)

    print("Success", res.success)
    print(res.x)

    R = params_to_transform(res.x)
    new_points = np.dot(R, moving_points)

    # Plot points
    plt.figure()
    plots_dir = Path(plots_dir)
    plt.scatter(ref_points[2, :], ref_points[0, :], label="ref")
    plt.scatter(moving_points[2, :], moving_points[0, :], label="move")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.scatter(new_points[2, :], new_points[0, :], label="new")
    plt.legend()
    plt.savefig(plots_dir / "reference_affine_transform_xz.png", dpi=300)
    plt.show()

    plt.figure()
    plt.scatter(ref_points[2, :], ref_points[1, :], label="ref")
    plt.scatter(moving_points[2, :], moving_points[1, :], label="move")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(new_points[2, :], new_points[1, :], label="new")
    plt.legend()
    plt.savefig(plots_dir / "reference_affine_transform_xy.png", dpi=300)
    plt.show()

    print("Error (new - ref)")
    print(new_points - ref_points)

    
    return ref_points, moving_points, new_points, res.x, R

def main():
    # Set paths

    scratch_dir = Path("/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem")
    fluo_dir = scratch_dir / "fluo"
    em_dir = scratch_dir / "em"
    segm_em_nuclei_dir = scratch_dir / "segm_em_nuclei"

    fluo_path = fluo_dir / "fluo.n5"
    em_path = em_dir / "em.n5"
    segm_em_nuclei_path = segm_em_nuclei_dir / "em_nuclei.n5"

    alignment_points_path = scratch_dir / "Probe_coordinates_for_Alyona.txt"

    plots_dir = Path("./plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Parse Dragonfly points
    image_names = {"fluo": "HT6_3Dcrop",
                "em": "F107_a2_ali_crop_from70_bin2_ORS-denoised_YZX"}

    coords = parse_points(alignment_points_path, image_names)

    print("Parsed coordinates")
    print(coords)

    # Open n5 files
    f_fluo = z5py.File(fluo_path, "a")
    f_em = z5py.File(em_path, "a")
    f_segm_em_nuclei = z5py.File(segm_em_nuclei_path, "a")

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