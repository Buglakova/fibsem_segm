import z5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening

from cryofib.n5_utils import read_volume, write_volume, get_attrs


def get_zero_component(img: np.ndarray):
    bg = label(img == 0)
    component_sizes = [np.count_nonzero(bg == i) for i in np.unique(bg)[1:]]
    if len(component_sizes) == 0:
        # No background
        return np.zeros_like(img) + 1
    bg_ind = np.argmax(component_sizes) + 1
    bg = (bg == bg_ind)
    fg = (bg != bg_ind)
    return fg


def get_fg_mask(raw: np.ndarray):
    print("Compute foreground mask")
    print("Raw data shape: ", raw.shape)
    fg_mask = np.array([get_zero_component(img) for img in raw])
    return fg_mask


def main():
    # Prepare data for multicut RF training
    # Make a single .n5 with all data that is used
    
    raw_path = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5")
    raw_key = "raw"
    raw_norm_key = "raw_norm"
    gt_key = "segmentation/corrected_instances"
    junctions_key = "segmentation/edoardo/junctions"
    nuclei_key = "segmentation/nuclei"
    boundaries_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em_3Dunet.n5")
    boundaries_key = "predictions/2D_s0_quantile_norm_mean/boundaries" 
    extra_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em_3Dunet.n5")
    extra_key = "predictions/2D_s0_quantile_norm_mean/extra"
    watershed_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em_3Dwatershed.n5")
    watershed_key = "predictions/3D_watershed"
    
    output_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_train_multicut.n5")
    
    # Copy datasets to a new file
    roi = np.s_[:]
    # raw = read_volume(raw_path, raw_key, roi)
    # write_volume(output_path, raw, "input/raw")
    nuclei = read_volume(raw_path, nuclei_key, roi)
    write_volume(output_path, nuclei, "input/nuclei")
    # raw_norm = read_volume(raw_path, raw_norm_key, roi)
    # write_volume(output_path, raw, "input/raw_norm")
    boundaries = read_volume(boundaries_path, boundaries_key, roi)
    # write_volume(output_path, boundaries, "predictions/boundaries")
    extra = read_volume(extra_path, extra_key, roi)
    # write_volume(output_path, extra, "predictions/extra")
    # watershed = read_volume(watershed_path, watershed_key, roi)
    # write_volume(output_path, watershed, "segmentation/3Dwatershed")
    # ground_truth = read_volume(raw_path, gt_key, roi)
    # write_volume(output_path, ground_truth, "input/gt_instance_segmentation")
    # junctions = read_volume(raw_path, junctions_key, roi)
    # write_volume(output_path, junctions, "input/junctions")
    
    # Store foreground mask
    print("Find foreground mask")
    # fg_mask = get_fg_mask(raw).astype(np.int32)
    # write_volume(output_path, fg_mask, "masks/fg_mask")
    fg_mask = read_volume(output_path, "masks/fg_mask", roi)
    
    # Find all black parts that are mostly within GT boundaries and write that into a separate file
    print("Find zero areas inside boundaries")
    extra_mask = (raw < 50).astype(np.int32)
    extra_mask = extra_mask * fg_mask * (boundaries + extra)
    extra_mask = extra_mask > 0.1
    extra_mask = binary_opening(extra_mask).astype(np.int32)
    
    write_volume(output_path, extra_mask, "masks/extra_mask")
    
    
if __name__ == "__main__":
    main()