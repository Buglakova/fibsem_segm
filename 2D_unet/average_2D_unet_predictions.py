import numpy as np
import matplotlib.pyplot as plt
import z5py
from pathlib import Path
from skimage.measure import label

from cryofib.n5_utils import read_volume, print_key_tree, write_volume
from cryofib.data_loaders import load_F107_A1_pred, load_F107_A1_n5, load_F107_A1_raw

def get_zero_component(img: np.ndarray):
    bg = label(img == 0)
    component_sizes = [np.count_nonzero(bg == i) for i in np.unique(bg)[1:]]
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
    print("Read raw data list")
    raw = load_F107_A1_raw()
    f_raw = load_F107_A1_n5()
    print("Open prediction files")
    f_pred = load_F107_A1_pred()
    print(f_pred)
    
    print("Get foreground mask")
    fg_mask = get_fg_mask(raw).astype(np.int8)
    print("fg mask type", fg_mask.dtype)
    write_volume(f_raw, fg_mask, "fg_mask")

    min_key = "predictions/2D_s0_quantile_norm_min"
    mean_key = "predictions/2D_s0_quantile_norm_mean"
    directions = ["predictions/2D_s0_quantile_norm_yzx", "predictions/2D_s0_quantile_norm_xzy", "predictions/2D_s0_quantile_norm"]
    for ds_name in ["/boundaries", "/extra"]:
        ds = [read_volume(f_pred, direction + ds_name) * fg_mask for direction in directions]
        ds_min = np.min(ds, axis=0)
        write_volume(f_pred, ds_min, min_key + ds_name)
        ds_mean = np.mean(ds, axis=0)
        write_volume(f_pred, ds_mean, mean_key + ds_name)


if __name__ == "__main__":
    main()