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
    
    roi = np.s_[:]
    raw = read_volume("/scratch/buglakova/data/cryofib/segm_fibsem/F059/F059_A1_em.n5/", "input/raw_norm")
    output_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F059/F059_A1_em_train_network.n5/")
    
    write_volume(output_path, raw, "input/raw_norm")
    
    # Copy datasets to a new file
    
    boundaries = read_volume("/g/kreshuk/buglakova/data/tmp_view/F059_A1_em_train_network.n5", "predictions/boundaries", roi)
    extra = read_volume("/g/kreshuk/buglakova/data/tmp_view/F059_A1_em_train_network.n5", "predictions/extra", roi)

    # Store foreground mask
    print("Find foreground mask")
    fg_mask = get_fg_mask(raw).astype(np.int32)
    # write_volume(output_path, fg_mask, "masks/fg_mask")
    # fg_mask = read_volume(output_path, "predictions/fg_mask", roi)
    
    # Find all black parts that are mostly within GT boundaries and write that into a separate file
    print("Find zero areas inside boundaries")
    
    # write_volume(output_path, 1 - fg_mask, "predictions/bg")
    # write_volume(output_path, boundaries * fg_mask, "predictions/boundaries")
    write_volume(output_path, fg_mask - boundaries * fg_mask, "predictions/fg")
    # write_volume(output_path, extra * fg_mask, "predictions/extra")
    
    channels = np.stack([fg_mask - boundaries * fg_mask - extra * fg_mask, boundaries * fg_mask, extra * fg_mask, (1 - fg_mask)])
    write_volume(output_path, channels, "segmentation", chunks=[1, 1, 512, 512])
    
    
if __name__ == "__main__":
    main()