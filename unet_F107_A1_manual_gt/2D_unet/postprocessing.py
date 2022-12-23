from cryofib.n5_utils import read_volume, write_volume
import z5py
import numpy as np
from skimage.segmentation import find_boundaries, watershed
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.filters import gaussian

from pathlib import Path
import matplotlib.pyplot as plt

import vigra
from cryofib.data_loaders import load_F107_A1_raw, load_F107_A1_pred, load_F107_A1_multicut


def fast_count_unique(array: np.array):
    if array.dtype is not np.int64:
        print(f"array type is {array.dtype}. Converting to int64, possibly can cause errors")
    n_features = int(np.max(array))
    counts = np.bincount(array.flatten().astype(np.int64), minlength=n_features + 1)
    return np.arange(n_features + 1), counts


def apply_size_filter(segmentation, boundaries, raw, size_filter, exclude=None):
    """ Apply size filter to segmentation.
    Arguments:
        segmentation [np.ndarray] - input segmentation.
        input_ [np.ndarray] - input height map.
        size_filter [int] - minimal segment size.
        exclude [list] - list of segment ids that will not be size filtered (default: None).
    Returns:
        np.ndarray - size filtered segmentation
        int - max id of size filtered segmentation
    """
    print("Find labels with small sizes")
    # ids, sizes = np.unique(segmentation, return_counts=True)
    ids, sizes = fast_count_unique(segmentation)
    print(ids)
    print(sizes)
    filter_ids = ids[sizes < size_filter]
    if exclude is not None:
        filter_ids = filter_ids[np.logical_not(np.in1d(filter_ids, exclude))]
    filter_mask = np.in1d(segmentation, filter_ids).reshape(segmentation.shape)
    segmentation[filter_mask] = 0
    
    print("Get foreground mask")
    # fg_mask = get_fg_mask(raw)
    # compactness parameter of watershed??
    print("Vigra watershed")
    # segmentation = watershed(boundaries, markers=segmentation, mask=fg_mask)
    print("Segmentation type", segmentation.dtype, segmentation.shape)
    print("Boundaries type", boundaries.dtype, boundaries.shape)
    boundaries = boundaries.astype(np.float32)
    segmentation = segmentation.astype(np.uint32)
    print("Segmentation type", segmentation.dtype, segmentation.shape)
    print("Boundaries type", boundaries.dtype, boundaries.shape)
    _, max_id = vigra.analysis.watershedsNew(boundaries, seeds=segmentation, out=segmentation, method="RegionGrowing")
    return segmentation


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
    roi = np.s_[:]
    print("Read raw data list")
    raw = load_F107_A1_raw()[roi]
    print("Read prediction file")
    f_pred = load_F107_A1_pred()
    boundaries = read_volume(f_pred, "predictions/2D_s0_quantile_norm_min/boundaries", roi)
    boundaries = gaussian(boundaries, sigma=2)

    f_multi = load_F107_A1_multicut()
    
    multicut = read_volume(f_multi, "2D_s0_quantile_norm_averaged/multicut_extra_0.6", roi)

    out_key = "segmentation/multicut_0.6_postprocessed_10000"

    seg = multicut.copy()
    fg_mask = get_fg_mask(raw)
    print("Segmentation max value", seg.max())
    seg[~fg_mask] = seg.max() + 1

    min_segment_size = 10000
    print("Filtering")
    seg_filtered = apply_size_filter(seg, boundaries, raw, min_segment_size, exclude=None)
    seg[~fg_mask] = 0

    write_volume(f_multi, seg_filtered, out_key)



if __name__ == "__main__":
    main()