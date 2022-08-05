from cryofib.n5_utils import read_volume, write_volume
import z5py
import numpy as np
from skimage.segmentation import find_boundaries, watershed
from skimage.morphology import remove_small_objects
from skimage.measure import label
from pathlib import Path

import vigra


def read_F107():
    raw_path = "/scratch/buglakova/F107_bin2_619-639_predictions/full_raw.n5" 
    pred_path = "/scratch/buglakova/F107_bin2_619-639_predictions/full_predictions.n5"
    multicut_path = "/scratch/buglakova/F107_bin2_619-639_predictions/full_multicut_beta_new.n5"
    
    
    f = z5py.File(pred_path, "r")
    f_raw = z5py.File(raw_path, "r")
    f_multicut = z5py.File(multicut_path, "r")

    roi = np.s_[:, :, :]
    raw = read_volume(f_raw, "raw/raw_data", roi)
    boundaries = read_volume(f, "predictions/boundaries", roi)
    extracellular = read_volume(f, "predictions/extracellular", roi)
    multicut = read_volume(f_multicut, f"segmentation/multicut_0.6", roi)

    return raw, boundaries, extracellular, multicut


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
    _, max_id = vigra.analysis.watershedsNew(boundaries, seeds=segmentation, out=segmentation)
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
    raw, boundaries, extra, multicut = read_F107()
    output_path = "/scratch/buglakova/F107_bin2_619-639_predictions/full_multicut_beta_corrected.n5"
    f_out = z5py.File(output_path, "a")
    out_key = "segmentation/multicut_0.6_postprocessed"

    min_extra_size = 100000
    extra_bin = extra > 0.9
    remove_small_objects(extra_bin, min_size=min_extra_size, out=extra_bin)

    seg = multicut.copy()
    fg_mask = get_fg_mask(raw)
    seg = seg + 2
    seg[extra_bin] = 2
    seg[~fg_mask] = 0

    min_segment_size = 10000
    print("Filtering")
    seg_filtered = apply_size_filter(seg, boundaries, raw, min_segment_size, exclude=None)
    seg[~fg_mask] = 0

    write_volume(f_out, raw, "raw/raw")
    write_volume(f_out, seg, "segmentation/multicut")
    write_volume(f_out, multicut, "segmentation/multicut_initial")
    write_volume(f_out, seg_filtered, "segmentation/filtered")



if __name__ == "__main__":
    main()