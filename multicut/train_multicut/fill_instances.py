import z5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening
from skimage.segmentation import find_boundaries, watershed
from elf.segmentation.utils import normalize_input

from cryofib.n5_utils import read_volume, write_volume, get_attrs
import vigra


def main():
    # Spread ground truth instances to boundary parts, except where raw is close to 0
        
    train_multicut_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_train_multicut.n5")
    
    # Copy datasets to a new file
    roi = np.s_[:, :, :]
    raw = read_volume(train_multicut_path, "input/raw", roi)
    boundaries = read_volume(train_multicut_path, "predictions/boundaries", roi)
    ground_truth = read_volume(train_multicut_path, "input/gt_instance_segmentation", roi)
    fg_mask = read_volume(train_multicut_path, "masks/fg_mask", roi)
    extra_mask = read_volume(train_multicut_path, "masks/extra_mask", roi)


    cell_instances = -1 * np.ones_like(ground_truth)
    all_annotated_rois = [np.s_[200:230, :, :], np.s_[279:310, :, :], np.s_[618:627, :, :], np.s_[1149:1160, :, :], np.s_[566:567, :, :]]
    
    for roi in all_annotated_rois:
        print(roi)
        segmentation = ground_truth[roi]
        print("zero boundaries")
        segmentation[segmentation == 1] = 0
        print("zero extra")
        segmentation = segmentation * (1 - extra_mask[roi])
        segmentation = segmentation + ((1 - fg_mask[roi]) * (segmentation.max() + 1))
        print(segmentation.shape)
        hmap = normalize_input(boundaries[roi])
        segmentation = segmentation.astype(np.uint32)
        print(f"hmap {hmap.dtype} {hmap.shape}, seeds {segmentation.dtype} {segmentation.shape}")
        # segmentation = watershed(boundaries[roi], markers=segmentation, mask=(fg_mask[roi] * (1 - extra_mask[roi])).astype(bool))
        vigra.analysis.watershedsNew(hmap, seeds=segmentation, out=segmentation)
        # ws, max_id = blockwise_two_pass_watershed(hmap, block_shape, halo, mask=fg_mask, threshold=threshold, sigma_seeds=sigma_seeds, n_threads=args.n_threads, verbose=True)
        print("zero mask")
        segmentation = segmentation * (fg_mask[roi] * (1 - extra_mask[roi]))
        cell_instances[roi] = segmentation
        write_volume(train_multicut_path, cell_instances, "input/gt_instance_no_boundaries")

    # Write resulting segmentation
    # write_volume(train_multicut_path, cell_instances, "input/gt_instance_no_boundaries")

    
if __name__ == "__main__":
    main()