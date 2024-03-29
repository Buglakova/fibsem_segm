import argparse

import z5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import elf.segmentation as eseg
from elf.segmentation.utils import normalize_input
from elf.segmentation.workflows import edge_training, multicut_segmentation
from skimage.measure import label

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.vis import plot_overlay
import pickle


def main():
    # Spread ground truth instances to boundary parts, except where raw is close to 0
        
    train_multicut_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_train_multicut.n5")
    output_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_train_multicut_roi.n5")
    
    # Copy datasets to a new file
    roi = np.s_[300:600, 200:1000, 300:700]
    raw = read_volume(train_multicut_path, "input/raw_norm", roi)
    boundaries = read_volume(train_multicut_path, "predictions/boundaries", roi)
    write_volume(output_path, boundaries, "boundaries")
    fg_mask = read_volume(train_multicut_path, "masks/fg_mask", roi)
    extra_mask = read_volume(train_multicut_path, "masks/extra_mask", roi)
    mask = (fg_mask * (1 - extra_mask))
    ws = read_volume(train_multicut_path, "segmentation/3Dwatershed", roi)
    
    
    rf_file = open('rf.pkl', 'rb')
    rf = pickle.load(rf_file)
    rf_file.close()
    print(rf)

    all_annotated_rois = [np.s_[200:230, :, :], np.s_[279:310, :, :], np.s_[618:627, :, :], np.s_[1149:1160, :, :], np.s_[566:567, :, :]]
    roi = all_annotated_rois[0]
    
    hmap = normalize_input(boundaries)
    
    print(f"Raw {raw.dtype}")
    filters={"gaussianSmoothing": [1], "laplacianOfGaussian": [1], "hessianOfGaussianEigenvalues": [1]}
    seg = multicut_segmentation(raw.astype(np.float32), hmap, rf, False, watershed=ws, mask=mask, n_threads=16)
    print(seg.shape)
    
    write_volume(output_path, seg, "segmentation/rf_multicut")
    write_volume(output_path, ws, "segmentation/3Dwatershed")
    write_volume(output_path, raw, "raw")
    
    
if __name__ == "__main__":
    main()