import argparse
import logging
import sys

import z5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import elf.segmentation as eseg
from elf.segmentation.utils import normalize_input
from elf.segmentation.workflows import edge_training
from skimage.measure import label

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.vis import plot_overlay
import pickle


def main():
    
    parser = argparse.ArgumentParser(
        description="""Run blockwise watershed.
        """
        )

    parser.add_argument("pipeline_n5", type=str, help="Path of the input n5")
    parser.add_argument("--n_threads", type=int, default=16, help="")
    
    args = parser.parse_args()
    
    n5_path = Path(args.pipeline_n5)
    
    log_dir = n5_path.parent / (n5_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = n5_path.parent / (n5_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_dir / "train_multicut.log", mode="w"),
                            logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    # Read data
    roi = np.s_[:300, :, :]
    
    raw = read_volume(args.pipeline_n5, "input/raw_norm", roi=roi)
    assert raw is not None, "Key doesn't exist"
    
    fg = read_volume(args.pipeline_n5, "input/fg_mask", roi=roi).astype(np.bool8)
    assert fg is not None, "Key doesn't exist"
    
    ws = read_volume(args.pipeline_n5, "watershed/3D_ws", roi=roi)
    assert ws is not None, "Key doesn't exist"
    
    boundaries = read_volume(args.pipeline_n5, "prediction/boundaries", roi=roi)
    assert boundaries is not None, "Key doesn't exist"
    
    extra = read_volume(args.pipeline_n5, "prediction/extra", roi=roi)
    assert extra is not None, "Key doesn't exist"
    
    hmap = normalize_input((boundaries + extra) * fg)
    
    
    # Spread ground truth instances to boundary parts, except where raw is close to 0
        
    train_multicut_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_train_multicut.n5")
    
    # Copy datasets to a new file
    
    labels = read_volume(train_multicut_path, "input/gt_instance_no_boundaries", roi)
    all_annotated_rois = [np.s_[200:230, :, :], np.s_[279:310, :, :], np.s_[618:627, :, :], np.s_[1149:1160, :, :], np.s_[566:567, :, :]]
    roi = all_annotated_rois[0]

    print(f"Raw {raw.dtype}")
    # filters={"gaussianSmoothing": [1], "laplacianOfGaussian": [1], "hessianOfGaussianEigenvalues": [1]}
    rf = edge_training(raw[roi].astype(np.float32), hmap[roi], labels[roi], False, watershed=ws[roi], mask=fg[roi], n_threads=args.n_threads)
    print(rf)
    output = open('rf.pkl', 'wb')
    pickle.dump(rf, output)
    output.close()
    
    
if __name__ == "__main__":
    main()