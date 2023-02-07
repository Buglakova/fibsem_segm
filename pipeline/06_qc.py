import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.preprocess_utils import percentile_norm
from cryofib.vis import plot_three_slices, get_random_cmap
from cryofib.prediction_utils import run_multicut

import elf.segmentation as eseg
from elf.segmentation.utils import normalize_input


def main():
 
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""QC segmentation.
        """
    )

    parser.add_argument("pipeline_n5", type=str, help="Path of the input n5")
    parser.add_argument("--seg_key", type=str, default="multicut", help="Segmentation to check")
    parser.add_argument("--n_threads", type=int, default=8, help="")
    
    args = parser.parse_args()
    
    n5_path = Path(args.pipeline_n5)
    
    log_dir = n5_path.parent / (n5_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = n5_path.parent / (n5_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_dir / f"QC_{args.seg_key}.log", mode="w"),
                            logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    # Read data
    fg = read_volume(args.pipeline_n5, "input/fg_mask").astype(np.bool8)
    assert fg is not None, "Key doesn't exist"
    
    seg = read_volume(args.pipeline_n5, args.seg_key)
    assert seg is not None, "Key doesn't exist"
    
    logging.info(f"Segmentation shape {seg.shape}")
    seg = seg * fg
    
    max_id = int(np.max(seg))
    logging.info(f"Max id {max_id}")
    print(type(max_id))
    areas = []
    for seg_id in range(1, max_id + 1):
        print(f"{seg_id} out of {max_id}")
        seg_mask = seg == seg_id
        area = len(seg_mask[seg_mask])
        areas.append(area)
    
    # Plots
    plt.hist(areas, bins=100)
    plt.savefig(plots_dir / f"areas_{args.seg_key}.png", dpi=300)
    
    plt.hist(areas, bins=100)
    plt.savefig(plots_dir / f"areas_log_{args.seg_key}.png", dpi=300, log=True)
    
    
if __name__ == "__main__":
    main()