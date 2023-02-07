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

from skimage.morphology import binary_erosion


def main():
 
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Run blockwise watershed.
        """
    )

    parser.add_argument("pipeline_n5", type=str, help="Path of the input n5")
    parser.add_argument("--n_threads", type=int, default=8, help="")
    parser.add_argument("--beta", type=float, default=0.5, help="Boundary bias for converting merge probabilities to edge costs")
    
    args = parser.parse_args()
    
    n5_path = Path(args.pipeline_n5)
    
    log_dir = n5_path.parent / (n5_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = n5_path.parent / (n5_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_dir / "05_multicut.log", mode="w"),
                            logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    # Read data
    fg = read_volume(args.pipeline_n5, "input/fg_mask").astype(np.bool8)
    assert fg is not None, "Key doesn't exist"
    
    ws = read_volume(args.pipeline_n5, "watershed/3D_ws")
    assert ws is not None, "Key doesn't exist"
    
    boundaries = read_volume(args.pipeline_n5, "prediction/boundaries")
    assert boundaries is not None, "Key doesn't exist"
    
    extra = read_volume(args.pipeline_n5, "prediction/extra")
    assert extra is not None, "Key doesn't exist"
    
    fg = binary_erosion(binary_erosion(fg))
    
    hmap = normalize_input(np.maximum(boundaries, extra) * fg)
    plot_three_slices(hmap, plots_dir / "hmap_multicut.png", cmap="Greys_r")
    
    logging.info(f"Hmap min: {hmap.min()}, hmap max: {hmap.max()}")
    
    logging.info("Multicut start")
    seg = run_multicut(ws * fg, hmap, beta=args.beta, n_threads=args.n_threads)
    logging.info("Finish multicut")

    # Plots
    plot_three_slices(seg, plots_dir / "multicut.png", cmap=get_random_cmap())

    attrs = dict(get_attrs(args.pipeline_n5, "input/raw_norm"))
    attrs["description"] = f"Multicut, beta={args.beta}"
    write_volume(args.pipeline_n5, seg, "multicut", attrs=attrs)
    

   
    
    
if __name__ == "__main__":
    main()