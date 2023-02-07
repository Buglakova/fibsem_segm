import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.preprocess_utils import percentile_norm
from cryofib.vis import plot_three_slices, get_random_cmap

import elf.segmentation as eseg
from elf.segmentation.utils import normalize_input
from elf.segmentation.watershed import blockwise_two_pass_watershed





def main():
 
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Run blockwise watershed.
        """
    )

    parser.add_argument("pipeline_n5", type=str, help="Path of the input n5")
    parser.add_argument("--n_threads", type=float, default=8, help="Boundary bias for converting merge probabilities to edge costs")
    parser.add_argument("--block_shape", type=int, default=256, help="Boundary bias for converting merge probabilities to edge costs")
    parser.add_argument("--halo", type=int, default=64, help="Boundary bias for converting merge probabilities to edge costs")
    
    args = parser.parse_args()
    
    n5_path = Path(args.pipeline_n5)
    
    log_dir = n5_path.parent / (n5_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = n5_path.parent / (n5_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_dir / "04_3D_watershed.log", mode="w"),
                            logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    # Read data
    fg = read_volume(args.pipeline_n5, "input/fg_mask").astype(np.bool8)
    assert fg is not None, "Key doesn't exist"
    
    boundaries = read_volume(args.pipeline_n5, "prediction/boundaries")
    assert boundaries is not None, "Key doesn't exist"
    
    extra = read_volume(args.pipeline_n5, "prediction/extra")
    assert extra is not None, "Key doesn't exist"
    
    hmap = normalize_input((boundaries + extra) * fg)
    plt.hist(hmap.flatten(), bins=100, log=True)
    plt.savefig(plots_dir / "hmap_hist.png", dpi=300)
    plot_three_slices(hmap, plots_dir / "hmap.png", cmap="Greys_r")
    
    
    threshold = 0.4
    sigma_seeds = 2.0
    # block_shape = [256, 256, 256]
    block_shape = [args.block_shape] * 3
    # halo = [64, 64, 64]
    halo = [args.halo] * 3
    
    print("Compute watershed ...")
    ws, max_id = blockwise_two_pass_watershed(hmap, block_shape, halo, mask=fg, threshold=threshold, sigma_seeds=sigma_seeds, n_threads=args.n_threads, verbose=True)

    attrs = dict(get_attrs(args.pipeline_n5, "input/raw_norm"))
    attrs["description"] = f"Watershed, threshold={threshold}, sigma_seeds={sigma_seeds}"
    write_volume(args.pipeline_n5, ws, "watershed/3D_ws", attrs=attrs)
    
    # Plots
    plot_three_slices(ws, plots_dir / "3D_ws.png", cmap=get_random_cmap())
   
    
    
if __name__ == "__main__":
    main()