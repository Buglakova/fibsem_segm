import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.preprocess_utils import percentile_norm
from cryofib.vis import plot_three_slices



def main():
 
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Find a foreground mask. Background is the biggest connected component of zero pixels per z slice.
        """
    )

    parser.add_argument("pipeline_n5", type=str, help="Path of the input n5")
    parser.add_argument("--pmin", type=float, default=1, help="Min percentile")
    parser.add_argument("--pmax", type=float, default=98, help="Min percentile")
    
    args = parser.parse_args()
    
    n5_path = Path(args.pipeline_n5)
    
    log_dir = n5_path.parent / (n5_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = n5_path.parent / (n5_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_dir / "02_quantile_norm.log", mode="w"),
                            logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    # Read data
    img = read_volume(args.pipeline_n5, "input/raw")
    assert img is not None, "Key doesn't exist"
    
    fg = read_volume(args.pipeline_n5, "input/fg_mask").astype(np.bool8)
    assert fg is not None, "Key doesn't exist"
    
    # Normalize
    logging.info(f"Value range (not background): min {img[fg].min()}, max {img[fg].max()} of type {img.dtype}")
    pmin = np.percentile(img[fg], args.pmin)
    pmax = np.percentile(img[fg], args.pmax)
    logging.info(f"Normalize using percentiles pmin {args.pmin}={pmin} and pmax {args.pmax}={pmax}")

    img_norm = percentile_norm(img, args.pmin, args.pmax).astype(np.float32)
    img_norm = img_norm * fg

    attrs = dict(get_attrs(args.pipeline_n5, "input/raw"))
    if "description" in attrs.keys():
        description = attrs["description"]
    else:
        description = ""
    attrs["description"] = description + f"Normalized to percentiles pmin = {pmin}, pmax = {pmax}"
    write_volume(args.pipeline_n5, img_norm, "input/raw_norm", attrs=attrs)
    
    # Plots
    plot_three_slices(img, plots_dir / "raw.png", cmap="Greys_r")
    plot_three_slices(img_norm, plots_dir / "raw_norm.png", cmap="Greys_r")
    
    plt.hist(img_norm.flatten(), bins=100)
    plt.savefig(plots_dir / "raw_norm_hist.png", dpi=300)
    
    
    
if __name__ == "__main__":
    main()