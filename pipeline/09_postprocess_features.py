import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.preprocess_utils import percentile_norm
from cryofib.vis import plot_three_slices, get_random_cmap


from skimage.measure import label
from skimage.morphology import binary_opening, binary_erosion
from skimage.segmentation import find_boundaries, watershed

import vigra

from elf.segmentation.utils import normalize_input

import pandas as pd

def get_bbox_roi(img, coord_min, coord_max, margin=10):
    z_min = max(coord_min[0] - margin, 0)
    z_max = min(coord_max[0] + margin, img.shape[0])
    y_min = max(coord_min[1] - margin, 0)
    y_max = min(coord_max[1] + margin, img.shape[1])
    x_min = max(coord_min[2] - margin, 0)
    x_max = min(coord_max[2] + margin, img.shape[2])
    
    return np.s_[z_min:z_max, y_min:y_max, x_min:x_max]


def main():
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Postprocess segmentation.
        """
    )

    parser.add_argument("pipeline_n5", type=str, help="Path of the input n5")
    parser.add_argument("--min_area", type=int, default=100000, help="")
    
    args = parser.parse_args()
    
    n5_path = Path(args.pipeline_n5)
    
    log_dir = n5_path.parent / (n5_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = n5_path.parent / (n5_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_dir / f"postprocess_features.log", mode="w"),
                            logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    # Read data
    multicut = read_volume(args.pipeline_n5, "postprocess/renumber").astype(np.uint32)
    assert multicut is not None, "Key doesn't exist"
    
    nuclei = read_volume(args.pipeline_n5, "input/nuclei").astype(np.float32)
    assert nuclei is not None, "Key doesn't exist"
    
    logging.info(f"Segmentation shape {multicut.shape}")
   
    # Assign extracellular
 

    logging.info(f"Start calculating segment features")
    stat_feature_names = ["Count", "Sum"]
    coord_feature_names = ['Coord<Maximum >', 'Coord<Minimum >']
    feature_names = stat_feature_names + coord_feature_names
    node_features = vigra.analysis.extractRegionFeatures(nuclei, multicut, features=feature_names)
    logging.info(f"End calculating segment features")
    
    areas = node_features["Count"]
    indices = np.arange(0, node_features.maxRegionLabel() + 1)
    nuclei_areas = node_features["Sum"]
    
    result = pd.DataFrame({"cell_idx": indices, "area": areas, "nuclei_area": nuclei_areas})
    result.to_csv(log_dir / "features.csv", index=False)

    
if __name__ == "__main__":
    main()