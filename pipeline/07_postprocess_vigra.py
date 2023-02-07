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

from tqdm import tqdm




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
                    handlers=[logging.FileHandler(log_dir / f"postprocess.log", mode="w"),
                            logging.StreamHandler(sys.stdout)],
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    # Read data
    fg = read_volume(args.pipeline_n5, "input/fg_mask").astype(np.bool8)
    assert fg is not None, "Key doesn't exist"
    
    fg = binary_erosion(binary_erosion(fg))
    
    multicut = read_volume(args.pipeline_n5, "multicut")
    assert multicut is not None, "Key doesn't exist"
    
    extra = read_volume(args.pipeline_n5, "prediction/extra")
    assert extra is not None, "Key doesn't exist"
    
    boundaries = read_volume(args.pipeline_n5, "prediction/boundaries")
    assert boundaries is not None, "Key doesn't exist"
    
    logging.info(f"Segmentation shape {multicut.shape}")
   
    # Assign extracellular
    result = multicut.astype(np.uint32)
    logging.info(result.dtype)
    result = result + 2
    logging.info(result.dtype)
    logging.info("assign fg mask")
    logging.info(f"fg mask {fg.dtype}")
    result = result * fg
    logging.info(result.dtype)
    

    logging.info(f"Start calculating segment features")
    stat_feature_names = ["Count", "Quantiles"]
    coord_feature_names = ['Coord<Maximum >', 'Coord<Minimum >']
    feature_names = stat_feature_names + coord_feature_names
    node_features = vigra.analysis.extractRegionFeatures(extra, result, features=feature_names)
    logging.info(f"End calculating segment features")
    
    coords_min = node_features['Coord<Minimum >'].astype(np.uint16)
    coords_max = node_features['Coord<Maximum >'].astype(np.uint16)
    extra_median = node_features["Quantiles"][:, 3]
    areas = node_features["Count"]
    indices = np.arange(0, node_features.maxRegionLabel() + 1)
    
    with open(log_dir / 'areas_multicut.txt','w') as tfile:
	    tfile.write('\n'.join([str(area) for area in areas]))
     
    plt.hist(areas[areas > 0], bins=100)
    plt.savefig(plots_dir / "areas_multicut.png", dpi=300)
    
    plt.figure()
    plt.hist(np.log10(areas[areas > 0]), bins=100)
    plt.savefig(plots_dir / "areas_multicut_log.png", dpi=300)
    
    replace_extra_indices = indices[extra_median > 0.5]
    logging.info(f"Start replacing extracellular segments {len(replace_extra_indices)}")
    for idx in tqdm(replace_extra_indices):
        bbox_roi = get_bbox_roi(result, coords_min[idx, :], coords_max[idx, :])
        idx_result = result[bbox_roi]
        idx_result[idx_result == idx] = 2
        result[bbox_roi] = idx_result
        areas[idx] = 0
    logging.info(f"End replacing extracellular segments")
    write_volume(args.pipeline_n5, result, "postprocess/extra")
    
    replace_small_segments = indices[(areas > 0) & (areas < 1e5)]
    logging.info(f"Start replacing extracellular segments {len(replace_small_segments)}")
    for idx in tqdm(replace_small_segments):
        bbox_roi = get_bbox_roi(result, coords_min[idx, :], coords_max[idx, :])
        idx_result = result[bbox_roi]
        idx_seeds = result[bbox_roi].copy()
        idx_seeds[idx_result == idx] = 0
        idx_extra = extra[bbox_roi]
        idx_boundaries = boundaries[bbox_roi]
        ws_fill, _ = vigra.analysis.watershedsNew(normalize_input(np.maximum(idx_boundaries, idx_extra)), seeds=idx_seeds)
        idx_result[idx_result == idx] = ws_fill[idx_result == idx]
        result[bbox_roi] = idx_result
        areas[idx] = 0

    with open(log_dir / 'areas_postprocess.txt','w') as tfile:
        tfile.write('\n'.join([str(area) for area in areas]))
     
    plt.hist(areas[areas > 0], bins=100)
    plt.savefig(plots_dir / "areas_postprocess.png", dpi=300)
    
    plt.figure()
    plt.hist(np.log10(areas[areas > 0]), bins=100)
    plt.savefig(plots_dir / "areas_postprocess_log.png", dpi=300)     
    plot_three_slices(result, plots_dir / "postprocess.png", cmap=get_random_cmap())
    
    write_volume(args.pipeline_n5, result, "postprocess/extra_size")

    
if __name__ == "__main__":
    main()