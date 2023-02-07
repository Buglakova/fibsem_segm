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
from skimage.morphology import binary_opening
from skimage.segmentation import find_boundaries, watershed

import vigra

from elf.segmentation.utils import normalize_input


def bbox_3D(img, margin=10):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return max(rmin - margin, 0), min(rmax + margin, img.shape[0]), max(cmin - margin, 0), min(cmax + margin, img.shape[1]), max(zmin - margin, 0), min(zmax + margin, img.shape[2])



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
    
    multicut = read_volume(args.pipeline_n5, "multicut")
    assert multicut is not None, "Key doesn't exist"
    
    extra = read_volume(args.pipeline_n5, "prediction/extra")
    assert extra is not None, "Key doesn't exist"
    
    boundaries = read_volume(args.pipeline_n5, "prediction/boundaries")
    assert boundaries is not None, "Key doesn't exist"
    
    logging.info(f"Segmentation shape {multicut.shape}")
   
    # Assign extracellular
    result = multicut
    logging.info(result.dtype)
    result = result + 2
    logging.info(result.dtype)
    logging.info("assign fg mask")
    logging.info(f"fg mask {fg.dtype}")
    result = result * fg
    logging.info(result.dtype)


    max_id = int(np.max(result))
    logging.info(f"Max id {max_id}")
    logging.info(type(max_id))
    areas = []
    nonzero_id = 3
    for seg_id in range(3, max_id + 1):
        logging.info(f"{seg_id} out of {max_id}")
        seg_mask = result == seg_id
        area = len(seg_mask[seg_mask])
        areas.append(area)
        logging.info(area)
        if area > 0:
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(seg_mask)
            logging.info(f"{rmin}, {rmax}, {cmin}, {cmax}, {zmin}, {zmax}")
            id_extra = extra[rmin:rmax, cmin:cmax, zmin:zmax]
            id_boundaries = boundaries[rmin:rmax, cmin:cmax, zmin:zmax]
            id_multicut = result[rmin:rmax, cmin:cmax, zmin:zmax]
            id_mask = seg_mask[rmin:rmax, cmin:cmax, zmin:zmax]
            logging.info(f"Bbox shape {id_mask.shape}")
            
            if np.median(id_extra[id_mask]) > 0.5:
                result[seg_mask] = 2
                logging.info("Extracellular")
            
            elif area < args.min_area:
                id_multicut[id_mask] = 0
                ws_fill, _ = vigra.analysis.watershedsNew(normalize_input(np.maximum(id_boundaries, id_extra)), seeds=id_multicut.astype(np.uint32))
                id_multicut[id_mask] = ws_fill[id_mask]
                result[rmin:rmax, cmin:cmax, zmin:zmax] = id_multicut
                logging.info("Merge")
                
        if seg_id % 1000 == 0:
            write_volume(args.pipeline_n5, result.astype(np.uint32), "postprocess/extra_size")
        
    write_volume(args.pipeline_n5, result.astype(np.uint32), "postprocess/extra_size")

    with open(log_dir / 'areas.txt','w') as tfile:
	    tfile.write('\n'.join([str(area) for area in areas]))
    
    areas = np.array(areas)
    plt.hist(areas[areas > 0], bins=100)
    plt.savefig(plots_dir / "areas_multicut.png", dpi=300)
    
    plt.figure()
    plt.hist(np.log10(areas[areas > 0]), bins=100)
    plt.savefig(plots_dir / "areas_multicut_log.png", dpi=300)
     
    plot_three_slices(result, plots_dir / "postprocess.png", cmap=get_random_cmap())

    
if __name__ == "__main__":
    main()