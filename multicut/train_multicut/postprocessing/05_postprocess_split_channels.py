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


def bbox_3D(img, margin=10):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return max(rmin - margin, 0), min(rmax + margin, img.shape[0]), max(cmin - margin, 0), min(cmax + margin, img.shape[1]), max(zmin - margin, 0), min(zmax + margin, img.shape[2])



def main():
    # Spread ground truth instances to boundary parts, except where raw is close to 0
        
    train_multicut_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_train_multicut.n5")
    postprocess_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_postprocess.n5")
    network_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_train_network.n5")
    
    roi = np.s_[:, :, :]
    raw = read_volume(train_multicut_path, "input/raw", roi)
    write_volume(network_path, raw, "input/raw")
    extra = read_volume(train_multicut_path, "predictions/extra", roi)
    boundaries = read_volume(train_multicut_path, "predictions/boundaries", roi)
    fg_mask = read_volume(train_multicut_path, "masks/fg_mask", roi)
    multicut = read_volume(postprocess_path, "segmentation/extra_corrected", roi)
    
    # Low-intensity part of boundaries
    low_int = raw < 80
    low_int = low_int * boundaries
    low_int = (low_int > 0.99).astype(np.uint16)
    write_volume(postprocess_path, low_int, "boundaries/low_intensity")
    
    # Find instance boundaries
    instance_boundaries = find_boundaries(multicut, mode="outer") * fg_mask                      
    write_volume(postprocess_path, instance_boundaries.astype(np.uint16), "boundaries/instance_boundaries")
    
    boundaries = (low_int + instance_boundaries) > 0
    boundaries = boundaries.astype(np.uint16)
    extra = (multicut == 2).astype(np.uint16)
    foreground = (multicut > 2).astype(np.uint16)
    
    # Write channels separately
    write_volume(network_path, foreground, "channels/foreground")
    write_volume(network_path, boundaries, "channels/boundaries")
    write_volume(network_path, extra, "channels/extra")
    write_volume(network_path, (1 - fg_mask), "channels/out")
    
    # Write channels concatenated for training
    channels = np.stack([foreground, boundaries, extra, (1 - fg_mask)])
    write_volume(network_path, channels, "segmentation", chunks=[1, 1, 512, 512])

    
if __name__ == "__main__":
    main()