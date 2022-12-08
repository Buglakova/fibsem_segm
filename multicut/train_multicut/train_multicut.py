import argparse

import z5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import elf.segmentation as eseg
from elf.segmentation.utils import normalize_input
from elf.segmentation.watershed import blockwise_two_pass_watershed
from skimage.measure import label

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.vis import plot_overlay


def get_zero_component(img: np.ndarray):
    bg = label(img == 0)
    component_sizes = [np.count_nonzero(bg == i) for i in np.unique(bg)[1:]]
    if len(component_sizes) == 0:
        # No background
        return np.zeros_like(img) + 1
    bg_ind = np.argmax(component_sizes) + 1
    bg = (bg == bg_ind)
    fg = (bg != bg_ind)
    return fg


def get_fg_mask(raw: np.ndarray):
    print("Compute foreground mask")
    print("Raw data shape: ", raw.shape)
    fg_mask = np.array([get_zero_component(img) for img in raw])
    return fg_mask


def main():
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Run 3D watershed based on boundary probabilities. Use 8 threads by default.
        """
    )

    parser.add_argument("raw_n5", type=str, help="Path of the input n5")
    parser.add_argument("raw_n5_key", type=str, help="Key of the raw dataset in the input n5")

    parser.add_argument("boundaries_n5", type=str, help="Path of the input n5")
    parser.add_argument("boundaries_n5_key", type=str, help="Key of the boundary probability in the input n5")
    
    parser.add_argument("extra_n5", type=str, help="Path of the input n5")
    parser.add_argument("extra_n5_key", type=str, help="Key of the boundary probability in the input n5")

    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_n5_key", type=str, help="Group in the output n5 where to write watershed and multicut results")
    
    parser.add_argument("--n_threads", type=float, default=8, help="Boundary bias for converting merge probabilities to edge costs")
    args = parser.parse_args()
    
    
    # Read predictions and raw data
    roi = np.s_[:]
    raw = read_volume(args.raw_n5, args.raw_n5_key, roi)
    boundaries = read_volume(args.boundaries_n5, args.boundaries_n5_key, roi)
    extra = read_volume(args.extra_n5, args.extra_n5_key, roi)

    # Sum up  boundaries and exrtacellular space probabilities,
    # Because otherwise some cells get joint through extracellular part

    boundaries = boundaries + extra
    boundaries = boundaries.astype(np.float32)

    # Get foreground mask
    # It's predicted weirdly by unet,
    # so better to just get biggest connected component of zeros in raw
    fg_mask = get_fg_mask(raw)

    # Compute watershed
    print("Normalize input")    
    hmap = normalize_input(boundaries)
    threshold = 0.4
    sigma_seeds = 2.0
    block_shape = [256, 256, 256]
    halo = [64, 64, 64]
    
    print("Compute watershed ...")
    ws, max_id = blockwise_two_pass_watershed(hmap, block_shape, halo, mask=fg_mask, threshold=threshold, sigma_seeds=sigma_seeds, n_threads=args.n_threads, verbose=True)
    
    # Store watershed
    chunks = (1, 512, 512)
    shape = ws.shape
    compression = "gzip"
    dtype = ws.dtype

    attrs = dict(get_attrs(args.raw_n5, args.raw_n5_key))
    attrs["description"] = f"Watershed, threshold={threshold}, sigma_seeds={sigma_seeds}"
    write_volume(args.output_n5, ws, args.output_n5_key, attrs=attrs, chunks=chunks)
    
    
if __name__ == "__main__":
    main()