import argparse

import z5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import elf.segmentation as eseg
from elf.segmentation.utils import normalize_input
from skimage.measure import label

from cryofib.n5_utils import read_volume, write_volume, get_attrs


def get_zero_component(img: np.ndarray):
    bg = label(img == 0)
    component_sizes = [np.count_nonzero(bg == i) for i in np.unique(bg)[1:]]
    if len(component_sizes) == 0:
        return img > 0
    bg_ind = np.argmax(component_sizes) + 1
    bg = (bg == bg_ind)
    fg = (bg != bg_ind)
    return fg


def get_fg_mask(raw: np.ndarray):
    print("Compute foreground mask")
    print("Raw data shape: ", raw.shape)
    fg_mask = np.array([get_zero_component(img) for img in raw])
    return fg_mask


def run_multicut(ws, boundaries, beta=0.5):
    rag = eseg.compute_rag(ws, n_threads=8)
    features = eseg.compute_boundary_mean_and_length(rag, boundaries, n_threads=8)
    bg_edges = (rag.uvIds() == 0).any(axis=1)
    z_edges = eseg.features.compute_z_edge_mask(rag, ws)
    costs, edge_sizes = features[:, 0], features[:, -1]
    costs = eseg.compute_edge_costs(
        costs, edge_sizes=edge_sizes, weighting_scheme="xyz", z_edge_mask=z_edges, beta=beta
    )
    # set all the weights to the background to be maximally repulsive
    assert len(bg_edges) == len(costs)
    costs[bg_edges] = -2 * np.max(np.abs(costs))
    node_labels = eseg.multicut.multicut_kernighan_lin(rag, costs)
    seg = eseg.project_node_labels_to_pixels(rag, node_labels, n_threads=8)
    return seg


if __name__ == "__main__":
    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Run watershed and multicut based on network prediction for probability of boundaries.
        """
    )

    parser.add_argument("raw_n5", type=str, help="Path of the input n5")
    parser.add_argument("raw_n5_key", type=str, help="Key of the dataset in the input n5")

    parser.add_argument("pred_n5", type=str, help="Path of the input n5")
    parser.add_argument("pred_n5_key", type=str, help="Key of the dataset in the input n5")

    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_n5_key", type=str, help="Group in the output n5 where to write waterched and multicut results")
    
    parser.add_argument("--beta", type=float, default=None, help="Boundary bias for converting merge probabilities to edge costs")
    args = parser.parse_args()

    beta = args.beta

    # Read predictions and raw data
    roi = np.s_[:]
    raw = read_volume(args.raw_n5, args.raw_n5_key, roi)
    boundaries = read_volume(args.pred_n5, args.pred_n5_key + "/boundaries", roi)
    extra = read_volume(args.pred_n5, args.pred_n5_key + "/extra", roi)

    # Sum up  boundaries and exrtacellular space probabilities,
    # Because otherwise some cells get joint through extracellular part

    boundaries = boundaries + extra
    boundaries = boundaries.astype(np.float32)

    # Get foreground mask
    # It's predicted weirdly by unet,
    # so better to just get biggest connected component of zeros in raw
    fg_mask = get_fg_mask(raw)

    # Compute watershed
    print("Compute watershed ...")
    hmap = normalize_input(boundaries)
    threshold = 0.4
    sigma_seeds = 2.0
    ws, _ = eseg.stacked_watershed(hmap, mask=fg_mask, n_threads=8, threshold=threshold, sigma_seeds=sigma_seeds)

    # Store watershed
    chunks = (1, 512, 512)
    shape = ws.shape
    compression = "gzip"
    dtype = ws.dtype

    attrs = dict(get_attrs(args.raw_n5, args.raw_n5_key))
    attrs["description"] = f"Watershed, threshold={threshold}, sigma_seeds={sigma_seeds}"
    write_volume(args.output_n5, ws, args.output_n5_key + "/ws", attrs=attrs, chunks=chunks)

    if beta is None:
        betas = [0.4, 0.5, 0.6]
    else:
        betas = [beta]

    for beta in betas:
        print(f"Beta = {beta}")
        # Run multicut
        print("Run multicut ...")
        seg = run_multicut(ws, boundaries, beta=beta)

        # Store multicut
        print("Write segmentation after multicut")
        attrs = dict(get_attrs(args.raw_n5, args.raw_n5_key))
        attrs["description"] = f"Multicut "
        write_volume(args.output_n5, seg, args.output_n5_key + "/multicut_" + str(beta), attrs=attrs, chunks=chunks)
