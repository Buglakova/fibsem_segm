import argparse

import z5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import elf.segmentation as eseg
from elf.segmentation.utils import normalize_input
from skimage.measure import label

def read_raw(f: z5py.File):
    raw = f["raw"]["raw_data"]
    raw.n_threads = 8
    print("Reading raw data into memory")
    raw = raw[:]
    print("Raw data shape: ", raw.shape, type(raw))
    return raw


def read_boundaries(f: z5py.File):
    g = f["predictions"]
    g.n_threads = 8
    print("Reading boundary probabilities into memory")
    boundaries = g["boundaries"][:]
    # extra = g["extracellular"][:]
    return boundaries


def get_zero_component(img: np.ndarray):
    bg = label(img == 0)
    component_sizes = [np.count_nonzero(bg == i) for i in np.unique(bg)[1:]]
    bg_ind = np.argmax(component_sizes) + 1
    bg = (bg == bg_ind)
    fg = (bg != bg_ind)
    return fg


def get_fg_mask(raw: np.ndarray):
    print("Compute foreground mask")
    print("Raw data shape: ", raw.shape)
    fg_mask = np.array([get_zero_component(img) for img in raw])
    return fg_mask


def run_multicut(ws, boundaries):
    rag = eseg.compute_rag(ws, n_threads=8)
    features = eseg.compute_boundary_mean_and_length(rag, boundaries, n_threads=8)
    bg_edges = (rag.uvIds() == 0).any(axis=1)
    z_edges = eseg.features.compute_z_edge_mask(rag, ws)
    costs, edge_sizes = features[:, 0], features[:, -1]
    costs = eseg.compute_edge_costs(
        costs, edge_sizes=edge_sizes, weighting_scheme="xyz", z_edge_mask=z_edges
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
    parser.add_argument("pred_path", type=str, help="Path to n5 file with network predictions")
    parser.add_argument("raw_path", type=str, help="Path to n5 path with raw EM stack")
    parser.add_argument("output_path", type=str, help="Path to n5 file, in which resulting segmentation will be stored")

    args = parser.parse_args()

    # Read predictions and raw data
    f = z5py.File(args.pred_path, "r")
    f_raw = z5py.File(args.raw_path, "r")

    # Create output file
    f_out = z5py.File(args.output_path, "a")

    boundaries = read_boundaries(f)
    raw = read_raw(f_raw)

    # Get foreground mask
    # It's predicted weirdly by unet,
    # so better to just get biggest connected component of zeros in raw
    fg_mask = get_fg_mask(raw)

    # Compute watershed
    print("Compute watershed ...")
    hmap = normalize_input(boundaries)
    ws, _ = eseg.stacked_watershed(hmap, mask=fg_mask, n_threads=8, threshold=0.4, sigma_seeds=2.0)

    # Store watershed
    chunks = (1, 512, 512)
    shape = ws.shape
    compression = "gzip"
    dtype = ws.dtype

    g = f_out.create_group("watershed")
    ds_ws = g.create_dataset("watershed", shape=shape, compression="gzip",
                                chunks=chunks, dtype=dtype)
    ds_ws.n_threads = 8
    print("Writing watershed")
    ds_ws[:] = ws

    # Run multicut
    print("Run multicut ...")
    seg = run_multicut(ws, boundaries)

    # Store multicut
    print("Write segmentation after multicut")
    g = f_out.create_group("segmentation")
    ds_seg = g.create_dataset("multicut", shape=shape, compression="gzip",
                                chunks=chunks, dtype=dtype)
    ds_seg.n_threads = 8
    print("Writing watershed")
    ds_seg[:] = seg