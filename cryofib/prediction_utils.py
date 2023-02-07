from bioimageio.core.prediction import predict_with_tiling
from cryofib.n5_utils import write_volume
from xarray import DataArray
import bioimageio.core
import numpy as np
import elf.segmentation as eseg


def predict_network(raw, model_path, output_f, prefix):
    shape = raw.shape
    chunks = (1, 512, 512)
    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:
        input_ = DataArray(raw[np.newaxis, np.newaxis, ...], dims=tuple("bczyx"))
        print("Input shape: ", input_.shape)
        print(pp.input_specs)
        # tiling = {"tile": {"x": 512, "y": 512, "z": 16},
        #             "halo": {"x": 32, "y": 32, "z": 4}}
        
        tiling = {"tile": {"x": 512, "y": 512, "z": 16},
                    "halo": {"x": 128, "y": 128, "z": 4}}
        # pred = predict_with_padding(pp, input_, padding=True)
        pred = predict_with_tiling(pp, input_, tiling=tiling)
        print(pred)
        pred = pred[0].values[0]
        print("Prediction shape: ", pred.shape)
        print("Input shape: ", input_.shape)
        # assert pred.shape[2:] == raw.shape[2:]
        write_volume(output_f, pred[0, ...], prefix + "/fg")
        write_volume(output_f, pred[1, ...], prefix + "/boundaries")
        write_volume(output_f, pred[2, ...], prefix + "/extra")
        write_volume(output_f, pred[3, ...], prefix + "/bg")
        
    return pred


def run_multicut(ws, boundaries, beta=0.5, n_threads=8):
    rag = eseg.compute_rag(ws, n_threads=8)
    features = eseg.compute_boundary_mean_and_length(rag, boundaries, n_threads=n_threads)
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