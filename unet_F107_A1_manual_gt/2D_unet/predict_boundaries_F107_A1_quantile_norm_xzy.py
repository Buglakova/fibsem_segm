import os

import bioimageio.core
import elf.segmentation as eseg
import numpy as np

from bioimageio.core.prediction import predict_with_padding, predict_with_tiling
from tqdm import trange
from xarray import DataArray
from cryofib.data_loaders import load_F107_A1_raw, load_F107_A1_raw_norm, load_F107_A1_pred
from cryofib.n5_utils import write_volume

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def predict_network(raw, model_path, output_f, prefix):
    shape = raw.shape
    chunks = (1, 512, 512)
    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:

        print(pp.input_specs)
        tiling = {"tile": {"x": 512, "y": 512},
                    "halo": {"x": 128, "y": 128}}
        raw = np.moveaxis(raw, [2, 0], [0, 1])
        pred_3d = np.zeros([4] + list(raw.shape))
        for idx, frame in enumerate(raw):
            input_ = DataArray(frame[np.newaxis, np.newaxis, ...], dims=tuple("bcyx"))
            print("Input shape: ", input_.shape)
            pred = predict_with_tiling(pp, input_, tiling=tiling)
            # print(pred)
            pred = pred[0].values[0]
            print("Prediction shape: ", pred.shape)
            print("Input shape: ", input_.shape)
            # assert pred.shape[2:] == raw.shape[2:]
            pred_3d[:, idx, :, :] = pred
        pred_3d = np.moveaxis(pred_3d, [1, 2], [3, 1])
        write_volume(output_f, pred_3d[0, ...], prefix + "/fg")
        write_volume(output_f, pred_3d[1, ...], prefix + "/boundaries")
        write_volume(output_f, pred_3d[2, ...], prefix + "/extra")
        write_volume(output_f, pred_3d[3, ...], prefix + "/bg")


def main():
    print("Read raw data list")
    raw = load_F107_A1_raw()
    print("Create prediction files")
    f_pred = load_F107_A1_pred()
    print(f_pred)
    model_path = "modelzoo/2D_s0_quantile_norm/2D_s0_quantile_norm.zip"
    prefix = "predictions/2D_s0_quantile_norm_xzy"
    predict_network(raw, model_path, f_pred, prefix)

if __name__ == "__main__":
    main()