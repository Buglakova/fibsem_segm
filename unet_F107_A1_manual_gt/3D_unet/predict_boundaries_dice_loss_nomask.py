import os

import bioimageio.core
import elf.segmentation as eseg
import numpy as np

from bioimageio.core.prediction import predict_with_padding, predict_with_tiling
from tqdm import trange
from xarray import DataArray
from cryofib.data_loaders import load_F107_A1_raw, load_F107_A1_pred
from cryofib.n5_utils import write_volume


def predict_network(raw, model_path, output_f, prefix):
    shape = raw.shape
    chunks = (1, 512, 512)
    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:
        input_ = DataArray(raw[np.newaxis, np.newaxis, ...], dims=tuple("bczyx"))
        print("Input shape: ", input_.shape)
        print(pp.input_specs)
        tiling = {"tile": {"x": 512, "y": 512, "z": 16},
                    "halo": {"x": 16, "y": 16, "z": 2}}
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


def main():
    print("Read raw data list")
    raw = load_F107_A1_raw()
    print("Create prediction files")
    f_pred = load_F107_A1_pred()
    print(f_pred)
    model_path = "modelzoo/full_masked_dice_s0_16x512x512_quantile_norm_no_mask/full_masked_dice_s0_16x512x512_quantile_norm_no_mask.zip"
    prefix = "predictions/dice_s0_16x512x512_quantile_norm_no_mask"
    predict_network(raw, model_path, f_pred, prefix)

if __name__ == "__main__":
    main()