import os

import bioimageio.core
import elf.segmentation as eseg
import numpy as np

from bioimageio.core.prediction import predict_with_padding, predict_with_tiling
from tqdm import trange
from xarray import DataArray
from cryofib.data_loaders import load_platynereis_memb_ds, load_platynereis_memb_n5, load_platynereis_pred_n5
from cryofib.n5_utils import write_volume


def predict_network(raw, model_path, output_f, prefix):
    shape = raw.shape
    chunks = (1, 512, 512)
    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:
        input_ = DataArray(raw[np.newaxis, np.newaxis, ...], dims=tuple("bczyx"))
        print("Input shape: ", input_.shape)
        print(pp.input_specs)
        tiling = {"tile": {"x": 128, "y": 128, "z": 64},
                    "halo": {"x": 16, "y": 16, "z": 4}}
        # pred = predict_with_padding(pp, input_, padding=True)
        pred = predict_with_tiling(pp, input_, tiling=tiling)
        print(pred)
        pred = pred[0].values[0]
        print("Prediction shape: ", pred.shape)
        print("Input shape: ", input_.shape)
        # assert pred.shape[2:] == raw.shape[2:]
        write_volume(output_f, pred[0, ...], prefix + "/bg")
        write_volume(output_f, pred[1, ...], prefix + "/fg")
        write_volume(output_f, pred[2, ...], prefix + "/boundaries")


def main():
    print("Read raw data list")
    raw_list = [f["volumes/raw/s1"]for f in load_platynereis_memb_n5()]
    print("Create prediction files")
    pred_list = load_platynereis_pred_n5()
    print(pred_list)
    model_path = "modelzoo/full/best/full.zip"
    prefix = "predictions/full"
    for f_raw, f_pred in zip(raw_list, pred_list):
        raw = f_raw[:]
        predict_network(raw, model_path, f_pred, prefix)

if __name__ == "__main__":
    main()