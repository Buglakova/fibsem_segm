import os
import bioimageio.core
import numpy as np
import xarray as xr
import argparse

from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.prediction import predict_with_tiling
from bioimageio.core.resource_tests import test_model

from cryofib.n5_utils import read_volume, write_volume, get_attrs
from cryofib.preprocess_utils import percentile_norm, zero_mean_unit_variance


def test_model_resource(model_resource):
    print("Test model")
    test_result = test_model(model_resource)
    if test_result["status"] == "failed":
        print("model test:", test_result["name"])
        print("The model test failed with:", test_result["error"])
        print("with the traceback:")
        print("".join(test_result["traceback"]))
    else:
        test_result["status"] == "passed"
        print("The model passed all tests")


def load_loyalsquid_model():
    rdf_doi = "10.5281/zenodo.6383429"
    rdf_path = "/g/kreshuk/buglakova/projects/fibsem_segm/fluo_segmentation/models/3d-unet-mouse-embryo-fixed_pytorch_state_dict.zip"
    model_resource = bioimageio.core.load_resource_description(rdf_path)
    

    print("Available weight formats for this model:", model_resource.weights.keys())
    print("Pytorch state dict weights are stored at:", model_resource.weights["pytorch_state_dict"].source)
    print()
    print("The model requires as inputs:")
    for inp in model_resource.inputs:
        print("Input with axes:", inp.axes, "and shape", inp.shape)
    print()
    print("The model returns the following outputs:")
    for out in model_resource.outputs:
        print("Output with axes:", out.axes, "and shape", out.shape)

    test_model_resource(model_resource)

    return model_resource

    
def predict_numpy(model, input_, devices=[0], weight_format=None):
    pred_pipeline = create_prediction_pipeline(
        bioimageio_model=model, devices=devices, weight_format=weight_format
    )

    axes = tuple(model.inputs[0].axes)
    input_tensor = xr.DataArray(input_, dims=axes)
    
    tiling = {"halo": {"x": 32, "y": 32, "z": 8}, "tile": {"x": 128, "y": 128, "z": 32}}
    prediction = predict_with_tiling(pred_pipeline, input_tensor, tiling)[0]
    return prediction



def main():
    parser = argparse.ArgumentParser(
        description="""Run nuclei segmentation for single-channel DAPI-stained organoids using 10.5281/zenodo.6383429.
        """
    )
    parser.add_argument("input_n5", type=str, help="Path of the input n5")
    parser.add_argument("input_n5_key", type=str, help="Key of the dataset in the input n5")
    parser.add_argument("output_n5", type=str, help="Path of the output n5")
    parser.add_argument("output_n5_key", type=str, help="Key of the prediction in the output n5")
    args = parser.parse_args()

    os.environ["BIOIMAGEIO_CACHE_PATH"] = "/g/kreshuk/buglakova/biomodel_zoo_tmp/"

    model_resource = load_loyalsquid_model()


    roi = np.s_[:]
    raw = read_volume(args.input_n5, args.input_n5_key, roi)
    raw = zero_mean_unit_variance(percentile_norm(raw, 1, 99.6))
    raw = raw[np.newaxis, np.newaxis, :, :, :]

    prediction = predict_numpy(model_resource, raw.astype(np.float32))

    prediction_attrs = get_attrs(args.input_n5, args.input_n5_key)
    prediction_attrs["description"] = "Nuclei segmentation with loyal squid network \
            after (1, 99.6) percentile normalization and zero mean unit variance normalization"

    write_volume(args.output_n5, prediction[0, 0, :, :, :], args.output_n5_key + "/nuclei", attrs=prediction_attrs)
    write_volume(args.output_n5, prediction[0, 1, :, :, :], args.output_n5_key + "/boundaries", attrs=prediction_attrs)

if __name__=="__main__":
    main()