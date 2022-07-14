from pathlib import Path
import z5py
from cryofib.n5_utils import read_volume
from cryofib.preprocess_utils import percentile_norm, zero_mean_unit_variance
from stardist.models import Config3D, StarDist3D, StarDistData3D
import matplotlib
import numpy as np
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt


def main(): 
    # CUDA_VISIBLE_DEVICES=4 python predict_3d.py 
    # Load data
    scratch_dir = Path("/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem")
    fluo_dir = scratch_dir / "fluo"
    fluo_n5_path = fluo_dir / "fluo.n5"
    f_fluo = z5py.File(fluo_n5_path, "a")

    roi = np.s_[:]
    raw = read_volume(f_fluo, "raw", roi)
    raw = zero_mean_unit_variance(percentile_norm(raw, 1, 99.6))[:, :, :, np.newaxis]
    print("Raw", raw.shape)

    # Load model
    model = StarDist3D(None, name='stardist', basedir='models')

    labels, details = model.predict_instances(raw, n_tiles=(1, 16, 8, 1))

    print("Labels", labels.shape)
    print(details)

    # Store segmentation
    chunks = (1, 512, 512)
    shape = labels.shape
    compression = "gzip"
    dtype = labels.dtype

    g = f_fluo["segmentation"]
    ds_segm = g.create_dataset("stardist3D_segm", shape=shape, compression="gzip",
                                chunks=chunks, dtype=dtype)
    ds_segm.n_threads = 8
    print("Writing segmentation")
    ds_segm[:] = labels

if __name__ == "__main__":
    main()