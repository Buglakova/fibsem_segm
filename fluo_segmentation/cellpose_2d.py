from n5_utils import read_volume
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose import plot, io
from cellpose.dynamics import compute_masks

import torch
from pathlib import Path
import z5py
import skimage.filters as filters


def remove_background(img):
    filtered_image = img / filters.gaussian(img, sigma=25)
    filtered_image = filtered_image - np.nanmedian(filtered_image)
    filtered_image = np.nan_to_num(filtered_image)
    return filtered_image


if __name__=="__main__":

    # Set input paths
    scratch_dir = Path("/scratch/buglakova/data/cryofib/registration_fluo/F107_A2_3dclem")
    fluo_dir = scratch_dir / "fluo"
    fluo_n5_path = fluo_dir / "fluo.n5"
    f_fluo = z5py.File(fluo_n5_path, "r")

    # Set output paths
    output_dir = Path("/scratch/buglakova/data/cryofib/segm_fluo/cellpose")
    output_n5_path = output_dir / "cellpose_2D.n5"
    f_out = z5py.File(output_n5_path, "a")

    # Read the volume
    roi = np.s_[:, : , :]
    raw = read_volume(f_fluo, "raw", roi)

    filtered_raw = np.array([remove_background(img) for img in raw])

    # PyTorch device
    # Set device
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device(4)
        print(device)
    else:
        print("GPU is not available")
        device = torch.device("cpu")


    # model_type='cyto' or 'nuclei' or 'cyto2'
    model = models.Cellpose(model_type='nuclei', gpu=True, device=device)

    nimg = len(raw)

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    channels = [[0,0]]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended)
    # diameter can be a list or a single number for all images

    # print("Raw shape", raw.shape)
    # masks, flows, styles, diams = model.eval(raw, diameter=None, channels=channels, z_axis=0)

    # print("Masks: ", masks.shape, type(masks))
    # # print("Flows: ", flows.shape, type(flows))

    
    # chunks = (1, 512, 512)
    # shape = np.array(masks).shape
    # compression = "gzip"
    # dtype = np.array(masks).dtype
    # print("Segmentation type: ", dtype)

    # n5_key = "cellpose"
    # # ds = f_out.create_dataset(n5_key, shape=shape, compression=compression,
    # #                     chunks=chunks, dtype=dtype, n_threads=8)
    # print(f"Writing to {output_n5_path}, key {n5_key}")
    # # ds[:] = np.array(masks)

    # print("Saving cellpose examples")
    # nimg = raw.shape[0]
    # for idx in range(0, nimg, 5):
    #     maski = masks[idx]
    #     flowi = flows[idx][0]

    #     fig = plt.figure(figsize=(12,5))
    #     plot.show_segmentation(fig, raw[idx, :, :], maski, flowi, channels=channels)
    #     plt.tight_layout()
    #     plt.savefig(output_dir / f"{idx}_fluo_cellpose_segm.png")

    # print("Saving cellpose output")
    # io.masks_flows_to_seg(raw, 
    #                   masks, 
    #                   flows, 
    #                   diams, 
    #                   output_dir / "fluo_seg.npy", 
    #                   channels)



    masks, flows, styles, diams = model.eval(filtered_raw, diameter=None, channels=channels, z_axis=0)

    chunks = (1, 512, 512)
    shape = np.array(masks).shape
    compression = "gzip"
    dtype = np.array(masks).dtype
    print("Segmentation type: ", dtype)

    n5_key = "cellpose_background"
    ds = f_out.create_dataset(n5_key, shape=shape, compression=compression,
                        chunks=chunks, dtype=dtype, n_threads=8)
    print(f"Writing to {output_n5_path}, key {n5_key}")
    ds[:] = np.array(masks)

    print("Saving cellpose examples")
    nimg = raw.shape[0]
    for idx in range(0, nimg, 5):
        maski = masks[idx]
        flowi = flows[idx][0]

        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, raw[idx, :, :], maski, flowi, channels=channels)
        plt.tight_layout()
        plt.savefig(output_dir / f"{idx}_fluo_cellpose_background_segm.png")

    print("Saving cellpose output")
    io.masks_flows_to_seg(raw, 
                      masks, 
                      flows, 
                      diams, 
                      output_dir / "fluo_background_seg.npy", 
                      channels)