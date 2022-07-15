from cryofib.data_loaders import load_platynereis_memb_n5
from cryofib.n5_utils import read_volume
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_erosion
import numpy as np


def main():
    f_n5_list = load_platynereis_memb_n5()
    for f in f_n5_list:
        raw = read_volume(f, "volumes/raw/s1")
        segm = read_volume(f, "volumes/labels/segmentation/s1")

        # When finding boundaries, one pixel on the border of background is also marked as boundary

        print("Find boundaries")
        boundaries = find_boundaries(segm, connectivity=segm.ndim, mode="outer")
        print("Calculate channels for prediction")
        foreground = binary_erosion((segm > 0).astype(int))
        background = 1 - foreground
        raw[background] = 0
        boundaries = boundaries * foreground
        cells = (1 - boundaries) * foreground

        labels = np.stack([background, cells, boundaries])
        print("Labels shape", labels.shape)

        if "3dunet/raw" not in f.keys():
            chunks = (1, 512, 512)
            shape = raw.shape
            compression = "gzip"
            dtype = raw.dtype
            ds = f.create_dataset("3dunet/raw", shape=shape, compression=compression,
                            chunks=chunks, dtype=dtype, n_threads=8)
        else:
            ds = f["3dunet/raw"]
        
        print(f"Writing to {f.name}, key 3dunet/raw")
        ds[:] = raw

        if "3dunet/labels" not in f.keys():
            chunks = (1, 512, 512)
            shape = labels.shape
            compression = "gzip"
            dtype = labels.dtype
            ds = f.create_dataset("3dunet/labels", shape=shape, compression=compression,
                            chunks=chunks, dtype=dtype, n_threads=8)
        else:
            ds = f["3dunet/labels"]
        
        print(f"Writing to {f.name}, key 3dunet/labels")
        ds[:] = labels


if __name__ == "__main__":
    main()