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

        # s1, because it's the pixel size of the segmentation
        # Segmentation is the instances
        # Only square area in the middle is labeled, so zero out everything outside it in raw and use as out-of-sample part
        # When finding boundaries, one pixel on the border of background is also marked as boundary, so final area should be one pixel smaller
        # Ch 0 - "out-of-sample" background
        # Ch 1 - "foreground" (1 - boundaries)boundaries
        # Ch 2 - boundaries
        

        print("Find boundaries")
        boundaries = find_boundaries(segm, connectivity=segm.ndim, mode="outer")
        print("Calculate channels for prediction")
        print("erosion")
        foreground = binary_erosion((segm > 0).astype(int))
        print("get background")
        background = 1 - foreground
        print("zero out background in raw")
        # raw[background] = 0
        raw_new = raw * foreground
        print("get boundaries and cells")
        boundaries = boundaries * foreground
        cells = (1 - boundaries) * foreground

        labels = np.stack([background, cells, boundaries])
        print("Labels shape", labels.shape)

        if "3dunet/raw" not in f.keys():
            chunks = (1, 512, 512)
            shape = raw_new.shape
            compression = "gzip"
            dtype = raw_new.dtype
            ds = f.create_dataset("3dunet/raw", shape=shape, compression=compression,
                            chunks=chunks, dtype=dtype, n_threads=8)
        else:
            ds = f["3dunet/raw"]
        
        print(f"Writing to {f.name}, key 3dunet/raw")
        ds[:] = raw_new

        if "3dunet/labels" not in f.keys():
            chunks = (3, 1, 512, 512)
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