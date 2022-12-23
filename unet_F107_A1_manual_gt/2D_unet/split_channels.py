from cryofib.n5_utils import read_volume, write_volume, get_attrs
from pathlib import Path
import numpy as np
import argparse

def thick_boundary_transform(labels):
    # labels:
    # 0: no data -> map to channel 3
    # 1: boundary -> map to channel 1
    # 2: extracellular -> map to channel 2
    # >2: cells -> map to channel 0
    n_channels = 4
    out = np.zeros((n_channels,) + labels.shape, dtype="float32")

    mask = (labels == -1)
    print("Foreground")
    out[0] = (labels > 2).astype("float32")
    out[0, mask] = -1
    print("Boundaries")
    out[1] = (labels == 1).astype("float32")
    out[1, mask] = -1
    print("Extracellular")
    out[2] = (labels == 2).astype("float32")
    out[2, mask] = -1
    print("Background")
    out[3] = (labels == 0).astype("float32")
    out[3, mask] = -1

    return out


def main():
        parser = argparse.ArgumentParser(
        description="""Split manual annotations into channels for neural network prediction.
        """
        )

        parser.add_argument("input_n5", type=str, help="Path of the input n5")
        parser.add_argument("input_n5_key", type=str, help="Key of the dataset in the input n5")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_n5_key", type=str, help="Key of the output dataset")
        args = parser.parse_args()

        roi = np.s_[:]
        img = read_volume(args.input_n5, args.input_n5_key, roi)
        attrs = dict(get_attrs(args.input_n5, args.input_n5_key))

        ground_truth_channels = thick_boundary_transform(img)

        write_volume(args.output_n5, ground_truth_channels, args.output_n5_key, chunks=(img.shape[0], 1, 512, 512), attrs=attrs)


if __name__ == "__main__":
    main()
