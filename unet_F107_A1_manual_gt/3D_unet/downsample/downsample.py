from pathlib import Path
import numpy as np
import argparse

from skimage.transform import downscale_local_mean, rescale
from skimage.morphology import cube

from cryofib.n5_utils import read_volume, write_volume, get_attrs


def main():
        parser = argparse.ArgumentParser(
        description="""Downsample input volume by factors of 2 and 4 using skimage.transform.rescale.
        """
        )

        parser.add_argument("input_n5", type=str, help="Path of the input n5")
        parser.add_argument("input_n5_key", type=str, help="Key of the dataset in the input n5")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_n5_key", type=str, help="Group in the output n5 where to write downscaled images")
        parser.add_argument("rescale_order", type=int, help="Interpolation order")
        args = parser.parse_args()

        roi = np.s_[:]
        img = read_volume(args.input_n5, args.input_n5_key, roi)
        attrs = dict(get_attrs(args.input_n5, args.input_n5_key))
        description = attrs["description"]
        

        if img.ndim > 3:
            print("Image has more than 3 dims, using axis 0 as channel axis")
            channel_axis = 0
            chunks = (img.shape[0], 1, 512, 512)
        else:
            channel_axis = None
            chunks = (1, 512, 512)

        print("Downsample by a factor of 2")
        img_bin2 = rescale(img, 0.5, order=args.rescale_order, anti_aliasing=False, channel_axis=channel_axis)
        print("Downsample by a factor of 4")
        img_bin4 = rescale(img, 0.25, order=args.rescale_order, anti_aliasing=False, channel_axis=channel_axis)

        attrs["description"] = description + f" Downscaled by a factor of 2 without anti-aliasing, interpolation order {args.input_n5_key}"
        write_volume(args.output_n5, img_bin2, args.output_n5_key + "/s1", attrs=attrs, chunks=chunks)
        attrs["description"] = description + f" Downscaled by a factor of 4 without anti-aliasing, interpolation order {args.input_n5_key}"
        write_volume(args.output_n5, img_bin4, args.output_n5_key + "/s2", attrs=attrs, chunks=chunks)



if __name__ == "__main__":
    main()

