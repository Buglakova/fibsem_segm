from pathlib import Path
import numpy as np
import argparse

from skimage.measure import label

from cryofib.n5_utils import read_volume, write_volume, get_attrs

import re


def get_zero_component(img: np.ndarray):
    bg = label(img == 0)
    component_sizes = [np.count_nonzero(bg == i) for i in np.unique(bg)[1:]]
    if len(component_sizes) == 0:
        return img > 0
    bg_ind = np.argmax(component_sizes) + 1
    bg = (bg == bg_ind)
    fg = (bg != bg_ind)
    return fg


def get_fg_mask(raw: np.ndarray):
    print("Compute foreground mask")
    print("Raw data shape: ", raw.shape)
    fg_mask = np.array([get_zero_component(img) for img in raw])
    return fg_mask


def percentile_norm(img, pmin, pmax, eps=1e-10):
        return (img - pmin) / (pmax - pmin + eps)


def get_frame_i(fname, regexp=r"\d{4}"):
    """
        Get numbers of annotated frames from file names
    """
    result = re.findall(regexp, fname)
    assert len(result) == 1
    return int(result[0]) - 1


def get_annotated_frame_list(gt_dir):
        """
        Get numbers of annotated frames from file names
        """
        fnames = sorted(list(gt_dir.glob("*.tiff")))
        frames_i = [get_frame_i(fname.name) for fname in fnames]
        return frames_i


def main():
        parser = argparse.ArgumentParser(
        description="""Normalize input raw volume using percentile normalization.
        """
        )

        parser.add_argument("input_n5", type=str, help="Path of the input n5")
        parser.add_argument("input_n5_key", type=str, help="Key of the dataset in the input n5")
        parser.add_argument("output_n5", type=str, help="Path of the output n5")
        parser.add_argument("output_n5_key", type=str, help="Group in the output n5 where to write downscaled images")
        parser.add_argument("pmin", type=float, help="Min percentile")
        parser.add_argument("pmax", type=float, help="Max percentile")
        args = parser.parse_args()

        roi = np.s_[:]
        img = read_volume(args.input_n5, args.input_n5_key, roi)
        fg = get_fg_mask(img)
        attrs = dict(get_attrs(args.input_n5, args.input_n5_key))
        if "description" in attrs.keys():
            description = attrs["description"]
        else:
            description = ""

        print(f"Value range (not background): min {img[fg].min()}, max {img[fg].max()} of type {img.dtype}")
        pmin = np.percentile(img[fg], args.pmin)
        pmax = np.percentile(img[fg], args.pmax)
        print(f"Normalize using percentiles pmin {args.pmin}={pmin} and pmax {args.pmax}={pmax}")

        img_norm = percentile_norm(img, pmin, pmax).astype(np.float32)

        attrs["description"] = description + f"Normalized to percentiles pmin = {pmin}, pmax = {pmax} {args.input_n5_key}"
        write_volume(args.output_n5, img_norm, args.output_n5_key, attrs=attrs)


if __name__ == "__main__":
    main()

