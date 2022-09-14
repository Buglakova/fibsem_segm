import numpy as np
from elf.io import open_file
import z5py
from pathlib import Path
import re
from skimage.io import imread


from cryofib.n5_utils import read_volume, tif2n5, create_ind, write_volume

def match_transpose(array):
        """
        Edoardo: F107_A2 data transposed to match fluorescence coordinates (xyz --> zxy).
        In fact it was xyz --> yzx, so will try like this.
        """
        assert array.ndim == 3
        return np.moveaxis(array, 0, -1)

def match_transpose_rev(array):
        """
        Edoardo: F107_A2 data transposed to match fluorescence coordinates (xyz --> zxy).
        In fact it was xyz --> yzx, so will try like this.
        Reverse to do segmentation with the correct position of Z axis.
        """
        assert array.ndim == 3
        return np.moveaxis(array, -1, 0)

# def replace_parts(em_raw, em_segm):
#         """
#         Ground truth is given as a set of tiffs from different parts of stack.
#         Create volume same shape as raw with unannotated parts labeled -1.
#         Magical constants:
#         Frames 1 to 9 are 618 to 626  
#         Frame 0 is 566
#         """
#         em_segm = em_segm.astype(np.int32)
#         em_annotation = -1 * np.ones(em_raw.shape, dtype=np.int32)
#         em_annotation[566, ...] = em_segm[0, ...]
#         em_annotation[618:627, ...] = em_segm[1:, ...]
#         return em_annotation


def get_frame_i(fname, regexp=r"\d{4}"):
    result = re.findall(regexp, fname)
    assert len(result) == 1
    return int(result[0]) - 1


def replace_parts(em_raw, gt_dir):
        """
        Get numbers of annotated frames from file names
        """
        fnames = sorted(list(gt_dir.glob("*.tiff")))
        frames_i = [get_frame_i(fname.name) for fname in fnames]
        print("Frame numbers: ", frames_i)
        segmentations = [imread(fname) for fname in fnames]
        print("Assign annotations to corresponding frames")
        em_annotation = -1 * np.ones(em_raw.shape, dtype=np.int32)
        for i, seg in zip(frames_i, segmentations):
                print(i)
                em_annotation[i, ...] = seg

        return em_annotation

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
        data_dir = Path("/g/kreshuk/buglakova/data/cryofib/raw/F107")
        fluo_path = data_dir / "fluo" / "F107_raw"
        em_path = data_dir / "em"

        registration_dir = Path("/g/kreshuk/buglakova/data/cryofib/registration_fluo/F107")
        segmentation_dir = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F107")
        
        # A bit more accurate commenting out
        convert_fluo = False
        convert_A1 = True
        convert_A1_annotated = True
        convert_A2 = False


        # Convert fluorescent data to n5
        if convert_fluo:
                tif2n5(tif_dir=fluo_path,
                        n5_path=registration_dir / "F107_fluo.n5",
                        n5_key="raw",
                        reg_exp="*.tiff",
                        description="Raw fluorescence data F107",
                        order="zyx",
                        unit="nm",
                        resolution=(1849.5, 221.4, 221.4),
                        roi=None)


        # Convert EM A1 data
        if convert_A1:
                em_raw, em_raw_attrs = tif2n5(tif_dir=em_path / "F107_A1_raw/",
                        n5_path=segmentation_dir / "F107_A1_em.n5",
                        n5_key="raw",
                        reg_exp="*.tiff",
                        description="Raw FIB-SEM data F107_A1",
                        order="zyx",
                        unit="nm",
                        resolution=(40, 30, 30),
                        roi=None)

                # em_segm_nuclei, em_segm_nuclei_attrs = tif2n5(tif_dir=em_path / "F107_A1_segm_nuclei",
                #         n5_path=segmentation_dir / "F107_A1_em.n5",
                #         n5_key="segmentation/nuclei",
                #         reg_exp="*.tiff",
                #         description="Nuclei segmentation done by Edoardo in Dragonfly, manually corrected",
                #         order="zyx",
                #         unit="nm",
                #         resolution=(40, 30, 30),
                #         roi=None)

                # For registration volume has to be rotated

                # em_raw_rotated = match_transpose(em_raw)
                # em_segm_nuclei_rotated = match_transpose(em_segm_nuclei)
                
                # em_raw_attrs = dict(em_raw_attrs)
                # em_segm_nuclei_attrs = dict(em_segm_nuclei_attrs)

                # em_raw_attrs["order"] = "yxz"
                # em_segm_nuclei_attrs["order"] = "yxz"

                # em_raw_attrs["resolution"] = (30, 30, 40)
                # em_segm_nuclei_attrs["resolution"] = (30, 30, 40)

                # start, stop, step = create_ind(em_raw_rotated)
                # roi = {"start": start, "stop": stop, "step": step}
                # em_raw_attrs['roi'] = roi
                # em_segm_nuclei_attrs['roi'] = roi

                # write_volume(registration_dir / "F107_A1_em.n5",
                #                         em_raw_rotated,
                #                         "raw",
                #                         attrs=em_raw_attrs)

                # write_volume(registration_dir / "F107_A1_em.n5",
                #                 em_segm_nuclei_rotated,
                #                 "segmentation/nuclei",
                #                 attrs=em_segm_nuclei_attrs)

                # tif2n5(tif_dir=em_path / "F107_A1_boundaries",
                #         n5_path=segmentation_dir / "F107_A1_em.n5",
                #         n5_key="segmentation/edoardo/boundaries",
                #         reg_exp="*.tiff",
                #         description="Boundary segmentation done by Edoardo in Dragonfly with 2.5D U-Net",
                #         order="zyx",
                #         unit="nm",
                #         resolution=(40, 30, 30),
                #         roi=None)

                # tif2n5(tif_dir=em_path / "F107_A1_foreground",
                #         n5_path=segmentation_dir / "F107_A1_em.n5",
                #         n5_key="segmentation/edoardo/foreground",
                #         reg_exp="*.tiff",
                #         description="Cell segmentation ('not boundary and not out of sample') done by Edoardo in Dragonfly with 2.5D U-Net",
                #         order="zyx",
                #         unit="nm",
                #         resolution=(40, 30, 30),
                #         roi=None)

                if convert_A1_annotated:
                        em__annotated_raw, em_annotated_raw_attrs = tif2n5(tif_dir=em_path / "F107_A1_gt_corrected" / "raw",
                                n5_path=segmentation_dir / "F107_A1_em_gt.n5",
                                n5_key="raw",
                                reg_exp="*.tiff",
                                description="Raw FIB-SEM data F107_A1",
                                order="zyx",
                                unit="nm",
                                resolution=(40, 30, 30),
                                roi=None)

                        # em_segm, em_segm_attrs = tif2n5(tif_dir=em_path / "F107_A1_gt_corrected" / "segmentation",
                        #         n5_path=segmentation_dir / "F107_A1_em_gt.n5",
                        #         n5_key="segmentation/ground_truth",
                        #         reg_exp="*.tiff",
                        #         description="Ground truth segmentation done by Edoardo in Dragonfly, manually corrected",
                        #         order="zyx",
                        #         unit="nm",
                        #         resolution=(40, 30, 30),
                        #         roi=None)

                        # ground_truth = replace_parts(em_raw, em_path / "F107_A1_gt_corrected" / "segmentation").astype(np.int64)

                        # em_segm_attrs = dict(em_segm_attrs)
                        # start, stop, step = create_ind(ground_truth)
                        # roi = {"start": start, "stop": stop, "step": step}
                        # em_segm_attrs['roi'] = roi

                        # write_volume(segmentation_dir / "F107_A1_em.n5",
                        #         ground_truth,
                        #         "segmentation/ground_truth",
                        #         attrs=em_segm_attrs)

                        # print("Split into channels")
                        # ground_truth_channels = thick_boundary_transform(ground_truth)

                        # write_volume(segmentation_dir / "F107_A1_em.n5",
                        #         ground_truth_channels,
                        #         "segmentation/ground_truth_channels",
                        #         chunks=(4, 1, 512, 512),
                        #         attrs=em_segm_attrs)


        if convert_A2:
              
                em_raw, em_raw_attrs = tif2n5(tif_dir=em_path / "F107_A2_raw/",
                        n5_path=registration_dir / "F107_A2_em.n5",
                        n5_key="raw",
                        reg_exp="*.tiff",
                        description="Raw FIB-SEM data F107_A2",
                        order="yxz",
                        unit="nm",
                        resolution=(30, 30, 40),
                        roi=None)

                em_segm_nuclei, em_segm_nuclei_attrs = tif2n5(tif_dir=em_path / "F107_A2_segm_nuclei",
                        n5_path=registration_dir / "F107_A2_em.n5",
                        n5_key="segmentation/nuclei",
                        reg_exp="*.tiff",
                        description="Nuclei segmentation done by Edoardo in Dragonfly",
                        order="yxz",
                        unit="nm",
                        resolution=(30, 30, 40),
                        roi=None)

                # For registration the volume was rotated, so has to be rotated back to have correct axis order in segmentation

                em_raw_rotated = match_transpose_rev(em_raw)
                em_segm_nuclei_rotated = match_transpose_rev(em_segm_nuclei)

                em_raw_attrs = dict(em_raw_attrs)
                em_segm_nuclei_attrs = dict(em_segm_nuclei_attrs)

                em_raw_attrs["resolution"] = (40, 30, 30)
                em_segm_nuclei_attrs["resolution"] = (40, 30, 30)

                em_raw_attrs["order"] = "zyx"
                em_segm_nuclei_attrs["order"] = "zyx"

                start, stop, step = create_ind(em_raw_rotated)
                roi = {"start": start, "stop": stop, "step": step}
                em_raw_attrs['roi'] = roi
                em_segm_nuclei_attrs['roi'] = roi

                write_volume(segmentation_dir / "F107_A2_em.n5",
                                        em_raw_rotated,
                                        "raw",
                                        attrs=em_raw_attrs)

                write_volume(segmentation_dir / "F107_A2_em.n5",
                                em_segm_nuclei_rotated,
                                "segmentation/nuclei",
                                attrs=em_segm_nuclei_attrs)


if __name__ == "__main__":
    main()