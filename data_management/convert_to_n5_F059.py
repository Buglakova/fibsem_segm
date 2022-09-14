import numpy as np
from elf.io import open_file
import z5py
from pathlib import Path


from cryofib.n5_utils import read_volume, tif2n5, create_ind, write_volume

def rotate_x_90(array):
        """
        Edoardo:
        To register the FIBSEM data I first flip it horizontally
        and then invert the axes as XYZ —> YZX [didn't work this way] (not clear yet why Dragonfly does it this way)
        to achieve a 90° rotation around the x axis to match the fluorescence orientation
        """
        assert array.ndim == 3
        # return np.moveaxis(np.flip(array, axis=2), -1, 0)
        return np.moveaxis(np.flip(array, axis=2), 0, 1)


def main():
        data_dir = Path("/g/kreshuk/buglakova/data/cryofib/raw/F059")
        fluo_path = data_dir / "fluo" / "After-FS_Block_06_BT474_H_SirActin_organoid01_branding_confocal_from_surface_slow-white-nuclei_only.tif"
        em_path = data_dir / "em" / "F059_A1_raw"
        em_segm_nuclei_path = data_dir / "em" / "F059_A1_segm_nuclei"

        registration_dir = Path("/g/kreshuk/buglakova/data/cryofib/registration_fluo/F059")
        segmentation_dir = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F059")
        
        # Convert fluorescent data to n5
        tif2n5(tif_dir=fluo_path,
                n5_path=registration_dir / "F059_fluo.n5",
                n5_key="raw",
                reg_exp="",
                description="Raw fluorescence data F059",
                order="zyx",
                unit="nm",
                resolution=(2869.8, 332.1, 332.1),
                roi=None)


        # Convert EM data for segmentation
        em_raw, em_raw_attrs = tif2n5(tif_dir=em_path,
                n5_path=segmentation_dir / "F059_A1_em.n5",
                n5_key="raw",
                reg_exp="*.tif",
                description="Raw FIB-SEM data F059_A1",
                order="zyx",
                unit="nm",
                resolution=(20, 20, 20),
                roi=None)

        em_segm_nuclei, em_segm_nuclei_attrs = tif2n5(tif_dir=em_segm_nuclei_path,
                n5_path=segmentation_dir / "F059_A1_em.n5",
                n5_key="segmentation/nuclei",
                reg_exp="*.tiff",
                description="Nuclei segmentation done by Edoardo in Dragonfly",
                order="zyx",
                unit="nm",
                resolution=(20, 20, 20),
                roi=None)

        # For registration the volume has to be rotated
        # Edoardo:
        # To register the FIBSEM data I first flip it horizontally
        # and then invert the axes as XYZ —> YZX (not clear yet why Dragonfly does it this way)
        # to achieve a 90° rotation around the x axis to match the fluorescence orientation

        em_raw_rotated = rotate_x_90(em_raw)
        em_segm_nuclei_rotated = rotate_x_90(em_segm_nuclei)
        
        em_raw_attrs = dict(em_raw_attrs)
        em_segm_nuclei_attrs = dict(em_segm_nuclei_attrs)

        em_raw_attrs["order"] = "xzy"
        em_segm_nuclei_attrs["order"] = "xzy"

        start, stop, step = create_ind(em_raw_rotated)
        roi = {"start": start, "stop": stop, "step": step}
        em_raw_attrs['roi'] = roi
        em_segm_nuclei_attrs['roi'] = roi

        write_volume(registration_dir / "F059_A1_em.n5",
                                em_raw_rotated,
                                "raw",
                                attrs=em_raw_attrs)

        write_volume(registration_dir / "F059_A1_em.n5",
                        em_segm_nuclei_rotated,
                        "segmentation/nuclei",
                        attrs=em_segm_nuclei_attrs)


if __name__ == "__main__":
    main()