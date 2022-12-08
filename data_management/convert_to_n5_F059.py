import numpy as np
from elf.io import open_file
import z5py
from pathlib import Path


from cryofib.n5_utils import read_volume, tif2n5, create_ind, write_volume


def main():
        data_dir = Path("/g/kreshuk/buglakova/data/cryofib/raw/F059")
        fluo_path = data_dir / "fluo" / "After-FS_Block_06_BT474_H_SirActin_organoid01_branding_confocal_from_surface_slow-white-nuclei_only.tif"
        em_path = data_dir / "em" / "F059_A1_raw"
        em_segm_nuclei_path = data_dir / "em" / "F059_A1_segm_nuclei"

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


if __name__ == "__main__":
    main()