import z5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening
from skimage.segmentation import find_boundaries, watershed
from elf.segmentation.utils import normalize_input

from cryofib.n5_utils import read_volume, write_volume, get_attrs
import vigra


def bbox_3D(img, margin=10):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return max(rmin - margin, 0), min(rmax + margin, img.shape[0]), max(cmin - margin, 0), min(cmax + margin, img.shape[1]), max(zmin - margin, 0), min(zmax + margin, img.shape[2])



def main():
    # Spread ground truth instances to boundary parts, except where raw is close to 0
        
    train_multicut_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_train_multicut.n5")
    postprocess_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_postprocess.n5")
    
    roi = np.s_[:, :, :]
    # raw = read_volume(train_multicut_path, "input/raw", roi)
    # write_volume(postprocess_path, raw, "input/raw")
    extra = read_volume(train_multicut_path, "predictions/extra", roi)
    boundaries = read_volume(train_multicut_path, "predictions/boundaries", roi)
    fg_mask = read_volume(train_multicut_path, "masks/fg_mask", roi)
    # extra_mask = read_volume(train_multicut_path, "masks/extra_mask", roi)
    multicut = read_volume(postprocess_path, "segmentation/extra", roi)
    
    # Assign extracellular
    result = multicut
    print(result.dtype)
    result = result + 2
    print(result.dtype)
    print("assign fg mask")
    print("fg mask", fg_mask.dtype)
    fg_mask = fg_mask > 0
    result = result * fg_mask
    print(result.dtype)
    # print("assign extra mask")
    # result = result * (1 - extra_mask)
    # result = result + extra_mask
    # result = result.astype(np.uint64)

    max_id = int(np.max(result))
    print(f"Max id {max_id}")
    print(type(max_id))
    areas = []
    nonzero_id = 3
    for seg_id in range(3, max_id + 1):
        print(f"{seg_id} out of {max_id}")
        seg_mask = result == seg_id
        area = len(seg_mask[seg_mask])
        print(area)
        if area > 0:
            areas.append(area)
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(seg_mask)
            print(rmin, rmax, cmin, cmax, zmin, zmax)
            id_extra = extra[rmin:rmax, cmin:cmax, zmin:zmax]
            # id_boundaries = boundaries[rmin:rmax, cmin:cmax, zmin:zmax]
            # id_multicut = result[rmin:rmax, cmin:cmax, zmin:zmax]
            id_mask = seg_mask[rmin:rmax, cmin:cmax, zmin:zmax]
            print("Bbox shape", id_mask.shape)
            
            if np.median(id_extra[id_mask]) > 0.6:
                result[seg_mask] = 2
                print("Extracellular")
                
        if seg_id % 100 == 0:
            write_volume(postprocess_path, result.astype(np.uint32), "segmentation/extra_all")
        
    write_volume(postprocess_path, result.astype(np.uint32), "segmentation/extra_all")
    plt.hist(areas, bins=100)
    plt.savefig("areas.png", dpi=300)
    # Write resulting segmentation
    # write_volume(train_multicut_path, cell_instances, "input/gt_instance_no_boundaries")

    
if __name__ == "__main__":
    main()