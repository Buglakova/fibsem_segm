from pathlib import Path
import h5py
from tifffile import imread

def load_trackmate_data():
    raw_path = "/g/kreshuk/buglakova/data/trackmate_spheroids/LblImg_Spheroid-3D.tif"
    labels_path = "/g/kreshuk/buglakova/data/trackmate_spheroids/LblImg_day 6 shctrl dqcol and fn - 4_XY1479384156_Z00_T0_C0.tif"
    raw = imread(raw_path)
    labels = imread(labels_path)
    return raw, labels


def load_mouse_embryo_data():
    data_path = Path("/g/kreshuk/buglakova/data/adrian_mouse_embryo/Nuclei/train")
    img_paths = list(data_path.glob("*.h5"))
    print(img_paths)

    imgs = []
    labels = []
    for ds_path in img_paths:
        with h5py.File(ds_path, "r") as f:
            print("Loading ", ds_path)
            print(f.keys())
            raw = f["raw"][:]
            label = f["label"][:]
            imgs.append(raw)
            labels.append(label)
    return imgs, labels


def load_mouse_embryo_data_val():
    data_path = Path("/g/kreshuk/buglakova/data/adrian_mouse_embryo/Nuclei/test")
    img_paths = list(data_path.glob("*.h5"))
    print(img_paths)

    imgs = []
    labels = []
    for ds_path in img_paths:
        with h5py.File(ds_path, "r") as f:
            print("Loading ", ds_path)
            print(f.keys())
            raw = f["raw"][:]
            label = f["label"][:]
            imgs.append(raw)
            labels.append(label)
    return imgs, labels