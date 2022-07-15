from pathlib import Path
import h5py
from tifffile import imread
import z5py

def load_trackmate_data():
    raw_path = "/g/kreshuk/buglakova/data/trackmate_spheroids/Spheroid-3D.tif"
    labels_path = "/g/kreshuk/buglakova/data/trackmate_spheroids/LblImg_Spheroid-3D.tif"
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


def load_platynereis_memb_n5():
    """
        Get list of opened n5 files
    """
    data_dir = Path("/g/kreshuk/buglakova/data/platynereis_em_membranes/membrane")
    n5_paths = list(data_dir.glob("*.n5"))
    f_n5_list = [z5py.File(n5_path, "a") for n5_path in n5_paths]
    return f_n5_list