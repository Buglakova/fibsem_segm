from pathlib import Path
import h5py
from tifffile import imread
import z5py
from cryofib.n5_utils import print_key_tree, read_volume

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
        Get list of n5 file handlers
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/platynereis_em_membranes/membrane")
    data_dir = Path("/scratch/buglakova/data/platynereis_em_membranes/membrane")
    n5_paths = list(data_dir.glob("train*.n5"))
    n5_paths.sort()
    f_n5_list = [z5py.File(n5_path, "a") for n5_path in n5_paths]
    return f_n5_list


def load_platynereis_memb_ds():
    """
        Get list of datasets inside n5 files
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/platynereis_em_membranes/membrane")
    data_dir = Path("/scratch/buglakova/data/platynereis_em_membranes/membrane")
    n5_paths = list(data_dir.glob("train*.n5"))
    n5_paths.sort()
    f_n5_list = [z5py.File(n5_path, "a") for n5_path in n5_paths]
    raw_list = [f["3dunet/raw"] for f in f_n5_list]
    labels_list = [f["3dunet/labels"] for f in f_n5_list]
    return raw_list, labels_list


def load_platynereis_pred_n5():
    """
        Get list of n5 file handlers
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/platynereis_em_membranes/membrane")
    data_dir = Path("/scratch/buglakova/data/platynereis_em_membranes/membrane")
    n5_paths = list(data_dir.glob("*.n5"))
    n5_paths.sort()
    f_n5_list = [z5py.File(n5_path.parent / "predictions" / f"prediction_membrane_0{i}.n5", "a") for i, n5_path in enumerate(n5_paths)]
    return f_n5_list


def load_F107_A1_n5():
    """
        Get n5 file handler
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    n5_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    f_n5= z5py.File(n5_path, "a")
    return f_n5


def load_F107_A1_raw():
    """
        Get raw volume
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    n5_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    f_n5= z5py.File(n5_path, "a")
    raw = read_volume(f_n5, "raw")
    return raw


def load_F107_A1_raw_norm():
    """
        Get raw volume
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    n5_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    f_n5= z5py.File(n5_path, "a")
    raw = read_volume(f_n5, "raw_norm")
    return raw


def load_F107_A2_raw():
    """
        Get raw volume
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    n5_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A2_em.n5/")
    f_n5= z5py.File(n5_path, "a")
    raw = read_volume(f_n5, "raw")
    return raw


def load_F107_A1_pred():
    """
        Get n5 file handler
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    n5_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em_3Dunet.n5/")
    f_n5= z5py.File(n5_path, "a")
    return f_n5


def load_F107_A2_pred():
    """
        Get n5 file handler
    """
    # data_dir = Path("/g/kreshuk/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5/")
    n5_path = Path("/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A2_em_3Dunet.n5/")
    f_n5= z5py.File(n5_path, "a")
    return f_n5