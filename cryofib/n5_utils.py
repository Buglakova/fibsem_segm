import z5py
from pathlib import Path, PurePath
import numpy as np
from typing import Tuple, Dict
from elf.io import open_file



def print_key_tree(f: z5py.File):
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "r")
    print(f"Key structure of z5 file {f.filename}")
    f.visititems(lambda name, obj: print(name))


def read_volume(f: z5py.File, key: str, roi: np.lib.index_tricks.IndexExpression = np.s_[:]):
    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "r")

    try:
        ds = f[key]
    except KeyError:
        print(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None
        
    ds.n_threads = 8
    print(f"Reading roi {roi} of volume {key} from {f.filename}")
    vol = ds[roi]
    print(f"Read volume with shape {vol.shape}, data type {vol.dtype}")
    
    return vol

def get_attrs(f: z5py.File, key: str):
    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    try:
        ds = f[key]
    except KeyError:
        print(f"No key {key} in file {f.filename}")
        print_key_tree(f)
        return None
    
    return ds.attrs


def create_ind(arr: np.array):
    stop = arr.shape
    start = [0] * len(stop)
    step = [1] * len(stop)
    return start, stop, step


def read_ind(start: Tuple, stop: Tuple, step: Tuple) -> np.lib.index_tricks.IndexExpression:
    ind_exp = tuple(slice(*slice_param) for slice_param in zip(start, stop, step))
    return ind_exp


def write_ind(ind_exp: np.lib.index_tricks.IndexExpression):
    start = tuple(sl.start for sl in ind_exp)
    stop = tuple(sl.stop for sl in ind_exp)
    step = tuple(sl.step for sl in ind_exp)
    return start, stop, step


def write_volume(f, arr: np.array, key, chunks=(1, 512, 512), attrs=None):
    shape = arr.shape
    compression = "gzip"
    dtype = arr.dtype

    print(type(f))
    if isinstance(f, (str, PurePath)):
        f = z5py.File(f, "a")

    if key not in f.keys():
        print(f"Created dataset {key}")
        ds = f.create_dataset(key, shape=shape, compression=compression,
                                chunks=chunks, dtype=dtype)
    else:
        print(f"Overwriting {key}")
        ds = f[key]
    
    ds.n_threads = 8
    print(f"Writing array to {key}")
    ds[:] = arr

    if attrs is not None:
        print(f"Assigning attributes of the dataset")
        for key in attrs.keys():
            ds.attrs[key] = attrs[key]


def tif2n5(tif_dir: Path,
            n5_path: Path,
            n5_key: str,
            reg_exp: str = "*.tiff",
            description: str = "",
            order: str = "zyx",
            unit: str = "nm",
            resolution: Tuple = (1, 1, 1),
            roi: Dict = None):

    f_out = z5py.File(n5_path, "a")
    with open_file(str(tif_dir)) as f:
        print(f"Reading tif files from {tif_dir}")
        tiff_stack_obj = f[reg_exp]
        imgs = tiff_stack_obj[:]
        print(f"Read {tiff_stack_obj.shape} of type {tiff_stack_obj.dtype}")

        chunks = (1, 512, 512)
        shape = imgs.shape
        compression = "gzip"
        dtype = imgs.dtype

        if n5_key not in f_out.keys():
            print(f"Created dataset {n5_key}")
            ds = f_out.create_dataset(n5_key, shape=shape, compression=compression,
                                chunks=chunks, dtype=dtype, n_threads=8)

        else:
            print(f"Overwriting {n5_key}")
            ds = f_out[n5_key]

        print(f"Writing to {n5_path}, n5_key {n5_key}")
        ds[:] = imgs

        # Attributes required for future registration/viewing
        print(f"Assigning attributes of the dataset")
        attributes = ds.attrs
        attributes["description"] = description
        attributes["unit"] = unit
        attributes["resolution"] = resolution
        if roi is None:
            start, stop, step = create_ind(imgs)
            roi = {"start": start, "stop": stop, "step": step}
        attributes['roi'] = roi

    return imgs, attributes
