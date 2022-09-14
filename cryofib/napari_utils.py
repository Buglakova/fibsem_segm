import z5py
from pathlib import Path, PurePath
import numpy as np
import napari
from functools import partial

def napari_add_group(f: z5py.File, group_key: str, viewer: napari.Viewer):
    add_layer_viewer = partial(add_group_layer, viewer=viewer)
    f[group_key].visititems(add_layer_viewer)


def add_group_layer(name, obj, viewer=None, roi=np.s_[:]):
    assert viewer is not None 
    if isinstance(obj, z5py.dataset.Dataset):
        data = obj[roi]
        if np.issubdtype(data.dtype, np.integer):
            viewer.add_labels(data, name=name, scale=obj.attrs["resolution"])
        else:
            viewer.add_image(data, name=name, scale=obj.attrs["resolution"])