import napari
import numpy as np
from elf import io

raw_folder = "/media/buglakova/embl_data/data/cryoSEM/F107a1_bin2"
boundary_folder = "/media/buglakova/embl_data/data/cryoSEM/F107a1_bin2_boundaries"
label_folder = "/media/buglakova/embl_data/data/cryoSEM/F107a1_bin2_cells"

with io.open_file(raw_folder) as f:
    raw = f["*.tiff"][:]
    print(raw.shape)
with io.open_file(boundary_folder) as f:
    boundaries = f["*.tiff"][:]
    print(np.unique(boundaries))
    boundaries = boundaries == 255
    print(boundaries.shape)
with io.open_file(label_folder) as f:
    labels = f["*.tiff"][:]
    print(np.unique(labels))
    labels = labels == 255
    print(labels.shape)

v = napari.Viewer()
v.add_image(raw)
v.add_labels(boundaries)
v.add_labels(labels)
napari.run()
