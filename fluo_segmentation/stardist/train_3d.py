from stardist.models import Config3D, StarDist3D, StarDistData3D
import matplotlib
import numpy as np
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
from cryofib.data_loaders import load_mouse_embryo_data, load_mouse_embryo_data_val, load_trackmate_data
from csbdeep.utils import Path, normalize
from cryofib.preprocess_utils import percentile_norm, zero_mean_unit_variance
from stardist import Rays_GoldenSpiral


def random_fliprot(img, mask, axis=None): 
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y



def main():
    print("Train 3D Stardist model")

    print("Read data")
    imgs, labels = load_mouse_embryo_data()
    img_trackmate, label_trackmate = load_trackmate_data()

    imgs += [img_trackmate]
    labels += [label_trackmate]

    print("Normalize raw")
    imgs = [zero_mean_unit_variance(percentile_norm(img, 1, 99.6))[:, :, :, np.newaxis] for img in imgs]

    print("Load and normalize validation data")
    imgs_val, labels_val = load_mouse_embryo_data_val()
    imgs_val = [zero_mean_unit_variance(percentile_norm(img, 1, 99.6))[:, :, :, np.newaxis] for img in imgs_val]

    print("Set training parameters")
    print(Config3D.__doc__)

    n_channel_in = 1
    anisotropy = [1.85, 0.2, 0.2]
    n_rays = 96
    use_gpu = True
    grid = [1, 2, 2]
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D (
    rays             = rays,
    grid             = grid,
    anisotropy       = anisotropy,
    use_gpu          = False,
    n_channel_in     = n_channel_in,
    # adjust for your data below (make patch size as large as possible)
    train_patch_size = (48,96,96),
    train_batch_size = 2,
    )
    print(conf)
    vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        # Total memory: Total amount of available GPU memory (in MB)
        limit_gpu_memory(0.8, total_memory=11264)

    print("Create model")
    model = StarDist3D(conf, name='stardist', basedir='models')

    print("Start training")
    model.train(imgs, labels, validation_data=(imgs_val, labels_val), augmenter=augmenter)

    print("Optimize threshold")
    model.optimize_thresholds(imgs_val, labels_val)
    

if __name__ == "__main__":
    main()