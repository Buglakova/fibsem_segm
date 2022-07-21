import argparse
import numpy as np
import torch.nn as nn
import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer, _check_plt
from torch_em.transform.raw import get_default_raw_augmentations
from torch_em.transform.augmentation import get_augmentations, RandomElasticDeformation3D
import torch

from torch.cuda import mem_get_info

from pathlib import Path
import z5py
import matplotlib.pyplot as plt

def check_data(data_paths, data_key, label_paths, label_key, rois):
    print("Loading the raw data from:", data_paths, data_key)
    print("Loading the labels from:", label_paths, label_key)
    try:
        ds = torch_em.default_segmentation_dataset(data_paths, data_key, label_paths, label_key, patch_shape, rois=rois, with_label_channels=True, is_seg_dataset=True)
        print("Raw dtype", ds[0][0].dtype)
        print("Label shape", ds[0][1].shape)
        print("Label dtype", ds[0][1].dtype)
    except Exception as e:
        print("Loading the dataset failed with:")
        raise e

    return ds


class ApplyMask:
    def __call__(self, prediction, target, mask_const=-1):
        assert target.dim() == prediction.dim(), f"{target.dim()}, {prediction.dim()}"
        assert target.shape[2:] == prediction.shape[2:], f"{str(target.shape)}, {str(prediction.shape)}"

        mask = target != mask_const
        mask.requires_grad = False

        # mask the prediction
        # print("Prediction range", prediction.min(), prediction.max())
        prediction = prediction * mask
        
        target = target * mask
        # print("Target range", target.min(), target.max())

        # with open("metrics.txt", "a") as f:
        #     f.write("pred " + str(prediction.min()) + str(prediction.max()))
        #     f.write(r"/n target " + str(target.min()) + str(target.max()))
        return prediction, target


def get_data_paths():
    data_dir = Path("/g/kreshuk/buglakova/data/platynereis_em_membranes/membrane")
    n5_paths = [str(path) for path in data_dir.glob("*.n5")]
    n5_paths.sort()
    train_data_paths = n5_paths[:7]
    print("Train datasets: ", len(train_data_paths))
    val_data_paths = n5_paths[7:]
    print("Validation datasets: ", val_data_paths)
    data_key = "3dunet/raw"
    label_key = "3dunet/labels"
    return train_data_paths, val_data_paths, data_key, label_key


if __name__ == "__main__":

    # Set paths to the train and test data
    # Dataset consists of 9 labeled stacks => use 01 - 07 for training
    # 08, 09 for test
    train_data_paths, val_data_paths, data_key, label_key = get_data_paths()

    experiment_name = "test"

    # Set parameters of the network
    # patch_shape = (32, 256, 256)
    # Suggested by Constantine
    
    patch_shape = (64, 128, 128)
    assert len(patch_shape) == 3

    # # Loss, metric, batch size
    batch_size = 4
    loss = "dice"
    metric = "dice"

    # Roi: whole dataset
    train_rois = None
    val_rois = None

    # Check inputs
    print("Checking the training dataset:")
    train_ds = check_data(train_data_paths, data_key, train_data_paths, label_key, train_rois)
    print("/n/n/n")
    print("Train dataset")
    print(len(train_ds.datasets[0]))
    print("Length", len(train_ds))
    print(train_ds[0][0].shape)
    val_ds = check_data(val_data_paths, data_key, val_data_paths, label_key, val_rois)

    raw_transforms = get_default_raw_augmentations()
    transforms = ["RandomHorizontalFlip3D", "RandomVerticalFlip3D", "RandomDepthicalFlip3D"]
    transforms = get_augmentations(ndim=3, transforms=transforms)

    print("Create data loaders")
    train_loader = torch_em.default_segmentation_loader(
        train_data_paths, data_key, train_data_paths, label_key,
        batch_size, patch_shape, with_label_channels=True, raw_transform=raw_transforms,
        transform=transforms
    )
    val_loader = torch_em.default_segmentation_loader(
        train_data_paths, data_key, train_data_paths, label_key,
        batch_size, patch_shape, with_label_channels=True
    )

    print("Plot several samples")
    # fig = _check_plt(train_loader, 5, False)
    # fig.tight_layout()
    # fig.savefig("train_loader_examples.png", dpi=300)

    x, y = None, None
    for i, (patch, labels) in zip([0], val_loader):
        x, y = patch, labels

    print("Test batch", x.shape, y.shape)

    elastic_3D = RandomElasticDeformation3D(alpha=(5, 5), sigma=(30, 30), p=1)
    elastic_params = elastic_3D.generate_parameters(x.shape)
    print(elastic_params["noise"].shape)

    elastic_3D.flags['interpolation'] = torch.tensor(1)
    transformed_x = elastic_3D.apply_transform(x, elastic_params)
    elastic_3D.flags['interpolation'] = torch.tensor(0)
    transformed_y = elastic_3D.apply_transform(y, elastic_params)

    fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
    to_plot = [x[0, 0, 30, :, :], y[0, 1, 30, :, :],
                x[0, 0, 30, :, :], y[0, 2, 30, :, :],
                transformed_x[0, 0, 30, :, :], transformed_y[0, 1, 30, :, :],
                transformed_x[0, 0, 30, :, :], transformed_y[0, 2, 30, :, :]]
    names = ["raw untransformed", "foreground untransformed",
            "raw untransformed", "boundaries untransformed",
            "raw transformed", "foreground transformed",
            "raw transformed", "boundaries untransformed"]
    for ax, img, name in zip(axs.flat, to_plot, names):
        ax.imshow(img)
        # ax.set(xlabel='y', ylabel='x')
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig("elastic_3D_xy.png", dpi=300)


    fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
    to_plot = [x[0, 0, :, 60, :], y[0, 1, :, 60, :],
                x[0, 0, :, 60, :], y[0, 2, :, 60, :],
                transformed_x[0, 0, :, 60, :], transformed_y[0, 1, :, 60, :],
                transformed_x[0, 0, :, 60, :], transformed_y[0, 2, :, 60, :]]
    names = ["raw untransformed", "foreground untransformed",
            "raw untransformed", "boundaries untransformed",
            "raw transformed", "foreground transformed",
            "raw transformed", "boundaries untransformed"]
    for ax, img, name in zip(axs.flat, to_plot, names):
        ax.imshow(img)
        ax.set(xlabel='y', ylabel='z')
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig("elastic_3D_zy.png", dpi=300)