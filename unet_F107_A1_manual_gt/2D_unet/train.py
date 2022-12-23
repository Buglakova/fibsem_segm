import argparse
import numpy as np
import torch.nn as nn
import torch_em
from torch_em.model import UNet2d
from torch_em.util.debug import check_loader, check_trainer, _check_plt
from torch_em.transform.raw import get_default_raw_augmentations
from torch_em.transform.augmentation import get_augmentations
import torch

from torch.cuda import mem_get_info

from pathlib import Path
import z5py

def check_data(data_paths, data_key, label_paths, label_key, rois):
    print("Loading the raw data from:", data_paths, data_key)
    print("Loading the labels from:", label_paths, label_key)
    try:
        ds = torch_em.default_segmentation_dataset(data_paths, data_key, label_paths, label_key, patch_shape, rois=rois, with_label_channels=True, is_seg_dataset=True, ndim=2)
        print("Raw dtype", ds[0][0].dtype)
        print("Label shape", ds[0][1].shape)
        print("Label dtype", ds[0][1].dtype)
    except Exception as e:
        print("Loading the dataset failed with:")
        raise e

    return ds

    
def get_loss(loss_name, loss_transform=None):
    loss_names = ["bce", "ce", "dice"]
    if isinstance(loss_name, str):
        assert loss_name in loss_names, f"{loss_name}, {loss_names}"
        if loss_name == "dice":
            loss_function = torch_em.loss.DiceLoss()
        elif loss == "ce":
            loss_function = nn.CrossEntropyLoss()
        elif loss == "bce":
            loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = loss_name
    
    # we need to add a loss wrapper for affinities
    if loss_transform:
        loss_function = torch_em.loss.LossWrapper(
            loss_function, transform=loss_transform
        )
    return loss_function


if __name__ == "__main__":

    # Set paths to the train and test data
    train_data_paths = "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em_gt.n5"
    val_data_paths = "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em_gt.n5"
    data_key = "raw"
    label_key = "segmentation/ground_truth_channels"

    experiment_name = "2D_s0"

    # Set parameters of the network
    # patch_shape = (32, 256, 256)
    # Suggested by Constantine
    
    patch_shape = (1, 512, 512)
    # assert len(patch_shape) == 2

    # # Loss, metric, batch size
    batch_size = 1
    loss = "dice"
    metric = "dice"

    # Roi: whole dataset
    train_rois = np.s_[0:75, :, :]
    val_rois = np.s_[75:, :, :]

    # Check inputs
    print("Checking the training dataset:")
    train_ds = check_data(train_data_paths, data_key, train_data_paths, label_key, train_rois)
    # print("/n/n/n")
    # print("Train dataset")
    # print(len(train_ds.datasets[0]))
    # print("Length", len(train_ds))
    # print(train_ds[0][0].shape)
    val_ds = check_data(val_data_paths, data_key, val_data_paths, label_key, val_rois)

    # raw_transforms = get_default_raw_augmentations()
    raw_transforms = get_raw_transform(normalizer = lambda x: x)
    transforms = ["RandomHorizontalFlip", "RandomVerticalFlip"]
    transforms = get_augmentations(ndim=2, transforms=transforms)

    print("Create data loaders")
    train_loader = torch_em.default_segmentation_loader(
        train_data_paths, data_key, train_data_paths, label_key,
        batch_size, patch_shape, with_label_channels=True, raw_transform=raw_transforms,
        transform=transforms, num_workers=8, ndim=2
    )
    val_loader = torch_em.default_segmentation_loader(
        train_data_paths, data_key, train_data_paths, label_key,
        batch_size, patch_shape, with_label_channels=True, num_workers=8, ndim=2
    )

    # print("Plot several samples")
    fig = _check_plt(train_loader, 5, False)
    fig.tight_layout()
    fig.savefig("train_loader_examples.png", dpi=300)


    loss_function = get_loss(loss)
    metric_function = get_loss(metric)

    kwargs = dict(
        ndim=3, patch_shape=patch_shape, batch_size=batch_size, with_label_channels=True
    )


    # Network
    initial_features = 32
    final_activation = "Sigmoid"

    in_channels = 1
    out_channels = 4
    depth = 5

    print("Creating 2D UNet with", in_channels, "input channels and", out_channels, "output channels.")
    model = UNet2d(
        in_channels=in_channels, out_channels=out_channels, depth=depth, final_activation=final_activation
        )

    print("Model")
    print(model)

    # Train
    n_iterations = 100000
    learning_rate = 1.0e-4


    # Set device
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device(4)
    else:
        print("GPU is not available")
        device = torch.device("cpu")


    # Set logger ?

    trainer = torch_em.default_segmentation_trainer(
        name=experiment_name, model=model,
        train_loader=train_loader, val_loader=val_loader,
        loss=loss_function, metric=metric_function,
        learning_rate=learning_rate,
        mixed_precision=True,
        log_image_interval=50,
        device=device
    )

    print(trainer)

    trainer.fit(n_iterations)

