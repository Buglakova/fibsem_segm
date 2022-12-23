import argparse
import numpy as np
import torch.nn as nn
import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer, _check_plt
from torch_em.transform.raw import get_default_raw_augmentations
from torch_em.transform.augmentation import get_augmentations
from torch_em.loss.wrapper import MaskIgnoreLabel
import torch

from torch.cuda import mem_get_info

from pathlib import Path
import z5py



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


def thick_boundary_transform(labels):
    # labels:
    # 0: no data -> map to channel 3
    # 1: boundary -> map to channel 1
    # 2: extracellular -> map to channel 2
    # >2: cells -> map to channel 0
    n_channels = 4
    out = np.zeros((n_channels,) + labels.shape, dtype="float32")

    mask = (labels == -1)
    out[0] = (labels > 2).astype("float32")
    out[0, mask] = -1
    out[1] = (labels == 1).astype("float32")
    out[1, mask] = -1
    out[2] = (labels == 2).astype("float32")
    out[2, mask] = -1
    out[3] = (labels == 0).astype("float32")
    out[3, mask] = -1

    return out


if __name__ == "__main__":

    # Set paths to the train and test data
    # Dataset consists of 9 labeled stacks => use 01 - 07 for training
    # 08, 09 for test
    train_data_paths = ["/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5", "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5", "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5", "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5"]
    # train_data_paths = "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5"
    val_data_paths =  "/scratch/buglakova/data/cryofib/segm_fibsem/F107/F107_A1_em.n5"
    data_key = "raw"
    label_key = "segmentation/ground_truth_channels"

    experiment_name = "full_masked_dice_s0_64x256x256"

    # Set parameters of the network
    # patch_shape = (32, 256, 256)
    # Suggested by Constantine
    
    patch_shape = (64, 256, 256)
    assert len(patch_shape) == 3

    # # Loss, metric, batch size
    batch_size = 1
    loss = "dice"
    metric = "dice"

    # Roi: whole dataset
    # all_annotated_rois = [np.s_[200:229, :, :], np.s_[279:309, :, :], np.s_[617:625, :, :], np.s_[1149:1159, :, :], np.s_[565:566, :, :]]
    # train_rois = [np.s_[200:229, :, :], np.s_[279:309, :, :], np.s_[617:625, :, :], np.s_[1149:1159, :, :]]
    # val_rois = [np.s_[565:566, :, :]]

    # ROIs with +- 32 frames
    train_rois = (np.s_[182:247, :, :], np.s_[262:327, :, :], np.s_[589:654, :, :], np.s_[1122:1187, :, :])
    # train_rois = [np.s_[200:229, :, :], np.s_[279:310, :, :], np.s_[617:626, :, :], np.s_[1149:1160, :, :]]
    # val_rois = np.s_[565:566, :, :]
    val_rois = np.s_[533:598, :, :]


    # Check inputs
    print("Checking the training dataset:")
    train_ds = check_data(train_data_paths, data_key, train_data_paths, label_key, train_rois)
    print("/n/n/n")
    print("Train dataset")
    print(len(train_ds.datasets[0]))
    print("Length", len(train_ds))
    print(train_ds[0][0].shape)
    val_ds = check_data(val_data_paths, data_key, val_data_paths, label_key, val_rois)

    transforms = ["RandomHorizontalFlip3D", "RandomVerticalFlip3D", "RandomDepthicalFlip3D"]
    transforms = get_augmentations(ndim=3, transforms=transforms)

    print("Create data loaders")
    train_loader = torch_em.default_segmentation_loader(
        train_data_paths, data_key, train_data_paths, label_key,
        batch_size, patch_shape, with_label_channels=True, rois=train_rois,
        transform=transforms, num_workers=8
    )
    val_loader = torch_em.default_segmentation_loader(
        val_data_paths, data_key, val_data_paths, label_key,
        batch_size, patch_shape,  rois=val_rois, with_label_channels=True, num_workers=8
    )

    print("Plot several samples")
    fig = _check_plt(train_loader, 5, False)
    fig.tight_layout()
    fig.savefig("train_loader_s0_examples.png", dpi=300)


    loss_function = get_loss(loss, loss_transform=MaskIgnoreLabel(-1))
    metric_function = get_loss(metric, loss_transform=MaskIgnoreLabel(-1))


    # loss_function = get_loss(loss, loss_transform=ApplyMask())
    # metric_function = get_loss(metric, loss_transform=ApplyMask())

    kwargs = dict(
        ndim=3, patch_shape=patch_shape, batch_size=batch_size, with_label_channels=True
    )


    # Network
    # 4 levels with anisotropic scaling in the first two levels (scale only in xy)
    # Scale first x, y until more or less isotropic, then scale some more
    # Anisotropy factor here 20 / 15 = 1.33
    scale_factors = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    initial_features = 32
    final_activation = "Sigmoid"

    in_channels = 1
    out_channels = 4

    print("Creating 3D UNet with", in_channels, "input channels and", out_channels, "output channels.")
    model = AnisotropicUNet(
        in_channels=in_channels, out_channels=out_channels, scale_factors=scale_factors, final_activation=final_activation
        )

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

