import argparse
import numpy as np
import torch.nn as nn
import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer
import torch

from pathlib import Path


def check_data(data_paths, data_key, label_paths, label_key, rois):
    print("Loading the raw data from:", data_paths, data_key)
    print("Loading the labels from:", label_paths, label_key)
    try:
        torch_em.default_segmentation_dataset(data_paths, data_key, label_paths, label_key, patch_shape, rois=rois, with_label_channels=True)
    except Exception as e:
        print("Loading the dataset failed with:")
        raise e

    
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


class ApplyMask:
    def __call__(self, prediction, target, mask_const=-1):
        assert target.dim() == prediction.dim(), f"{target.dim()}, {prediction.dim()}"
        assert target.shape[2:] == prediction.shape[2:], f"{str(target.shape)}, {str(prediction.shape)}"

        mask = target != mask_const
        mask.requires_grad = False

        # mask the prediction
        prediction = prediction * mask
        # print("Prediction range", prediction.min(), prediction.max())
        
        target = target * mask
        # print("Target range", target.min(), target.max())

        # with open("metrics.txt", "a") as f:
        #     f.write("pred " + str(prediction.min()) + str(prediction.max()))
        #     f.write(r"/n target " + str(target.min()) + str(target.max()))
        return prediction, target


if __name__ == "__main__":

    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Run training of 3D UNet for volume EM segmentation.
        """
    )
    parser.add_argument("train_data_paths", type=str, help="Path to train images (converted to n5)")
    parser.add_argument("train_label_paths", type=str, help="Path to train labels (converted to n5)")
    parser.add_argument("data_key", type=str, help="Key of raw data inside the n5")
    parser.add_argument("label_key", type=str, help="Key of labels inside the n5")
    parser.add_argument("experiment_name", type=str, help="Experiment (and model) name")

    args = parser.parse_args()
    experiment_name = args.experiment_name

    # Set parameters of the network
    patch_shape = (32, 256, 256)

    # Todo: make it into parameters
    train_rois = np.s_[594:650, :, :]
    val_rois = np.s_[540:594, :, :]

    # Check inputs
    print("Checking the training dataset:")
    check_data(args.train_data_paths, args.data_key, args.train_label_paths, args.label_key, train_rois)
    check_data(args.train_data_paths, args.data_key, args.train_label_paths, args.label_key, val_rois)

    assert len(patch_shape) == 3


    # Loss, metric, batch size
    batch_size = 1
    loss = "dice"
    metric = "dice"

    loss_function = get_loss(loss, loss_transform=ApplyMask())
    metric_function = get_loss(metric, loss_transform=ApplyMask())

    kwargs = dict(
        ndim=3, patch_shape=patch_shape, batch_size=batch_size, with_label_channels=True
    )

    print("Create data loaders")
    train_loader = torch_em.default_segmentation_loader(
        args.train_data_paths, args.data_key, args.train_label_paths, args.label_key,
        rois=train_rois, **kwargs
    )
    val_loader = torch_em.default_segmentation_loader(
        args.train_data_paths, args.data_key, args.train_label_paths, args.label_key,
        rois=val_rois, **kwargs
    )


    # Network
    # example for 4 levels with anisotropic scaling in the first two levels (scale only in xy)
    scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]

    initial_features = 32
    final_activation = "Sigmoid"

    in_channels = 1
    out_channels = 4

    print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
    model = AnisotropicUNet(
        in_channels=in_channels, out_channels=out_channels, scale_factors=scale_factors, final_activation=final_activation
        )

    # Train
    n_iterations = 50000
    learning_rate = 1.0e-3


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
    trainer.fit(n_iterations)


    # Save model in bioimage io format

    export_folder = "./bio-model"

    # Whether to convert the model weights to additional formats.
    # Currently, torchscript and onnx are support it and this will enable running the model
    # in more software tools.
    additional_weight_formats = None
    # additional_weight_formats = ["torchscript"]

    doc = """# Default model
    Try to run the training with default parameters just to see that it's training and prediction is working.
    """

    import torch_em.util

    for_dij = additional_weight_formats is not None and "torchscript" in additional_weight_formats

    training_data = None

    pred_str = "out_boundary_extra_fg"

    default_doc = f"""#{experiment_name}

    This model was trained with [the torch_em 3d UNet notebook](https://github.com/constantinpape/torch-em/blob/main/experiments/3D-UNet-Training.ipynb).
    """
    if pred_str:
        default_doc += f"It predicts {pred_str}.\n"

    training_summary = torch_em.util.get_training_summary(trainer, to_md=True, lr=learning_rate)
    default_doc += f"""## Training Schedule

    {training_summary}
    """

    if doc is None:
        doc = default_doc

    torch_em.util.export_bioimageio_model(
        trainer, export_folder, input_optional_parameters=True,
        for_deepimagej=for_dij, training_data=training_data, documentation=doc
    )
    torch_em.util.add_weight_formats(export_folder, additional_weight_formats)

