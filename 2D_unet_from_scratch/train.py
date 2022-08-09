import argparse
from utils import EMDataset, run_training
from torch.utils.data import DataLoader


if __name__ == "__main__":

    ## Parse command line arguments
    parser = argparse.ArgumentParser(
        description="""Run training of 2D UNet for volume EM segmentation.
        """
    )
    parser.add_argument("train_images_dir", type=str, help="Path to train images")
    parser.add_argument("train_labels_dir", type=str, help="Path to train labels")
    parser.add_argument("test_images_dir", type=str, help="Path to validation images")
    parser.add_argument("test_labels_dir", type=str, help="Path to validation labels")
    parser.add_argument("model_path", type=str, help="Path to checkpoint")
    parser.add_argument("logger_path", type=str, help="Path to directory for tensorboard log")

    args = parser.parse_args()

    tile_size = (512, 512)
    n_epochs = 500000

    train_data = EMDataset(args.train_images_dir, args.train_labels_dir, tile_size=tile_size)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

    test_data = EMDataset(args.test_images_dir, args.test_labels_dir, tile_size=tile_size)
    val_loader = DataLoader(test_data, batch_size=1)

    run_training(train_loader, val_loader, args.model_path, args.logger_path, n_epochs, restart=True)

