import os
import matplotlib.pyplot as plt
from functools import partial
from itertools import product
import imageio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

from imageio import imread
from skimage.io import imsave
from tqdm import tqdm, trange
from pathlib import Path

from torchvision import transforms

import z5py



#
# helper functions to load and split the data
#

def convert_to_chan(mask):
    # 0 is out of sample
    # 1 is border
    # 2 is extracellular
    # >2 are cell instances
    mask_chan = np.stack([mask == 0, mask == 1, mask == 2, mask > 2], axis=-1).astype(int)
    return mask_chan


def split_with_overlap(image, tile_size=(256, 512), overlap=0.1):
    nonzero_ind = np.nonzero(image)
    tile_size = np.array(tile_size)
    overlap_size = np.floor(overlap * tile_size)
    trunc_size = (tile_size - overlap_size).astype(int)
    ind_min = np.min(nonzero_ind, axis=1)
    # print(ind_min)
    # Split the whole image
    ind_min = np.array((0, 0))
    ind_max = np.max(nonzero_ind, axis=1)

    n_tiles = np.ceil((ind_max - ind_min) / trunc_size).astype(int)
    ind_min = np.multiply(np.indices(dimensions=n_tiles), trunc_size[:, None, None]) + ind_min[:, None, None]
    ind_min = ind_min.reshape((2, -1))
    ind_max = ind_min + tile_size[:, None]
    return n_tiles, ind_min, ind_max


def crop_tile(img, ind_min, ind_max):
    pad_w = ind_max - img.shape
    pad_w[pad_w < 0] = 0
    pad_w = np.stack((np.zeros_like(pad_w), pad_w)).T
    img_padded = np.pad(img, pad_width=pad_w)
    return img_padded[ind_min[0]:ind_max[0], ind_min[1]:ind_max[1], ...]


def stack_chan(segm):
    segm = segm[0, :, :, :]
    return np.hstack(segm)


#
# transformations and datasets
#


class EMDataset(Dataset):
    """ A PyTorch dataset to load volume EM images and manually segmented masks """
    def __init__(self, img_path, mask_path, tile_size=(512, 512), transform=None):
        self.img_path = Path(img_path)  # the directory with all the training samples
        self.mask_path = Path(mask_path)  # the directory with all the training samples
        self.img_list = sorted(list(self.img_path.glob("*.tiff"))) # list the samples
        self.mask_list = sorted(list(self.mask_path.glob("*.tiff")))

        self.tile_size = tile_size
        self.transform = transform    # transformations to apply to both inputs and targets
        #  transformations to apply just to inputs
        self.inp_transforms = [transforms.ToTensor()]
        # transformations to apply just to targets
        self.mask_transforms = transforms.ToTensor()

    # get the total number of samples
    def __len__(self):
        return len(self.img_list)


    # fetch the training sample given its index
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.mask_list[idx]
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images 
        image = imageio.imread(img_path)
        mask = imageio.imread(mask_path)
        
        tile_layout, ind_min, ind_max = split_with_overlap(image, tile_size=self.tile_size, overlap=0.3)
        n_tiles = ind_min.shape[1]
        # print(n_tiles, ind_min.shape, ind_max.shape)

        # Crop random tile
        rand_n = np.random.randint(low=0, high=n_tiles)
        # print(rand_n)

        img_tile = crop_tile(image, ind_min[:, rand_n], ind_max[:, rand_n])
        mask_tile = crop_tile(mask, ind_min[:, rand_n], ind_max[:, rand_n])

        # Convert mask into channels
        mask_tile = convert_to_chan(mask_tile)
        # print("Converted tile shape ", mask_tile.shape)

        # Apply transformations
        inp_transforms_idx = self.inp_transforms.copy()
        inp_transforms_idx.append(transforms.Normalize([np.mean(image)], [np.std(image)]))
        inp_transforms_idx = transforms.Compose(self.inp_transforms)
        img_tile = inp_transforms_idx(img_tile)
        mask_tile = self.mask_transforms(mask_tile)

        # print("tile shape", img_tile.shape)
        if self.transform is not None:
            img_tile, mask_tile = self.transform([img_tile, mask_tile])
        return img_tile, mask_tile


    def split_tiles(self, idx):
        pass


    def get_tile_list(self, idx):
        pass

#
# visualisation functionality
#


#
# models
#

class EMDataset(Dataset):
    """ A PyTorch dataset to load volume EM images and manually segmented masks """
    def __init__(self, img_path, mask_path, tile_size=(512, 512), transform=None, overlap=0.3):
        self.img_path = Path(img_path)  # the directory with all the training samples
        self.mask_path = Path(mask_path)  # the directory with all the training samples
        self.img_list = sorted(list(self.img_path.glob("*.tiff"))) # list the samples
        self.mask_list = sorted(list(self.mask_path.glob("*.tiff")))

        self.tile_size = tile_size
        self.transform = transform    # transformations to apply to both inputs and targets
        #  transformations to apply just to inputs
        self.inp_transforms = [transforms.ToTensor()]
        # transformations to apply just to targets
        self.mask_transforms = transforms.ToTensor()

        self.overlap = overlap

    # get the total number of samples
    def __len__(self):
        return len(self.img_list)


    # fetch the training sample given its index
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.mask_list[idx]
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images 
        image = imageio.imread(img_path)
        mask = imageio.imread(mask_path)
        
        tile_layout, ind_min, ind_max = split_with_overlap(image, tile_size=self.tile_size, overlap=self.overlap)
        n_tiles = ind_min.shape[1]
        # print(n_tiles, ind_min.shape, ind_max.shape)

        # Crop random tile
        rand_n = np.random.randint(low=0, high=n_tiles)
        # print(rand_n)

        img_tile = crop_tile(image, ind_min[:, rand_n], ind_max[:, rand_n])
        mask_tile = crop_tile(mask, ind_min[:, rand_n], ind_max[:, rand_n])

        # Convert mask into channels
        mask_tile = convert_to_chan(mask_tile)
        # print("Converted tile shape ", mask_tile.shape)

        # Apply transformations
        inp_transforms_idx = self.inp_transforms.copy()
        inp_transforms_idx.append(transforms.Normalize([np.mean(image)], [np.std(image)]))
        inp_transforms_idx = transforms.Compose(self.inp_transforms)
        img_tile = inp_transforms_idx(img_tile)
        mask_tile = self.mask_transforms(mask_tile)

        # print("tile shape", img_tile.shape)
        if self.transform is not None:
            img_tile, mask_tile = self.transform([img_tile, mask_tile])
        return img_tile, mask_tile


    def get_tile_list(self, idx):
        img_path = self.img_list[idx]
        image = imageio.imread(img_path)
        tile_layout, ind_min, ind_max = split_with_overlap(image, tile_size=self.tile_size, overlap=0.3)

        tiles = [crop_tile(image, ind_min[:, n], ind_max[:, n]) for n in range(ind_min.shape[1])]

        # Apply transformations
        inp_transforms_idx = self.inp_transforms.copy()
        inp_transforms_idx.append(transforms.Normalize([np.mean(image)], [np.std(image)]))
        inp_transforms_idx = transforms.Compose(self.inp_transforms)
        tiles = [inp_transforms_idx(tile) for tile in tiles]

        return image, tile_layout, ind_min, ind_max, tiles


    def get_image(self, idx):
        img_path = self.img_list[idx]
        image = imageio.imread(img_path)
        
        return image


    def get_mask(self, idx):
        mask_path = self.mask_list[idx]
        mask = imageio.imread(mask_path)
        mask = convert_to_chan(mask)

        return mask

    def write_n5_pred(self, output_path):
        stitched_predictions = np.array(self.stitched_predictions)
        chunks = (1, 512, 512)
        shape = stitched_predictions[:, 0, :, :].shape
        compression = "gzip"
        dtype = stitched_predictions.dtype

        f = z5py.File(output_path, "a")
        g = f.create_group("predictions")

        ds_fg = g.create_dataset("foreground", shape=shape, compression="gzip",
                                 chunks=chunks, dtype="float32")
        ds_fg.n_threads = 8
        ds_bd = g.create_dataset("boundaries", shape=shape, compression="gzip",
                            chunks=chunks, dtype="float32")
        ds_bd.n_threads = 8
        ds_extra = g.create_dataset("extracellular", shape=shape, compression="gzip",
                            chunks=chunks, dtype="float32")
        ds_extra.n_threads = 8
        ds_bg = g.create_dataset("background", shape=shape, compression="gzip",
                            chunks=chunks, dtype="float32")
        ds_bg.n_threads = 8

        ds_fg[:] = stitched_predictions[:, 3, :, :]
        ds_bd[:] = stitched_predictions[:, 1, :, :]
        ds_extra[:] = stitched_predictions[:, 2, :, :]
        ds_bg[:] = stitched_predictions[:, 0, :, :]


    def write_n5_raw(self, output_path):
        raw_data = np.array([self.get_image(i) for i in range(len(self.img_list))])
        chunks = (1, 512, 512)
        shape = raw_data.shape
        compression = "gzip"
        dtype = raw_data.dtype

        f = z5py.File(output_path, "a")
        g = f.create_group("raw")
        ds_raw = g.create_dataset("raw_data", shape=shape, compression="gzip",
                                 chunks=chunks, dtype="uint8")
        ds_raw.n_threads = 8
        ds_raw[:] = raw_data


    def _stitch_tiles(self, ind_min, ind_max, predictions, image_shape):
        '''
            Predictions: array in format of NCHW, where N is number of tiles
        '''
        stitched = np.zeros((predictions.shape[1], *image_shape))
        stitched_n = np.zeros((predictions.shape[1], *image_shape))

        for i_min, i_max, tile in zip(ind_min.T, ind_max.T, predictions):
            x_size = stitched.shape[1] - i_min[0]
            y_size = stitched.shape[2] - i_min[1]

            # Multiply by linearly falling mask to reduce stitching artefacts

            mask = np.ones_like(tile)
            overlap = [int((self.overlap * size)) for size in tile.shape[1:]]
            # print(tile.shape, overlap[0])

            gradient_x = np.linspace(1, 0, overlap[0])[:, None]
            gradient_x = np.tile(gradient_x, (1, mask.shape[2]))
            # print("Slice shape", mask[:, -overlap[0]:, :].shape)
            # print("Mask part shape", gradient_x.shape)
            mask[:, -overlap[0]:, :] = mask[:, -overlap[0]:, :] * gradient_x

            gradient_y = np.linspace(1, 0, overlap[1])[:, None]
            gradient_y = np.tile(gradient_y, (1, mask.shape[1]))
            mask[:, :, -overlap[1]:] *= gradient_y.T

            if i_min[0] > 0:
                gradient_x = np.linspace(0, 1, overlap[0])[:, None]
                gradient_x = np.tile(gradient_x, (1, mask.shape[2]))
                mask[:, 0:overlap[0], :] *= gradient_x
            if i_min[1] > 0:
                
                gradient_y = np.linspace(0, 1, overlap[1])[:, None]
                gradient_y = np.tile(gradient_y, (1, mask.shape[1]))
                mask[:, :, 0:overlap[1]] *= gradient_y.T

            # plt.imshow(mask[2, :, :])
            # plt.title(str(i_min))
            # plt.colorbar()
            # plt.show()
            masked_tile = tile * mask
            stitched[:, i_min[0]:i_max[0], i_min[1]:i_max[1]] += masked_tile[:, 0:x_size, 0:y_size]
            stitched_n[:, i_min[0]:i_max[0], i_min[1]:i_max[1]] += 1

        stitched_n[stitched_n == 0] = 1

        # stitched = stitched / stitched_n

        return stitched


    def predict_boundaries(self, model_path):
        # Load network
        # 4 because my GPU is number 4
        if torch.cuda.is_available():
            print("GPU is available")
            device = torch.device(4)
        else:
            print("GPU is not available")
            device = torch.device("cpu")

        model_loaded = UNet(1, 4, final_activation=nn.Sigmoid())
        checkpoint = torch.load(model_path)
        model_loaded.load_state_dict(checkpoint['model_state_dict'])
        model_loaded = model_loaded.to(device)


        self.stitched_predictions = []
        for i in trange(len(self.img_list)):
            image, tile_layout, ind_min, ind_max, tiles = self.get_tile_list(i)

            predictions = []

            for tile in tiles:
                predictions.append(model_loaded(tile[None, ...].to(device)).to("cpu").detach().numpy()[0, ...])

            predictions = np.array(predictions)
            stitched = self._stitch_tiles(ind_min, ind_max, predictions, image.shape)
            self.stitched_predictions.append(stitched)


    def save_predictions(self, predictions_path):
        predictions_path = Path(predictions_path)
        print("Saving predictions to", predictions_path)
        for idx, prediction in enumerate(self.stitched_predictions):
            # print(self.img_list[idx])
            # print(prediction.shape)
            img_path = self.img_list[idx]
            new_path = predictions_path / (img_path.stem + "_predicted" + img_path.suffix)
            # print(new_path)
            imsave(new_path, np.moveaxis(prediction, 0, -1))




#
# models
#

class UNet(nn.Module):
    """ UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """
    
    # _conv_block and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements
    
    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                             nn.ReLU())       


    # upsampling via transposed 2d convolutions
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size=2, stride=2)
    
    def __init__(self, in_channels=1, out_channels=1, tile_size=(512, 512),
                 final_activation=None):
        super().__init__()
        
        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = 4

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation, nn.Module), "Activation must be torch module"
        
        # all lists of conv layers (or other nn.Modules with parameters) must be wraped
        # itnto a nn.ModuleList
        init_chan = 64
        
        # modules of the encoder path
        self.encoder = nn.ModuleList([self._conv_block(in_channels, init_chan),
                                      self._conv_block(init_chan, init_chan * 2 ** 1),
                                      self._conv_block(init_chan * 2 ** 1, init_chan * 2 ** 2),
                                      self._conv_block(init_chan * 2 ** 2, init_chan * 2 ** 3)])
        # the base convolution block
        self.base = self._conv_block(init_chan * 2 ** 3, init_chan * 2 ** 4)
        # modules of the decoder path
        self.decoder = nn.ModuleList([self._conv_block(init_chan * 2 ** 4, init_chan * 2 ** 3),
                                      self._conv_block(init_chan * 2 ** 3, init_chan * 2 ** 2),
                                      self._conv_block(init_chan * 2 ** 2, init_chan * 2 ** 1),
                                      self._conv_block(init_chan * 2 ** 1, init_chan)])
        
        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([nn.MaxPool2d(2) for _ in range(self.depth)])
        # the upsampling layers
        image_size = tile_size[0]
        self.upsamplers = nn.ModuleList([self._upsampler(init_chan * 2 ** 4, init_chan * 2 ** 3),
                                         self._upsampler(init_chan * 2 ** 3, init_chan * 2 ** 2),
                                         self._upsampler(init_chan * 2 ** 2, init_chan * 2 ** 1),
                                         self._upsampler(init_chan * 2 ** 1, init_chan)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv2d(init_chan, out_channels, 1)
        self.activation = final_activation
    
    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)
        
        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            # print(level)
            x = self.upsamplers[level](x)
            x = self.decoder[level](torch.cat((x, encoder_out[level]), dim=1))
        
        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

#
# loss
#

# Copied from https://github.com/constantinpape/torch-em/blob/4b3205048a21308b30a832170aa2d41f400eff98/torch_em/loss/dice.py

def flatten_samples(input_):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
    """
    # Get number of channels
    num_channels = input_.size(1)
    # Permute the channel axis to first
    permute_axes = list(range(input_.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    # For input shape (say) NCHW, this should have the shape CNHW
    permuted = input_.permute(*permute_axes).contiguous()
    # Now flatten out all but the first axis and return
    flattened = permuted.view(num_channels, -1)
    return flattened


def dice_score(input_, target, invert=False, channelwise=True, eps=1e-7):
    if channelwise:
        # Flatten input and target to have the shape (C, N),
        # where N is the number of samples
        input_ = flatten_samples(input_)
        target = flatten_samples(target)
        # Compute numerator and denominator (by summing over samples and
        # leaving the channels intact)
        numerator = (input_ * target).sum(-1)
        denominator = (input_ * input_).sum(-1) + (target * target).sum(-1)
        channelwise_score = 2 * (numerator / denominator.clamp(min=eps))
        if invert:
            channelwise_score = 1. - channelwise_score
        # Sum over the channels to compute the total score
        score = channelwise_score.sum()
    else:
        numerator = (input_ * target).sum()
        denominator = (input_ * input_).sum() + (target * target).sum()
        score = 2. * (numerator / denominator.clamp(min=eps))
        if invert:
            score = 1. - score
    return score


class DiceLoss(nn.Module):
    def __init__(self, channelwise=True, eps=1e-7):
        super().__init__()
        self.channelwise = channelwise
        self.eps = eps

        # all torch_em classes should store init kwargs to easily recreate the init call
        self.init_kwargs = {"channelwise": channelwise, "eps": self.eps}

    def forward(self, input_, target):
        return dice_score(input_, target,
                          invert=True, channelwise=self.channelwise,
                          eps=self.eps)


#
# training and validation functions
#



# apply training for one epoch
def train(model, loader, optimizer, loss_function,
          epoch, device, log_interval=100, log_image_interval=20, tb_logger=None):

    # set the model to train mode
    model.train()
    
    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # apply model, calculate loss and run backwards pass
        prediction = model(x)
        print(prediction.shape)
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()
        
        # log to console
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_id * len(x),
                  len(loader.dataset),
                  100. * batch_id / len(loader), loss.item()))

       # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=step)
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(tag='input', img_tensor=x[0, :, :, :].to('cpu'), global_step=step, dataformats="CHW")
                tb_logger.add_images(tag='target', img_tensor=stack_chan(y.to('cpu')), global_step=step, dataformats="HW")
                tb_logger.add_images(tag='prediction', img_tensor=stack_chan(prediction.to('cpu').detach()), global_step=step, dataformats="HW")


# run validation after training epoch
def validate(model, loader, loss_function, metric, device, step=None, tb_logger=None):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0
    
    # disable gradients during validation
    with torch.no_grad():
        
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction, y)
            val_metric += metric(prediction, y)
    
    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)
    
    if tb_logger is not None:
        assert step is not None, "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(tag='val_metric', scalar_value=val_metric, global_step=step)
        # we always log the last validation images
        tb_logger.add_images(tag='val_input', img_tensor=x[0, :, :, :].to('cpu'), global_step=step, dataformats="CHW")
        # tb_logger.add_images(tag='val_target', img_tensor=y.to('cpu')[:, 1, :, :], global_step=step)
        print("Test y shape", y.shape)
        print("Test prediction shape", prediction.shape)
        tb_logger.add_images(tag='val_target', img_tensor=stack_chan(y.to('cpu')), global_step=step, dataformats="HW")
        tb_logger.add_images(tag='val_prediction', img_tensor=stack_chan(prediction.to('cpu')), global_step=step, dataformats="HW")
        
    print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(val_loss, val_metric))

    return val_loss


def run_training(train_loader, val_loader, model_path, logger_path, n_epochs, restart=False):
    # check if we have  a gpu
    # 4 because my GPU is 4
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device(4)
    else:
        print("GPU is not available")
        device = torch.device("cpu")

    # build the dice coefficient metric
    metric = DiceLoss()
    loss_function = DiceLoss() 
    # start a tensorboard writer
    logger = SummaryWriter(logger_path)

    # use adam optimizer
    learning_rate = 1e-4
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= "min", factor=0.5, patience=5)
    start_epoch = 0

    if restart:
        checkpoint = torch.load(model_path)

        # build a default unet with sigmoid activation
        # to normalize predictions to [0, 1]
        net = UNet(1, 4, final_activation=nn.Sigmoid())
        net.load_state_dict(checkpoint['model_state_dict'])

        # move the model to GPU
        net = net.to(device)

        # use adam optimizer
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    else:
        # build a default unet with sigmoid activation
        # to normalize predictions to [0, 1]
        net = UNet(1, 4, final_activation=nn.Sigmoid())
        # move the model to GPU
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    

    # during the training you can inspect the 
    # predictions in the tensorboard
    min_loss = 1
    for epoch in range(start_epoch, start_epoch + n_epochs):
        # train
        train(net, train_loader, optimizer, loss_function,
                epoch, device, log_interval=100, log_image_interval=20, tb_logger=logger)

        step = epoch * len(train_loader.dataset)
        # validate
        val_loss = validate(net, val_loader, loss_function, metric, device, step=step, tb_logger=logger)

        # If loss decreased, save model
        if val_loss < min_loss:
                min_loss = val_loss
                model_path = model_path
                print(f"Save model to {model_path}")
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            }, model_path)