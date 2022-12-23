import matplotlib.pyplot as plt
import numpy as np
from cryofib.preprocess_utils import percentile_norm, zero_mean_unit_variance

def plot_overlay(img1, img2, save_path=None, x_pos=None, y_pos=None, z_pos=None):
    """Plot slices of two 3D images along each axis.

    Args:
        img1: _description_target_shape = (25, 22, 29)
        img2: _description_
        save_path: _description_. Defaults to None.
        x_pos: _description_. Defaults to None.
        y_pos: _description_. Defaults to None.
        z_pos: _description_. Defaults to None.
    """
    assert img1.ndim == 3
    assert img2.ndim == 3
    
    if x_pos is None:
        x_pos = min(int(img1.shape[2] // 2), int(img2.shape[2] // 2))
    if y_pos is None:
        y_pos = min(int(img1.shape[1] // 2), int(img2.shape[1] // 2))
    if z_pos is None:
        z_pos = min(int(img1.shape[0] // 2), int(img2.shape[0] // 2))
    
    plt.figure(figsize=(30, 10), dpi=300)
    plt.subplot(1,3,1)
    plt.title(f'z slice at {z_pos}')
    img1_alpha = (percentile_norm(img1, 0, 100) > 0) * 0.5
    img2_alpha = (percentile_norm(img2, 0, 100) > 0) * 0.5
    plt.imshow(img1[z_pos, :, :], cmap='Greys', alpha=img1_alpha[z_pos, :, :])
    plt.imshow(img2[z_pos, :, :], cmap="Blues", alpha=img2_alpha[z_pos, :, :])
    
    plt.subplot(1,3,2)
    plt.title(f'y slice at {y_pos}')
    plt.imshow(img1[:, y_pos, :], cmap='Greys', alpha=img1_alpha[:, y_pos, :])
    plt.imshow(img2[:, y_pos, :], cmap="Blues", alpha=img2_alpha[:, y_pos, :])
    
    plt.subplot(1,3,3)
    plt.title(f'x slice at {x_pos}')
    plt.imshow(img1[:, :, x_pos], cmap='Greys', alpha=img1_alpha[:, :, x_pos])
    plt.imshow(img2[:, :, x_pos], cmap="Blues", alpha=img2_alpha[:, :, x_pos])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
    plt.close()