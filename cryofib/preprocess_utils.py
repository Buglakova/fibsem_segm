import numpy as np
from skimage.measure import label

def percentile_norm(img, pmin, pmax, mask=None, eps=1e-10):
    """ Percentile normalization of the image.

    Args:
        img: _description_
        pmin: lower percentile in percents
        pmax: higher percentile in percents
        mask: normalize only part of the image. Default: normalize whole image.
        eps: constant to avoid devision by 0. Defaults to 1e-10.

    Returns:
        normalized image
        """
    if mask is None:
        pmin = np.percentile(img, pmin)
        pmax = np.percentile(img, pmax)
        norm_img = (img - pmin) / (pmax - pmin + eps)
            
    else:
        assert mask.shape == img.shape, "Mask is not same shape as image"
        pmin = np.percentile(img[mask], pmin)
        pmax = np.percentile(img[mask], pmax)
        norm_img[mask] = (img[mask] - pmin) / (pmax - pmin + eps)
    return norm_img


def zero_mean_unit_variance(img):
    return (img - np.mean(img)) / np.std(img)


def get_zero_component(img: np.ndarray, bg_const=0):
    bg = label(img == bg_const)
    component_sizes = [np.count_nonzero(bg == i) for i in np.unique(bg)[1:]]
    if len(component_sizes) == 0:
        print(f"No pixels with value equal to {bg_const}")
        return img > 0
    bg_ind = np.argmax(component_sizes) + 1
    bg = (bg == bg_ind)
    fg = (bg != bg_ind)
    return fg


def get_fg_mask(img: np.ndarray, bg_const=0):
    """Get foreground mask, where background is the largest connected component of zeros in each z-slice.

    Args:
        img: _description_

    Returns:
        Mask, which is 1 on foreground and 0 in the background.
    """
    print("Compute foreground mask")
    print("Image shape: ", img.shape)
    fg_mask = np.array([get_zero_component(z_slice, bg_const=bg_const) for z_slice in img])
    return fg_mask
