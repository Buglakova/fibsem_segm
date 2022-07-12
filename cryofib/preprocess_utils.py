import numpy as np

def percentile_norm(img, pmin, pmax, eps=1e-10):
        pmin = np.percentile(img, pmin)
        pmax = np.percentile(img, pmax)
        return (img - pmin) / (pmax - pmin + eps)


def zero_mean_unit_variance(img):
    return (img - np.mean(img)) / np.std(img)
