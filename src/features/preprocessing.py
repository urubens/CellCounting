# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage as snd

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


def scoremap_transform_edt(binary_mask, mean_radius, alpha):
    """
    Compute an exponentially shaped Euclidean distance transform 
    of a binary mask.

    Parameters
    ----------
    binary_mask : array-like
        The binary mask.
    mean_radius : integer
        The mean radius of objects contained in the image.
    alpha : integer, optional (default: 4)
        The parameter controlling the shape of the exponential.

    Returns
    -------
    scoremap: array-like
        Transformed binary mask using exponential Euclidean distance
        transform, with each score belonging to [0, 1] interval.

    References
    ----------
    . [1] Sironal et al., "Multiscale Centerline Detection by Learning
          a Scale-Scpace Distance Transform", CVPR, 2014.
    . [2] P. Kainz et al., "You should use regression to detect cells",
          MICCAI, 2015.
    """

    # Reverse mask to have (i,j)=0 if there is an annotation in (i,j)
    binary_mask = 1 - binary_mask

    scoremap = np.asarray(snd.distance_transform_edt(binary_mask))
    st_d_m = np.nonzero(scoremap < mean_radius)
    gt_d_m = np.nonzero(scoremap >= mean_radius)

    scoremap[st_d_m] = np.exp(alpha * (1 - (scoremap[st_d_m] / mean_radius))) - 1
    scoremap[gt_d_m] = 0

    # Normalization to have scores in [0, 1]
    scoremap /= scoremap.max()

    return scoremap


def scoremap_transform_density(binary_mask, mean_radius):
    """
    Compute a density map from a binary_mask, using a Gaussian filter.

    Parameters
    ----------
    binary_mask: array-like
        The binary mask.
    mean_radius: integer
        TThe mean radius of objects contained in the image.

    Returns
    -------
    scoremap: array-like
        Transformed binary mask using a Gaussian filter, where each 
        element (i,j) is the density of object per pixel at position
        (i,j) with values in [0, 1] interval.

    References
    ----------
    . [1] L. Fiaschi et al., "Learning to count with regression forest 
          and structured labels", ICPR, 2012.
    """

    #sigma = mean_radius / 2.
    sigma = 1.

    binary_mask = binary_mask.astype(np.float)
    # Multiply by 100. for FCRN ?
    scoremap = snd.filters.gaussian_filter(binary_mask, sigma=sigma)

    return scoremap


def scoremap_transform_proximity(binary_mask, mean_radius, alpha):
    # TODO - Not implemented yet
    return np.zeros_like(binary_mask)


if __name__ == '__main__':
    pass
    # from utils import open_image, open_scoremap
    # im = open_image(
    #     '/Users/ulysse/Documents/Programming/TFE/Counting/datasets/GRAZ-2015/images_with_cells/BM_GRAZ_HE_0001_01_withcells.png')
    # im = im[500:700, 500:700, ...]
    # cv2.imwrite('im.png', im)
    #
    # mask = open_scoremap(
    #     '/Users/ulysse/Documents/Programming/TFE/Counting/datasets/GRAZ-2015/groundtruths/BM_GRAZ_HE_0001_01.png')
    # mask = mask[500:700, 500:700]
    # cv2.imwrite('mask.png', np.asarray(mask * 255, dtype=np.uint))
    #
    # edt = np.asarray(snd.distance_transform_edt(1 - mask))
    # cv2.imwrite('edt.png', 255 - edt)
    #
    # scoremap = scoremap_transform_edt(mask, alpha=5, max_radius=39)
    # cv2.imwrite('scoremap.png', scoremap * 255)