# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

__author__ = "Ulysse Rubens <urubens@student.ulg.ac.be>"
__version__ = "0.1"


def pad_image(image, padding=(0,0), mode='reflect'):
    if padding == (0,0):
        return image

    if image.ndim == 3:
        pad = ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
    else:
        pad = ((padding[0], padding[0]), (padding[1], padding[1]))

    return np.pad(image, pad, mode=mode)


def open_image(filename, flag=None, padding=(0,0), padding_mode='reflect'):
    if not os.path.exists(filename):
        raise IOError("File " + filename + " does not exist !")

    if flag in ('L', cv2.IMREAD_GRAYSCALE):
        flag = cv2.IMREAD_GRAYSCALE
    elif flag in ('RGB', cv2.IMREAD_COLOR):
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_UNCHANGED

    im = cv2.imread(filename, flag)

    if im is None:
        raise ValueError("Image " + filename + "unreadable !")

    return pad_image(im, padding, padding_mode)


def open_image_with_mask(filename, padding=(0, 0), splitted=True):
    image = open_image(filename)
    if image.shape[2] == 3:
        mask = pad_image(np.ones_like(image[:, :, 0]), padding=padding, mode='constant')
        image = pad_image(image, padding=padding, mode='reflect')
    elif image.shape[2] == 4:
        mask = pad_image(image[:, :, 3], padding=padding, mode='constant')
        image = pad_image(image[:, :, :3], padding=padding, mode='reflect')
    else:
        raise ValueError("Impossible to load image")

    if not splitted:
        return np.dstack((image, mask))
    else:
        return image, mask


def open_scoremap(filename, padding=(0, 0)):
    scoremap = open_image(filename, flag='L', padding=padding)
    scoremap /= np.max(scoremap)
    return scoremap.astype(np.float16)


def files_in_directory(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
            and not f.startswith(".")]


def make_dirs(path, remove_filename=False):
    if remove_filename:
        path = os.path.dirname(path)

    if not os.path.exists(path):
        os.makedirs(path)
