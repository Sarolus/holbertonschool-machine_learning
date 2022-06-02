#!/usr/bin/env python3
"""
    Convolution Multiple Kernel Module
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
        Function that performs a convolution on images with channels

        Args:
            images: np.ndarray with shape (m, h, w, c) containing multiple
                    images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
                c: number of channels of the images
            kernels: np.ndarray with shape (kh, kw, c, m) containing the
                    kernels for the convolution
                kh: kernel
                kw: kernel
                c: number of channels of the images
                m: the number of images
            padding: string containing the type of padding. Can be 'same' or
                     'valid'
            stride: tuple with the strides for the convolution

            Returns:
                np.ndarray containing the convolved images
    """

    image_nb, image_h, image_w, _ = images.shape
    kernel_h, kernel_w, c2, kernel_nb = kernels.shape
    sh, sw = stride

    # Calculate the padding height and width
    if type(padding) is tuple:
        padding_h, padding_w = padding
    elif padding == 'same':
        padding_h = int((((image_h - 1) * sh + kernel_h - image_h) / 2) + 1)
        padding_w = int((((image_w - 1) * sw + kernel_w - image_w) / 2) + 1)
    elif padding == 'valid':
        padding_h = 0
        padding_w = 0
    else:
        raise NameError('padding must be same, valid, or a tuple')

    # Calculate the output height and width
    convoluted_h = int(((image_h + (2 * padding_h) - kernel_h) / sh) + 1)
    convoluted_w = int(((image_w + (2 * padding_w) - kernel_w) / sw) + 1)

    # Create a new matrice to hold the padded images
    convoluted_images = np.zeros(
        (image_nb, convoluted_h, convoluted_w, kernel_nb)
    )

    # Add padding to the images
    padding = np.pad(images, ((0, 0), (padding_h, padding_h),
                              (padding_w, padding_w), (0, 0)), 'constant')

    kernels_cpy = kernels.copy()
    for row in range(convoluted_h):
        for column in range(convoluted_w):
            x = row * sh
            y = column * sw
            images_slide = padding[:, x:x + kernel_h, y:y + kernel_w, :]

            for kernel_index in range(kernel_nb):
                convoluted_images[:, row, column, kernel_index] = np.tensordot(
                    images_slide,
                    kernels_cpy[:, :, :, kernel_index], axes=3
                )
    return convoluted_images
