#!/usr/bin/env python3
"""
    Pooling Module
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
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
    image_nb, image_height, image_width, image_channels = images.shape
    kernel_height, kernel_width = kernel_shape
    stride_height, stride_width = stride

    # Calculate the output height and width
    convoluted_height = int(
        (
            (image_height - kernel_height) / stride_height
        ) + 1
    )
    convoluted_width = int(
        (
            (image_width - kernel_width) / stride_width
        ) + 1
    )

    # Create a new matrice to hold the convoluted images
    convoluted_images = np.zeros(
        (image_nb, convoluted_height, convoluted_width, image_channels)
    )

    pooling = np.average if mode == 'avg' else np.max

    for column in range(convoluted_width):
        for row in range(convoluted_height):
            convoluted_images[:, row, column, :] = pooling(
                images[
                    :,
                    row * stride_height: row *
                    stride_height + kernel_height,
                    column * stride_width: column *
                    stride_width + kernel_width,
                    :
                ],
                axis=(1, 2),
            )

    return convoluted_images
