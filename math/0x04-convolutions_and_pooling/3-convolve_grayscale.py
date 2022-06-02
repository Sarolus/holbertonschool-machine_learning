#!/usr/bin/env python3
"""
    Convolution Strides Module
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
        Function that performs a convolution on grayscale images

        Args:
            images: np.ndarray with shape (m, h, w) containing multiple
            grayscale images
                m: the number of images
                h: height in pixels of the images
                w: width in pixels of the images
            kernel: np.ndarray with shape (kh, kw) containing the kernel
            for the convolution
                kh: kernel height
                kw: kernel width
            padding: string containing the type of padding to be used
            stride: tuple with the strides for the convolution
            Returns: np.ndarray containing the convolved images
    """

    image_nb, image_height, image_width = images.shape
    kernel_height, kernel_width = kernel.shape
    stride_height, stride_width = stride
    image_size = np.arange(image_nb)

    # Calculate the padding height and width
    if padding == 'same':
        padding_height = int(
            (
                (
                    (image_height - 1) * stride_height +
                    kernel_height - image_height
                ) / 2
            ) + 1
        )
        padding_width = int(
            (
                (
                    (image_width - 1) * stride_width +
                    kernel_width - image_width
                ) / 2
            ) + 1
        )
    elif padding == 'valid':
        padding_height = 0
        padding_width = 0
    else:
        padding_height, padding_width = padding

    # Calculate the output height and width
    convoluted_height = int(
        (image_height - kernel_height + 2 * padding_height) / stride_height + 1
    )
    convoluted_width = int(
        (image_width - kernel_width + 2 * padding_width) / stride_width + 1
    )

    # Add padding to the images
    padding = np.pad(
        images,
        pad_width=(
            (0, 0),
            (padding_height, padding_height),
            (padding_width, padding_width)
        ),
        mode='constant',
    )

    # Create a new matrice to hold the padded images
    convoluted_images = np.zeros(
        (image_nb, convoluted_height, convoluted_width))

    for row in range(convoluted_height):
        for column in range(convoluted_width):
            input_image = padding[
                image_size,
                row * stride[0]: row * stride[0] + kernel_height,
                column * stride[1]: column * stride[1] + kernel_width
            ]
            convoluted_images[image_size, row, column] = np.sum(
                input_image * kernel,
                axis=(1, 2)
            )
    return convoluted_images
