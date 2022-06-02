#!/usr/bin/env python3
"""
    Same Convolution Module
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
        Performs a same convolution on grayscale images.

        Args:
            images (np.ndarray): np.ndarray containing multiple
            grayscale images.
            kernel (np.ndarray): np.ndarray containing the kernel
            for the convolution.

        Returns:
            np.ndarray: np.ndarray containing the convolved images.
    """
    image_nb, image_height, image_width = images.shape
    kernel_height, kernel_width = kernel.shape
    image_size = np.arange(image_nb)

    padding_height = int((kernel_height - 1) / 2)
    padding_width = int((kernel_width - 1) / 2)

    if kernel_height % 2 == 0:
        padding_height += 1
    if kernel_width % 2 == 0:
        padding_width += 1

    padding = np.pad(images,
                     ((0, 0),
                      (padding_height, padding_height),
                      (padding_width, padding_width)),
                     mode="constant"
                     )
    output = np.zeros((image_nb, image_height, image_width))

    for row in range(image_height):
        for column in range(image_width):
            padded_input = padding[image_size,
                                   row:kernel_height+row,
                                   column:kernel_width+column
                                   ]
            output[image_size, row, column] = np.sum(padded_input * kernel,
                                                     axis=(1, 2)
                                                     )

    return output
