#!/usr/bin/env python3
"""
    Convolution with Padding Module
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
        Performs a same convolution on grayscale images.

        Args:
            images (np.ndarray): np.ndarray containing multiple
            grayscale images.
            kernel (np.ndarray): np.ndarray containing the kernel
            for the convolution.
            padding (tuple): tuple of padding height and width.

        Returns:
            np.ndarray: np.ndarray containing the convolved images.
    """
    image_nb, image_height, image_width = images.shape
    kernel_height, kernel_width = kernel.shape
    padding_height, padding_width = padding
    image_size = np.arange(image_nb)

    output_height = image_height - kernel_height + 1 + 2 * padding_height
    output_width = image_width - kernel_width + 1 + 2 * padding_width

    padding = np.pad(images,
                     ((0, 0),
                      (padding_height, padding_height),
                      (padding_width, padding_width)),
                     mode="constant"
                     )
    output = np.zeros((image_nb, output_height, output_width))

    for row in range(output_height):
        for column in range(output_width):
            padded_input = padding[image_size,
                                   row:kernel_height+row,
                                   column:kernel_width+column
                                   ]
            output[image_size, row, column] = np.sum(padded_input * kernel,
                                                     axis=(1, 2)
                                                     )

    return output
