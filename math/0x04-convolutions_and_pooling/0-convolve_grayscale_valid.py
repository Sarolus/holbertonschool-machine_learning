#!/usr/bin/env python3
"""
    Valid Convolution Module
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
        Performs a valid convolution on grayscale images.

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

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((image_nb, output_height, output_width))

    for row in range(output_height):
        for column in range(output_width):
            input = images[image_size,
                           row:kernel_height+row,
                           column:kernel_width+column
                           ]
            output[image_size, row, column] = np.sum(input * kernel,
                                                     axis=(1, 2)
                                                     )

    return output
