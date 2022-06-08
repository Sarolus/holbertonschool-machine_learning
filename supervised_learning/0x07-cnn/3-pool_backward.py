#!/usr/bin/env python3
"""
    Pooling Backward Propagation Module
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        Performs back propagation over a pooling layer of a neural network

        Args:
            dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
                containing the gradient of the cost with respect to the output
                of the pooling layer
            A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
            kernel_shape: tuple of (kh, kw) containing the size of the kernel
            stride: tuple of (sh, sw) containing the strides for the pooling
            mode: string containing either max or average, indicating whether
                to perform maximum or average pooling, resp.

        Returns:
            dA_prev: numpy.ndarray containing the gradient of the cost with
                     respect to the output of the previous layer<
    """

    def avg_inv(
        output_derivative, output_index,
        variables
    ):
        """
            Calculates the average of the values in the pooling layer

            Args:
                output_derivative: numpy.ndarray containing the gradient of
                                   the cost with respect to the output of the
                                   pooling layer
                output_index: numpy.ndarray containing the index of the
                              maximum value of the pooling layer
                variables:
                    row: int containing the row index of the pooling layer
                    column: int containing the column index of the pooling
                            layer
                    channel: int containing the channel index of the pooling
                             layer
                    kernel_height: int containing the height of the kernel
                    kernel_width: int containing the width of the kernel

            Returns:
                avg: int containing the average of the values in the pooling
        """

        return output_derivative[
            output_index, variables[row], variables[column], variables[channel]
        ] / (variables[filter_height] * variables[filter_width])

    def max_inv(
        A_prev, output_index, variables,
        x_start, x_end, y_start, y_end,
        output_derivative
    ):
        """
            Calculates the maximum of the values in the pooling layer

            Args:
                A_prev: numpy.ndarray containing the output of the previous
                        layer
                output_index: numpy.ndarray containing the index of the
                              maximum value of the pooling layer
                variables:
                    row: int containing the row index of the pooling layer
                    column: int containing the column index of the pooling
                            layer
                    channel: int containing the channel index of the pooling
                             layer
                x_start: int containing the starting x index of the pooling
                         layer
                x_end: int containing the ending x index of the pooling layer
                y_start: int containing the starting y index of the pooling
                         layer
                y_end: int containing the ending y index of the pooling layer
                output_derivative: numpy.ndarray containing the gradient of
                                   the cost with respect to the output of the
                                   pooling layer

            Returns:
                dA_prev: numpy.ndarray containing the gradient of the cost
                         with respect to the output of the previous layer
        """
        a_prev_slice = A_prev[
            output_index, x_start:x_end, y_start:y_end, channel
        ]
        mask = (a_prev_slice == np.max(a_prev_slice))
        return output_derivative[
            output_index, variables[row], variables[column], variables[channel]
        ] * mask

    output_nb, output_height, output_width, output_channels = dA.shape
    filter_height, filter_width = kernel_shape
    stride_height, stride_width = stride

    dA_prev = np.zeros_like(A_prev)

    # Loop over outputs
    for output_index in range(output_nb):

        # Loop over vertical axis
        for row in range(output_height):

            # Loop over horizontal axis
            for column in range(output_width):
                x_start = row * stride_height
                y_start = column * stride_width
                x_end = x_start + filter_height
                y_end = y_start + filter_width

                # Loop over channels
                for channel in range(output_channels):
                    variables = {
                        row: row, column: column, channel: channel,
                        filter_height: filter_height,
                        filter_width: filter_width
                    }
                    dA_prev[
                        output_index, x_start:x_end, y_start:y_end, channel
                    ] += avg_inv(
                        dA, output_index,
                        variables
                    ) if mode == 'avg' else max_inv(
                        A_prev, output_index, variables, x_start, x_end,
                        y_start, y_end, dA)

    return dA_prev
