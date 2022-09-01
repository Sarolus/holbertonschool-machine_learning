#!/usr/bin/env python3
"""
    Deep RNN
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
        Perform deep RNN

        Args:
            rnn_cells: list of RNNCell instances
            X: numpy.ndarray shape (m, t, d)
                input data
            h_0: numpy.ndarray shape (n, d)
                initial hidden state

        Returns:
            H: numpy.ndarray shape (n, t, d)
                hidden state at each time step
            Y: numpy.ndarray shape (n, t, d)
                output at each time step
    """

    t, m, d = X.shape
    n = len(rnn_cells)
    _, _, h = h_0.shape

    H = np.zeros((t + 1, n, m, h))
    Y = np.zeros((t, m, rnn_cells[n - 1].Wy.shape[1]))
    H[0, :] = h_0

    for iteration in range(t):
        for rnn_cell_index, rnn_cell in enumerate(rnn_cells):
            x_t = X[iteration] if rnn_cell_index == 0 else H[
                iteration + 1, rnn_cell_index - 1
            ]

            H[iteration + 1, rnn_cell_index], Y[iteration] = rnn_cell.forward(
                H[iteration, rnn_cell_index], x_t
            )

    return H, Y
