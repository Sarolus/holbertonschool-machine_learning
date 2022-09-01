#!/usr/bin/env python3
"""
    Forward propagation for a LSTM
"""

import numpy as np


class GRUCell:
    """
        GRU Cell class.

        Attributes:
            Wz: matrix of weights for the update gate
            Wr: matrix of weights for the reset gate
            Wh: matrix of weights for the hidden state
            Wy: matrix of weights for the output
            bz: vector of biases for the update gate
            br: vector of biases for the reset gate
            bh: vector of biases for the hidden state
            by: vector of biases for the output
            i: dimension of the input
            h: dimension of the hidden state
            o: dimension of the output
    """

    def __init__(self, i, h, o):
        """
            Constructor.

            Args:
                i: dimension of the input
                h: dimension of the hidden state
                o: dimension of the output
        """

        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.i = i
        self.h = h
        self.o = o

    def forward(self, h_prev, x_t):
        """
            Performs the forward step.

            Args:
                h_prev: previous hidden state
                x_t: input

            Returns:
                h_next: next hidden state
                y: output
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        concat = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(np.dot(concat, self.Wz) + self.bz)
        r = sigmoid(np.dot(concat, self.Wr) + self.br)

        r_concat = np.concatenate((r * h_prev, x_t), axis=1)
        h_candidate = np.tanh(np.dot(r_concat, self.Wh) + self.bh)
        h_next = z * h_candidate + (1 - z) * h_prev
        y = np.dot(h_next, self.Wy) + self.by
        y = softmax(y)

        return h_next, y
