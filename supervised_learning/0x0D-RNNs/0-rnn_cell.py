#!/usr/bin/env python3
"""
    Simple RNN Cell
"""

import numpy as np


class RNNCell:
    """
        Simple RNN

        Attributes:
            Wh: matrix of weights for the hidden state
            Wy: matrix of weights for the output
            bh: vector of biases for the hidden state
            by: vector of biases for the output
            i: dimension of the input
            h: dimension of the hidden state
            o: dimension of the output
    """

    def __init__(self, i, h, o):
        """
            Constructor

            Args:
                i: dimension of the input
                h: dimension of the hidden state
                o: dimension of the output
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.i = i
        self.h = h
        self.o = o

    def forward(self, h_prev, x_t):
        """
            Performs the forward step

            Args:
                h_prev: previous hidden state
                x_t: input

            Returns:
                h_next: next hidden state
                y: output
        """

        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
