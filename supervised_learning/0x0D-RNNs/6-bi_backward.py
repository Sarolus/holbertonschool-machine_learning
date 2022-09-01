#!/usr/bin/env python3
"""
    Represents a bidirectional cell of an RNN
"""

import numpy as np


class BidirectionalCell:
    """
        Represents a bidirectional cell of an RNN

        Attributes:
            Whf (np.ndarray): Hidden to hidden weights for the forward
                direction
            Whb (np.ndarray): Hidden to hidden weights for the backward
                direction
            Wy (np.ndarray): Hidden to output weights
            bhf (np.ndarray): Hidden bias for the forward direction
            bhb (np.ndarray): Hidden bias for the backward direction
            by (np.ndarray): Output bias
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
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.i = i
        self.h = h
        self.o = o

    def directional_switch(self, h, input, weights, biases):
        """
            Performs forward (or backward) propagation fo one time step

            Args:
                h: previous hidden state
                input: input
                weights: weights
                biases: biases

            Returns:
                h_next: next hidden state
        """

        return np.tanh(np.dot(np.hstack((h, input)), weights) + biases)

    def forward(self, h_prev, x_t):
        """
            Performs forward propagation fo one time step

            Args:
                h_prev: previous hidden state
                x_t: input

            Returns:
                h_next: next hidden state
        """

        return self.directional_switch(h_prev, x_t, self.Whf, self.bhf)

    def backward(self, h_next, x_t):
        """
            Calculates the hidden state in the backward direction
            for one time step

            Args:
                h_next: next hidden state
                x_t: input

            Returns:
                h_prev: previous hidden state
        """

        return self.directional_switch(h_next, x_t, self.Whb, self.bhb)
