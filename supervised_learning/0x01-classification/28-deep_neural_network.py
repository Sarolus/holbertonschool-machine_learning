#!/usr/bin/env python3
"""_
    Binary Classification Deep Neural Network model.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
        Class that defines a deep neural network with one hidden layer
        performing binary classification.

        Attributes:
            L (list) : The number of layers in the neural network.
            cache (dict) : A dictionary to hold all intermediary values
            of the network.
            weights (dict) : A dictionary to hold all weights and biased
            of the network.
    """

    __L = []
    __cache = {}
    __weights = {}
    __activation = None

    def __init__(self, nx: int, layers: list, activation='sig'):
        """
            Initialize the deep neural network class attributes.

            Args:
                nx (int): The number of input features.
                layers (list): The list representing the number of nodes
                in each layer of the network.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")

        if any(list(map(lambda x: x <= 0, layers))):
            raise TypeError("layers must be a list of positive integers")

        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if activation not in ["sig", "tanh"]:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__activation = activation

        for layer_nb in range(self.__L):
            current_layer = layers[layer_nb]

            if layer_nb == 0:
                self.__weights["W" + str(layer_nb + 1)] = np.random.randn(
                    current_layer, nx
                ) * np.sqrt(2 / nx)

            else:
                previous_layer = layers[layer_nb - 1]
                self.__weights["W" + str(layer_nb + 1)] = np.random.randn(
                    current_layer, previous_layer
                ) * np.sqrt(2 / previous_layer)

            self.__weights["b" + str(layer_nb + 1)] = np.zeros(
                (current_layer, 1)
            )

    @property
    def L(self):
        """
            Setter method for the number of layers in the neural network.

            Returns:
                list: Returns the number of layers in the neural network.
        """

        return self.__L

    @property
    def cache(self):
        """
            Setter method for the dictionary to hold all intermediary values
            of the network.

            Returns:
                dict: Returns the dictionary to hold all intermediary values
                of the network.
        """

        return self.__cache

    @property
    def weights(self):
        """
            Setter method for the dictionary to hold all weights and biased
            of the network.

            Returns:
                dict: Returns the dictionary to hold all weights and biased
                of the network.
        """

        return self.__weights

    @property
    def activation(self):
        """
            Setter method for the type of activation function used in the
            hidden layers.

            Returns:
                dict: Returns the type of activation function used in the
                hidden layers.
        """

        return self.__activation

    def softmax(self, Z):
        """
            Return the result of a softmax activation function.
        """
        exp = np.exp(Z)
        return exp / exp.sum(axis=0, keepdims=True)

    def sigmoid(self, Z):
        """
            Return the result of a sigmoid activation function.
        """
        return 1 / (1 + np.exp(-Z))

    def tanh(self, Z):
        """
            Returns the result of a tanh activation function.
        """
        return np.tanh(Z)

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the deep neural network.

            Args:
                X (numpy.ndarray): numpy.ndarray whit shape (nx, m) that
                contains the input data.

            Returns:
                int: Returns the Activated Ouputs of the neural network
                and the cache.
        """

        self.__cache["A0"] = X

        for layer_index in range(self.__L):
            a_previous = self.__cache["A" + str(layer_index)]
            w = self.__weights["W" + str(layer_index + 1)]
            b = self.__weights["b" + str(layer_index + 1)]
            Z = np.dot(w, a_previous) + b

            if layer_index == self.__L - 1:
                # Softmax Activation Function for Output Layer.
                self.__cache["A" + str(layer_index + 1)] = self.softmax(Z)
            elif self.__activation == "tanh":
                # Tanh Activation Function for Hidden layers.
                self.__cache["A" + str(layer_index + 1)] = self.tanh(Z)
            else:
                # Sigmoid Activation Function for Hidden layers.
                self.__cache["A" + str(layer_index + 1)] = self.sigmoid(Z)

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.

            Args:
                Y (np.ndarray): one-hot np.ndarray that contains the correct
                labels for the input data.
                A (np.ndarray): np.ndarray that contains the activated output
                of the neuron for each example.

            Returns:
                int: Returns the cost of the model.
        """

        return -1 / Y.shape[1] * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """
            Evaluates the neural network's predictions.

            Args:
                X (np.ndarray): np.ndarray that contains the input features
                to the neuron.
                Y (np.ndarray): np.ndarray that contains the correct labels
                for the input data.

            Returns:
                np.ndarray: Returns the neuron's prediction and the cost of
                the network, respectively.
        """

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        hot = self.one_hot_decode(A)
        prediction = self.one_hot_encode(hot, Y.shape[0])

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """_summary_

        Args:
            Y (np.ndarray): np.ndarray that contains the correct labels for
            the input data.
            cache (dict): Dictionary that contains all the intermediary values
            of the network.
            alpha (float, optional): The learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]
        weights = self.__weights.copy()

        for layer_index in range(self.__L, 0, -1):
            A = cache["A" + str(layer_index)]

            if layer_index == self.__L:
                dz = A - Y
            else:
                if self.activation == "sig":
                    dz = np.dot(
                        weights["W" + str(layer_index + 1)].T, dz) *\
                        A * (1 - A)
                else:
                    dz = np.dot(
                        weights["W" + str(layer_index + 1)].T, dz) *\
                        (1 - np.power(A, 2))

            dw = 1 / m * np.dot(dz, cache["A" + str(layer_index - 1)].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)

            self.__weights["W" + str(layer_index)] = weights[
                "W" + str(layer_index)] - alpha * dw
            self.__weights["b" + str(layer_index)] = weights[
                "b" + str(layer_index)] - alpha * db

    def train(
        self, X: np.ndarray,
        Y: np.ndarray,
        iterations: int = 5000,
        alpha: float = 0.05,
        verbose: bool = True,
        graph: bool = True,
        step: int = 100
    ):
        """
            Trains the neuron by updating the private attributes __W,
            __b, __A.

            Args:
                X (np.ndarray): np.ndarray that contains the input data.
                Y (np.ndarray): np.ndarray that contains the correct labels for
                the input data.
                iterations (int, optional): The number of iterations to train
                over. Defaults to 5000.
                alpha (float, optional): The learning rate of the neuron.
                Defaults to 0.05.
                verbose (bool, optional): _description_. Defaults to True.
                graph (bool, optional): _description_. Defaults to True.
                step (int, optional): _description_. Defaults to 100.

            Returns:
                np.ndarray: Returns the evaluation of the training data after
                iterations of training have occured.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")

        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")

        if alpha < 0:
            raise ValueError("alpha must be positive")

        if graph or verbose:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        x = []
        y = []

        for index in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose and index % step == 0:
                print("Cost after {} iterations: {}".format(
                    index, self.cost(Y, A)))

            if graph:
                x.append(index)
                y.append(self.cost(Y, A))

        if len(x) > 0 and len(y) > 0:
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Saves the instance object to a file in pickle format

            Args:
                filename (str): The file to which the object should be saved.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
            Loads a pickled DeepNeuralNetwork object.

            Args:
                filename (str): The file from which the object should
                be loaded.
        """

        try:
            if not filename.endswith(".pkl"):
                filename += ".pkl"

            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def one_hot_encode(self, Y, classes):
        """
            Converts a numeric label vector into a one-hot matrix

            Args:
                Y (np.ndarray): np.ndarray containing numeric class labels.
                classes (int): The maximum number of classes found in Y.

            Returns:
                np.ndarray: A one-hot encoding of Y wit shape (classes, m),
                or None on failure.
        """
        if type(Y) is not np.ndarray:
            return None

        if type(classes) is not int:
            return None

        try:
            return np.eye(classes)[Y].T
        except Exception:
            return None

    def one_hot_decode(self, one_hot):
        """
            Converts a one-hot matrix into a vector of labels

            Args:
                one_hot (np.ndarray): one-hot encoded np.ndarray with shape
                (classes, m).

            Returns:
                np.ndarray: np.ndarray with shape (m,) containing the numeric
                lagbels for each example, or None on failure.
        """
        if type(one_hot) is not np.ndarray:
            return None

        if len(one_hot.shape) != 2:
            return None

        try:
            return np.argmax(one_hot, axis=0)
        except Exception:
            return None
