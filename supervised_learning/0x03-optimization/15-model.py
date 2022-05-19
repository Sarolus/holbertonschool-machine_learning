#!/usr/bin/env python3
"""
    Model Building,Training & Saving Module
"""

import tensorflow.compat.v1 as tf
import numpy as np


def forward_prop(prev, layers, activations, epsilon):
    """
        Creates forward propgation with batch normalization
        for a neural network in tensorflow.

        Args:
            prev (tensor): The activated output of the previous layer.
            layers (list): List containing the number of nodes in each
            layer of the network.
            activations (list): The activation function list that should be
            used on the output of the layer.
            epsilon (int): A small number used to avoid division by zero.

        Returns:
            tensor: A tensor of the activated output for the neural network.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    batch_norm_outputs = prev

    for index in range(len(layers) - 1):
        layer = tf.keras.layers.Dense(units=layers[index],
                                      kernel_initializer=initializer)

        mean, variance = tf.nn.moments(layer(batch_norm_outputs), axes=[0])

        gamma = tf.Variable(tf.ones(layers[index]), trainable=True)
        beta = tf.Variable(tf.zeros(layers[index]), trainable=True)

        batch_norm = tf.nn.batch_normalization(layer(batch_norm_outputs),
                                               mean=mean,
                                               variance=variance,
                                               offset=beta,
                                               scale=gamma,
                                               variance_epsilon=epsilon)

        batch_norm_outputs = activations[index](batch_norm)

    output_layer = tf.keras.layers.Dense(layers[-1],
                                         activation=None,
                                         kernel_initializer=initializer)

    NN_output = output_layer(batch_norm_outputs)

    return NN_output


def create_placeholders(nx, classes):
    """
        Creates the placeholders needed for the model

        Args:
            nx: number of input features
            classes: number of classes

        Returns:
            x: placeholder for the input data
            y: placeholder for the input labels
    """

    x = tf.placeholder("float", [None, nx.shape[1]], name='x')
    y = tf.placeholder("float", [None, classes.shape[1]], name='y')

    return x, y


def calculate_loss(y, y_pred):
    """
        Calculates the loss of a prediction

        Args:
            y: real labels
            y_pred: predicted labels

        Returns:
            loss: loss of the prediction
    """

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss


def calculate_accuracy(y, y_pred):
    """
        Calculates the accuracy of a prediction

        Args:
            y: real labels
            y_pred: predicted labels

        Returns:
            accuracy: accuracy of the prediction
    """

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def create_train_op(loss, alpha, beta1, beta2, epsilon):
    """
        Creates the operation to perform the Adam optimization

        Args:
            loss: tensorflow tensor for the loss
            alpha: the learning rate
            beta1: the momentum term for the first moment
            beta2: the momentum term for the second moment
            epsilon: small number to avoid division by zero in Adam

        Returns:
            train_op: the Adam optimization operation
    """

    return tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        Creates the operation to perform the learning rate decay

        Args:
            alpha: learning rate
            decay_rate: decay rate
            global_step: global step
            decay_step: decay step

        Returns:
            learning_rate: learning rate
    """

    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )


def shuffle_data(X, Y):
    """
        Shuffles the data

        Args:
            X: numpy.ndarray of shape (n, m) containing the input data
            Y: numpy.ndarray of shape (n, 1) containing the data labels

        Returns:
            X_shuffled: numpy.ndarray of shape (n, m) containing the shuffled
                        input data
            Y_shuffled: numpy.ndarray of shape (n, 1) containing the shuffled
                        labels
    """

    permutation = np.random.permutation(Y.shape[0])
    X = X[permutation, :]
    Y = Y[permutation]

    return X, Y


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
        Build, trains and saves a neural network classifier

        Args:
            Data_train: tuple containing training input data and training
                        labels
            Data_valid: tuple containing validation input data and validation
                        labels
            layers: a list of integers that represent the number of nodes in
                    each layer
            activations: a list of strings that represent the activation
                         functions for each layer
            alpha: learning rate
            beta1: weight for the first moment in Adam
            beta2: weight for the second moment in Adam
            epsilon: small number to avoid division by zero
            decay_rate: decay rate for inverse time decay of the learning rate
            batch_size: number of data points in each batch
            epochs: number of times the training data is passed
            save_path: path to save the model

        Returns:
            save_path: path to save the model
    """

    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection
    x, y = create_placeholders(X_train, Y_train)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    # intialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # intialize global_step variable
    global_step = tf.Variable(0, trainable=False)

    # compute decay_steps
    decay_steps = 1

    # create "alpha" the learning rate decay operation in tensorflow
    alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_steps)

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    train_op = create_train_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    store = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        m = X_train.shape[0]
        steps = m // batch_size + 1

        for epoch in range(epochs + 1):
            train_accuracy, train_cost = session.run(
                [accuracy, loss], feed_dict={x: X_train, y: Y_train}
            )
            valid_accuracy, valid_cost = session.run(
                [accuracy, loss], feed_dict={x: X_valid, y: Y_valid}
            )

            # print training and validation cost and accuracy
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch == epochs:
                break

            # shuffle data
            x_shuffle, y_shuffle = shuffle_data(X_train, Y_train)

            for step in range(steps):
                start = batch_size * step
                end = batch_size * (step + 1)

                # get X_batch and Y_batch from X_train shuffled and Y_train
                # shuffled
                x_batch = x_shuffle[start:end]
                y_batch = y_shuffle[start:end]

                # run training operation
                session.run(
                    train_op,
                    feed_dict={x: x_batch, y: y_batch}
                )

                # print batch cost and accuracy
                if (step + 1) % 100 == 0:
                    step_accuracy, step_cost = session.run(
                        [accuracy, loss], feed_dict={x: x_batch, y: y_batch}
                    )

                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
            session.run(tf.assign(global_step, global_step + 1))

        # save and return the path to where the model was saved
        return store.save(session, save_path)
