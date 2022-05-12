#!/usr/bin/env python3
"""
    Tensorflow Training Module
"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
        Evaluates the output of a neural network.

        Args:
            X (np.ndarray): np.ndarray containing the input
            data to evaluate.
            Y (np.ndarray): np.ndarray containing the one-hot
            labels for X.
            save_path (string): The location to load the model
            from.

        Returns:
            The network's prediction, accuracy and loss.
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(session, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        y_eval, accuracy_eval, loss_eval = session.run(
            [y_pred, accuracy, loss],
            feed_dict={x: X, y: Y}
        )

        return y_eval, accuracy_eval, loss_eval
