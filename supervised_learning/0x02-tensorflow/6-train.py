#!/usr/bin/env python3
"""
    Tensorflow Training Module
"""
import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
    X_train, Y_train,
    X_valid, Y_valid,
    layer_sizes,
    activations,
    alpha,
    iterations,
    save_path="/tmp/model.ckpt"
):
    """
        Build, trains and saves a neural network classifier

        Args:
            X_train: The training input data
            Y_train: The training output data
            X_valid: The validation input data
            Y_valid: The validation output data
            layer_sizes: A list containing the number of nodes in each layer
            activations: A list containing the activation functions
            alpha: The learning rate
            iterations: The number of iterations
            save_path: The path to save the model

        Returns:
            save_path: The path to save the model
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    accuracy = calculate_accuracy(y, y_pred)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(initializer)

        for i in range(iterations + 1):
            train_acc, train_cost = session.run([accuracy, loss],
                                                feed_dict={x: X_train,
                                                           y: Y_train
                                                           }
                                                )
            valid_acc, valid_cost = session.run([accuracy, loss],
                                                feed_dict={x: X_valid,
                                                           y: Y_valid
                                                           }
                                                )

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))

            if i < iterations:
                session.run(train_op, feed_dict={x: X_train, y: Y_train})

        return saver.save(session, save_path)
