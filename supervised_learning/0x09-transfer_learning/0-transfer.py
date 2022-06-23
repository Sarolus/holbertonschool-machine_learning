#!/usr/bin/env python3
"""
    Transfer Learning Using Keras Application Module
"""
import tensorflow.keras as keras
import tensorflow as tf


def preprocess_data(X, Y):
    """
        Pre-processes the data for our model.

        Args:
            X (np.ndarray): Contains the CIFAR 10 data, where
            m is the number of data points.
            Y (np.ndarray): Contains the CIFAR 10 labels for X.

        Returns:
            np.ndarray: The preprocessed X datas and Y labels,
            respectively.
    """

    X_p = keras.applications.inception_resnet_v2.preprocess_input(X)
    Y_p = keras.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


if __name__ == "__main__":

    callbacks = []
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Create our base model
    base_model = keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3)
    )

    # Freeze our base model
    base_model.trainable = False

    # Create new model on top.
    inputs = keras.Input(shape=(32, 32, 3))
    input = keras.layers.Lambda(
        lambda image: tf.image.resize(
            image,
            (128, 128)
        )
    )(inputs)

    X = base_model(input, training=False)
    X = keras.layers.GlobalAveragePooling2D()(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(
        units=1000,
        activation="relu"
    )(X)
    X = keras.layers.Dropout(0.5)(X)
    X = keras.layers.BatchNormalization()(X)
    outputs = keras.layers.Dense(
        units=500,
        activation="relu"
    )(X)
    X = keras.layers.Dropout(0.2)(X)
    X = keras.layers.BatchNormalization()(X)

    outputs = keras.layers.Dense(
        units=10,
        activation="softmax"
    )(X)

    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam()

    # Compile our model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy']
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
    )
    callbacks.append(early_stopping)

    # Start the training of our model.
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=128,
        epochs=4,
        callbacks=callbacks
    )
    
    # Unfreeze the base_model
    base_model.trainable = True
    
    optimizer = keras.optimizers.Adam(1e-5)
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=128,
        epochs=1,
        verbose=1,
        callbacks=callbacks
    )

    model.save("cifar10.h5")
