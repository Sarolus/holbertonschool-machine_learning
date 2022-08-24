#!/usr/bin/env python3
"""
    Vanilla Autoencoder Module
"""
import tensorflow.keras as keras


def build_encoder(input, filters):
    """
        Creates an encoder.

    Args:
        input (keras.layer): The input layer.
        filters (list): The list containing the filters for
                        each convolutional layer.
        latent_dims (int): The dimensions of the latent space
                           representation.

    Returns:
        model: _description_
    """
    encoder_hidden_layers = input

    for kernel in filters:
        encoder_hidden_layers = keras.layers.Conv2D(
            filters=kernel,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(encoder_hidden_layers)
        encoder_hidden_layers = keras.layers.MaxPooling2D(
            pool_size=2,
            padding="same",
        )(encoder_hidden_layers)

    return keras.Model(inputs=input, outputs=encoder_hidden_layers)


def build_decoder(input, filters, input_dims):
    """
        Creates a decoder.

        Args:
            input (keras.layer): The input layer.
            filters (list): The list containing the filters for
                            each convolutional layer.
            input_dims (int): The dimensions of the model input.

        Returns:
            model: The decoder model.
    """

    decoder_hidden_layers = input

    for kernel in reversed(filters[1:]):
        decoder_hidden_layers = keras.layers.Conv2D(
            filters=kernel,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(decoder_hidden_layers)
        decoder_hidden_layers = keras.layers.UpSampling2D(
            size=2,
        )(decoder_hidden_layers)

    decoder_hidden_layers = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=3,
        activation='relu',
        padding='valid'
    )(decoder_hidden_layers)
    decoder_hidden_layers = keras.layers.UpSampling2D(
        size=2,
    )(decoder_hidden_layers)

    decoder_output_layer = keras.layers.Conv2D(
        filters=input_dims[2],
        kernel_size=3,
        activation='sigmoid',
        padding='same',
    )(decoder_hidden_layers)

    return keras.Model(inputs=input, outputs=decoder_output_layer)


def autoencoder(input_dims, filters, latent_dims):
    """
        Creates an autoencoder.

        Args:
            input_dims (int): The dimensions of the model input
            filters (list): The list containing the filters for
                            each convolutional layer.
            latent_dims (int): The dimensions of the latent space
                               representation.

        Returns:
            models: The encoder model, the decoder model &
                    the autoencoder model.
    """

    # Create the input layer
    input_layer = keras.layers.Input(shape=input_dims)

    # Create encoder
    encoder = build_encoder(input_layer, filters)

    # Create decoder input layer
    decoder_input_layer = keras.layers.Input(shape=latent_dims)

    # Create decoder
    decoder = build_decoder(decoder_input_layer, filters, input_dims)

    # Create autoencoder using encoder & decoder
    autoencoder = keras.Model(
        inputs=input_layer,
        outputs=decoder(encoder(input_layer))
    )

    # Compile the model with the adam optimizer & binary_crossentropy loss
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
