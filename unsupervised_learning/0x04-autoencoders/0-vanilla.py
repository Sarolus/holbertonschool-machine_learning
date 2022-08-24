#!/usr/bin/env python3
"""
    Vanilla Autoencoder Module
"""
import tensorflow.keras as keras


def build_encoder(input, hidden_layers, latent_dims):
    """
        Creates an encoder.

    Args:
        input (keras.layer): The input layer.
        hidden_layers (list): The number of nodes in each
                              hidden layers.
        latent_dims (int): The dimensions of the latent space
                           representation.

    Returns:
        model: _description_
    """
    encoder_hidden_layers = input

    for layer in hidden_layers:
        encoder_hidden_layers = keras.layers.Dense(
            layer,
            activation='relu',
        )(encoder_hidden_layers)

    encoder_output_layer = keras.layers.Dense(
        latent_dims,
        activation='relu',
    )(encoder_hidden_layers)

    return keras.Model(inputs=input, outputs=encoder_output_layer)


def build_decoder(input, hidden_layers, input_dims):
    """
        Creates a decoder.

        Args:
            input (keras.layer): The input layer.
            hidden_layers (list): The number of nodes in each
                                hidden layers.
            input_dims (int): The dimensions of the model input.

        Returns:
            model: The decoder model.
    """

    decoder_hidden_layers = input

    for layer in reversed(hidden_layers):
        decoder_hidden_layers = keras.layers.Dense(
            layer,
            activation='relu',
        )(decoder_hidden_layers)

    decoder_output_layer = keras.layers.Dense(
        input_dims,
        activation='sigmoid',
    )(decoder_hidden_layers)

    return keras.Model(inputs=input, outputs=decoder_output_layer)


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
        Creates an autoencoder.

        Args:
            input_dims (int): The dimensions of the model input
            hidden_layers (list): The number of nodes of each
                                  hidden layers.
            latent_dims (int): The dimensions of the latent space
                               representation.

        Returns:
            models: The encoder model, the decoder model &
                    the autoencoder model.
    """

    # Create the input layer
    input_layer = keras.layers.Input(shape=(input_dims,))

    # Create encoder
    encoder = build_encoder(input_layer, hidden_layers, latent_dims)

    # Create decoder input layer
    decoder_input_layer = keras.layers.Input(shape=(latent_dims,))

    # Create decoder
    decoder = build_decoder(decoder_input_layer, hidden_layers, input_dims)

    # Create autoencoder using encoder & decoder
    autoencoder = keras.Model(
        inputs=input_layer,
        outputs=decoder(encoder(input_layer))
    )

    # Compile the model with the adam optimizer & binary_crossentropy loss
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
