#!/usr/bin/env python3
"""
    Gensim Word2Vec model to keras
"""


def gensim_to_keras(model):
    """
        Converts a gensim word2vec model to a keras Embedding layer

        Args:
            model (gensim.models.word2vec.Word2Vec): The gensim word2vec model

        Returns:
            The keras Embedding layer
    """

    return model.wv.get_keras_embedding(train_embeddings=True)
