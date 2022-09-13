#!/usr/bin/env python3
"""
    Gensim Word2Vec model
"""

from gensim.models import Word2Vec


def word2vec_model(
    sentences, size=100, min_count=5, window=5,
    negative=5, cbow=True, iterations=5, seed=0, workers=1
):
    """
        Creates and trains a gensim word2vec model

        Args:
            sentences (list): A list of sentences to train the model on
            size (int): The dimensionality of the embedding layer
            min_count (int): The minimum number of occurrences of a word
                             for use
            window (int): The maximum distance between the current and
                          predicted word within a sentence
            negative (int): If > 0, negative sampling will be used, the int for
                            negative specifies how many “noise words” should be
                            drawn (usually between 5-20).
            cbow (bool): Determines the training algorithm. If True, CBOW is
                         used. Otherwise, skip-gram is employed.
            iterations (int): The number of iterations (epochs) over the corpus
            seed (int): The seed for the random number generator. Initial
                        vectors for each word are seeded with a hash of the
                        concatenation of word + str(seed). Note that for a
                        fully deterministically-reproducible run, you must also
                        limit the model to a single worker thread, to eliminate
                        ordering jitter from OS thread scheduling. In Python 3,
                        reproducibility between interpreter launches also
                        requires use of the PYTHONHASHSEED environment variable
                        to control hash randomization.
            workers (int): Use these many worker threads to train the model
                           (=faster training with multicore machines)

        Returns:
            model (Word2Vec): The trained model
    """

    # https://radimrehurek.com/gensim/models/word2vec.html
    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
    model = Word2Vec(
        sentences=sentences,
        size=size,
        window=window,
        min_count=min_count,
        negative=negative,
        seed=seed,
        workers=workers,
        sg=not cbow,
        iter=iterations,
    )

    return model
