#!/usr/bin/env python3
"""
    Script that performs semantic search on a corpus of documents
"""

import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
        Performs semantic search on a corpus of documents
        Args:
            corpus_path (str): the path to the corpus of reference documents
            sentence (str): the sentence from which to perform the search
        Returns:
            the reference text of the document most similar to sentence
    """

    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # Initialize references with the query sentence
    references = [sentence]

    for filename in os.listdir(corpus_path):
        if not filename.endswith(".md"):
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            # Append the reference text to the references list
            references.append(f.read())

    # Compute embedding's scores
    embedding_scores = model(references)

    # Compute the similarity matrix
    # with the dot product of the embedding's scores
    # and the transpose of the embedding's scores
    # (the dot product of the embedding's scores
    # and the embedding's scores is the cosine similarity)
    corr = np.inner(embedding_scores, embedding_scores)

    # Get the most similar document to the query sentence
    closest = np.argmax(corr[0, 1:])

    return references[closest + 1]
