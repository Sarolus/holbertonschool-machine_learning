#!/usr/bin/env python3
"""
    N-gram BLEU Score
"""

import numpy as np


def ngram_bleu(references, sentence, n):
    """
        Calculates the ngram BLEU score for a sentence

        Args:
            references: list of reference translations
                        each reference translation is a list of the
                        words in the translation
            sentence: list containing the model proposed sentence
            n: the size of the n-gram to use for evaluation

        Returns:
            the ngram BLEU score
    """

    def brevity_penalty(references, sentence):
        """
            Calculates the brevity penalty for a sentence

            Args:
                references: list of reference translations
                            each reference translation is a list of the
                            words in the translation
                sentence: list containing the model proposed sentence

            Returns:
                the brevity penalty
        """

        # Calculate the closest reference length
        closest_ref_len = min([len(ref) for ref in references],
                              key=lambda x: abs(len(sentence) - x))

        # Calculate the brevity penalty
        if len(sentence) > closest_ref_len:
            return 1
        else:
            return np.exp(1 - closest_ref_len / len(sentence))

    def clipped_count(references, sentence, n):
        """
            Calculates the clipped count for a sentence

            Args:
                references: list of reference translations
                            each reference translation is a list of the
                            words in the translation
                sentence: list containing the model proposed sentence
                n: the size of the n-gram to use for evaluation

            Returns:
                the clipped count
        """

        # Calculate the n-grams for the sentence
        sentence_ngrams = [tuple(sentence[i:i + n])
                           for i in range(len(sentence) - n + 1)]

        # Calculate the n-grams for the references
        references_ngrams = [
            [tuple(ref[i:i + n]) for i in range(
                len(ref) - n + 1)] for ref in references]

        # Calculate the clipped count
        clipped_count = 0
        for ngram in sentence_ngrams:
            if any([ngram in ref_ngrams for ref_ngrams in references_ngrams]):
                clipped_count += 1

        return clipped_count

    brevity_penaly = brevity_penalty(references, sentence)
    clipped_count = clipped_count(references, sentence, n)

    return brevity_penaly * np.exp(
        np.log(clipped_count) - np.log(len(sentence) - n + 1))
