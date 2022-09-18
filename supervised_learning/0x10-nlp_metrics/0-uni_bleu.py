#!/usr/bin/env python3
"""
    Unigram BLEU Score
"""

import numpy as np


def uni_bleu(references, sentence):
    """
        Calculates the unigram BLEU score for a sentence

        Args:
            references: list of reference translations
                        each reference translation is a list of the
                        words in the translation
            sentence: list containing the model proposed sentence

        Returns:
            the unigram BLEU score
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

        word_count = {}

        # Calculate the word count for the sentence
        for reference in references:
            # Calculate the n-gram for the reference
            for word in sentence:
                word_iteration = reference.count(word)

                # Check if the word is in the word count
                if word in word_count:
                    # Check if the word iteration is greater than the
                    if word_count[word] < word_iteration:
                        word_count.update({word: word_iteration})
                else:
                    word_count.update({word: word_iteration})

        return sum(word_count.values())

    brevity_penalty = brevity_penalty(references, sentence)
    clipped_count = clipped_count(references, sentence, 1)

    return brevity_penalty * clipped_count / len(sentence)
