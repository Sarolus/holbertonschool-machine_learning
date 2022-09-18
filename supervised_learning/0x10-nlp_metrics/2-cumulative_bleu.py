
#!/usr/bin/env python3
"""
    Script that calculates the cumulative ngram BLEU score for a sentence
"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """
        Calculates the cumulative ngram BLEU score for a sentence
        Args:
            references: list of reference translations
                        each reference translation is a list
                        of the  words in the translation
            sentence: list containing the model proposed sentence
            n: the size of the largest n-gram to use for evaluation
        Returns:
            the cumulative ngram BLEU score
    """

    def brevity_penalty(references, sentence):
        """
            Calculates the brevity penalty for a sentence
            Args:
                references: list of reference translations
                            each reference translation is a list of
                            the words in the translation
                sentence: list containing the model proposed sentence
            Returns:
                the brevity penalty
        """

        # Calculate the closest reference length
        closest_ref_len = min(
            [
                len(ref) for ref in references
            ], key=lambda x: abs(len(sentence) - x)
        )

        # Calculate the brevity penalty
        if len(sentence) > closest_ref_len:
            return 1
        else:
            return np.exp(1 - closest_ref_len / len(sentence))

    def clipped_count(references, sentence):
        """
            Calculates the clipped count for a sentence
            Args:
                references: list of reference translations
                            each reference translation is a list of
                            the words in the translation
                sentence: list containing the model proposed sentence
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

    def n_gram_generator(sentence, n=2):
        """
            Generates the n-gram for a sentence
            Args:
                sentence: list containing the model proposed sentence
                n: the size of the n-gram to use for evaluation
            Returns:
                the n-gram for the sentence
        """
        n_grams = []

        for i in range(len(sentence) - n + 1):
            n_grams.append(tuple(sentence[i:i + n]))

        return n_grams

    def cumul_precision(references, sentence, n):
        """
            Calculates the cumulative precision for a sentence
            Args:
                references: list of reference translations
                            each reference translation is a list of
                            the words in the translation
                sentence: list containing the model proposed sentence
                n: the size of the n-gram to use for evaluation
            Returns:
                the cumulative precision
        """

        cumul_bleu = []
        for i in range(1, n + 1):
            n_grams = n_gram_generator(sentence, i)
            n_gram_ref = [
                n_gram_generator(reference, i) for reference in references
            ]
            cc = clipped_count(n_gram_ref, n_grams)
            precision = cc / len(n_grams)
            cumul_bleu.append(precision)

        return cumul_bleu

    bp = brevity_penalty(references, sentence)
    cp = np.exp(
        np.sum(
            1/n * np.log(
                np.array(cumul_precision(references, sentence, n))
            )
        )
    )

    return bp * cp
