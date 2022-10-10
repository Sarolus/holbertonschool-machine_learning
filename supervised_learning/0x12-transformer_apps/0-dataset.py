#!/usr/bin/env python3
"""
    Script that loads and prepares a dataset for machine translation
"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

MAX_VOCAB_SIZE = 2**15


class Dataset:
    """
        Loads and prepares a dataset for machine translation

        Attributes:
            data_train: contains the ted_hrlr_translate/pt_to_en
                        tf.data.Dataset train split, loaded as_supervided
            data_valid: contains the ted_hrlr_translate/pt_to_en
                        tf.data.Dataset validate split, loaded as_supervided
            tokenizer_pt: Portuguese tokenizer created from the training set
            tokenizer_en: English tokenizer created from the training set
    """

    def __init__(self):
        """
            Constructor
        """

        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
            Creates sub-word tokenizers for our dataset

            Args:
                data: tf.data.Dataset whose examples are formatted as a tuple
                      (pt, en)
                    - pt is the tf.Tensor containing the Portuguese sentence
                    - en is the tf.Tensor containing the corresponding English
                      sentence

            Returns:
                tokenizer_pt, tokenizer_en
                    - tokenizer_pt is the Portuguese tokenizer
                    - tokenizer_en is the English tokenizer
        """

        subword_text_encoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = subword_text_encoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=MAX_VOCAB_SIZE)
        tokenizer_en = subword_text_encoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=MAX_VOCAB_SIZE)

        return tokenizer_pt, tokenizer_en
