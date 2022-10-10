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

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            examples['train'])

        self.data_train = examples['train'].map(self.tf_encode)
        self.data_valid = examples['validation'].map(self.tf_encode)

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

    def encode(self, pt, en):
        """
            Encodes a translation into tokens

            Args:
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English sentence

            Returns:
                pt_tokens, en_tokens
                    - pt_tokens is a np.ndarray containing the Portuguese tokens
                    - en_tokens is a np.ndarray. containing the English tokens
        """

        pt_tokens = [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy()) + \
            [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy()) + \
            [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
            Acts as a tensorflow wrapper for the encode instance method

            Args:
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English sentence

            Returns:
                pt_tokens, en_tokens
                    - pt_tokens is a tf.Tensor containing the Portuguese tokens
                    - en_tokens is a tf.Tensor containing the English tokens
        """

        pt_tokens, en_tokens = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
