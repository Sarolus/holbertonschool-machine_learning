#!/usr/bin/env python3
"""
    TF-IDF EMbedding
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def tf_idf(sentences, vocab=None):
    """
        Creates a bag of words embedding mattrix

        Args:
            sentences (list): A list of sentences to turn into a bag of words
            vocab (list): A list of words to use as the vocabulary

        Returns:
            embeddings (np.ndarray): A numpy.ndarray of shape (s, f) containing
                                     the embeddings where s is the number of
                                     sentences in sentences and f is the number
                                     of features analyzed.
            vocab (list): The vocabulary list
    """

    # https://www.analyticsvidhya.com/blog/2021/08/a-friendly-guide-to-nlp-bag-of-words-with-python-example/
    vectorizer = CountVectorizer(vocabulary=vocab)
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    # https://stackoverflow.com/questions/70640923/countvectorizer-object-has-no-attribute-get-feature-names-out
    X = vectorizer.fit_transform(sentences)

    # https://www.analyticsvidhya.com/blog/2021/08/a-friendly-guide-to-nlp-bag-of-words-with-python-example/
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)

    return tfidf.toarray(), vectorizer.get_feature_names()
