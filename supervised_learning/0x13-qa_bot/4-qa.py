#!/usr/bin/env python3
"""
    Script that answers from multiple reference texts
"""

specific_question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
        Answers questions from multiple reference texts
        Args:
            corpus_path (str): the path to the corpus of reference documents
    """

    while True:
        question = input('Q: ')

        if question.lower().strip() in ['quit', 'exit', 'bye']:
            raise ValueError('A: Goodbye')

        reference = semantic_search(corpus_path, question)
        answer = specific_question_answer(question, reference)

        if not answer:
            print(f"A: Sorry, I do not understand your question.")

        print(f'A: {answer}')
