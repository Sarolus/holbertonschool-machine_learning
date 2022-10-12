#!/usr/bin/env python3
"""
    Script that answers from a reference text
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
        Answers questions from a reference text
        Args:
            reference (str): the reference text from which to find the answer
        Returns:
            None
    """

    while True:
        question = input('Q: ')

        if question.lower().strip() in ['quit', 'exit', 'bye']:
            raise ValueError('A: Goodbye')

        answer = question_answer(question, reference)

        if not answer:
            print(f"A: Sorry, I do not understand your question.")

        print(f'A: {answer}')
