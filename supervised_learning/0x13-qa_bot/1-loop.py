#!/usr/bin/env python3
"""
    Script that takes in input from the user with the prompt,
    and prints the answer
"""


if __name__ == '__main__':
    try:
        while True:
            question = input('Q: ')

            if question.lower().strip() in ['quit', 'exit', 'bye']:
                raise ValueError('A: Goodbye')

            print('A: ')
    except Exception as exception:
        print(f"{exception}")
        exit(1)
