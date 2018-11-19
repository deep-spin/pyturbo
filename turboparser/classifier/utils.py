import numpy as np
import lzma

'''Several utility functions.'''


def nearly_eq_tol(a, b, tol):
    '''Checks if two numbers are equal up to a tolerance.'''
    return (a-b)*(a-b) <= tol


def nearly_binary_tol(a, tol):
    '''Checks if a number is binary up to a tolerance.'''
    return nearly_eq_tol(a, 0.0, tol) or nearly_eq_tol(a, 1.0, tol)


def nearly_zero_tol(a, tol):
    '''Checks if a number is zero up to a tolerance.'''
    return (a <= tol) and (a >= -tol)


def read_embeddings(path):
    '''
    Read a text file, or xzipped text file, with word embeddings.

    :return: a dictionary mapping words to indices and a numpy array
    '''
    counter = 0
    words = {}
    vectors = []

    open_fn = lzma.open if path.endswith('.xz') else open

    # read text as bytes to allow for catching exceptions when decoding utf-8
    with open_fn(path, 'rb') as f:
        next(f)  # ignore first line
        for line_number, line in enumerate(f, 2):
            line = line.strip()
            if line == '':
                continue

            fields = line.split()
            try:
                word = str(fields[0], 'utf-8')
            except UnicodeDecodeError:
                print('Error reading line %d of embeddings file, skipping' %
                      line_number)
                continue

            words[word] = counter
            counter += 1

            vector = np.array([float(field) for field in fields[1:]])
            vectors.append(vector)

    embeddings = np.array(vectors, dtype=np.float32)
    return words, embeddings
