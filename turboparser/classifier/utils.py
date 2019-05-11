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


def read_embeddings(path, extra_symbols=None, max_words=1000000):
    '''
    Read a text file, or xzipped text file, with word embeddings.

    :param path: path to the embeddings file
    :param extra_symbols: extra symbols such as UNK to create embeddings for.
        They are placed in the beginning of the matrix and NOT in the
        dictionary.
    :param max_words: maximum word embeddings to read (the rest will be ignored)
    :return: a dictionary mapping words to indices and a numpy array
    '''
    vectors = []
    words = {}
    counter = 0

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

            # use a counter instead of len(words) to avoid problems with
            # repeated words
            words[word] = counter
            counter += 1

            vector = np.array([float(field) for field in fields[1:]])
            vectors.append(vector)

            if len(words) == max_words:
                break

    assert len(vectors) == len(words)
    embeddings = np.array(vectors, dtype=np.float32)
    if extra_symbols is not None:
        shape = (len(extra_symbols), embeddings.shape[1])
        extra_embeddings = np.zeros(shape, dtype=embeddings.dtype)
        embeddings = np.concatenate([extra_embeddings, embeddings], 0)

    return words, embeddings
