import numpy as np
import lzma
import logging

'''Several utility functions.'''


logger = None


def nearly_eq_tol(a, b, tol):
    '''Checks if two numbers are equal up to a tolerance.'''
    return (a-b)*(a-b) <= tol


def nearly_binary_tol(a, tol):
    '''Checks if a number is binary up to a tolerance.'''
    return nearly_eq_tol(a, 0.0, tol) or nearly_eq_tol(a, 1.0, tol)


def nearly_zero_tol(a, tol):
    '''Checks if a number is zero up to a tolerance.'''
    return (a <= tol) and (a >= -tol)


def configure_logger(verbose):
    """Configure the log to be used by Turbo Parser"""
    level = logging.DEBUG if verbose else logging.INFO
    # logging.basicConfig(level=level)

    logger = logging.getLogger('TurboParser')
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def get_logger():
    '''
    Return the default logger used by Turbo Parser.
    '''
    return logging.getLogger('TurboParser')


def logsumexp(a, axis=None):
    """Compute the log of the sum of exponentials of input elements.


    (Copied and simplified from scipy to avoid the dependency for just one
    function)
    """
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis)
        out = np.log(s)

    a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out


def read_embeddings(path, extra_symbols=None, max_words=1000000):
    '''
    Read a text file, or xzipped text file, with word embeddings.

    :param path: path to the embeddings file
    :param extra_symbols: extra symbols such as UNK to create embeddings for.
        They are placed in the beginning of the matrix and of the word list.
    :param max_words: maximum word embeddings to read (the rest will be ignored)
    :return: a word list and a numpy matrix
    '''
    if extra_symbols:
        # copy
        words = extra_symbols[:]
    else:
        words = []

    vectors = []
    logger = get_logger()

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
                error = 'Error reading line %d of embeddings file, ' \
                        'skipping' % line_number
                logger.error(error)
                continue

            words.append(word)
            vector = np.array([float(field) for field in fields[1:]])
            vectors.append(vector)

            if len(words) == max_words:
                break

    embeddings = np.array(vectors, dtype=np.float32)
    if extra_symbols is not None:
        shape = (len(extra_symbols), embeddings.shape[1])
        extra_embeddings = np.zeros(shape, dtype=embeddings.dtype)
        embeddings = np.concatenate([extra_embeddings, embeddings], 0)

    assert len(embeddings) == len(words)

    return words, embeddings
