class Reader(object):
    '''An abstract reader.'''
    def __init__(self):
        pass

    def open(self, path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError
