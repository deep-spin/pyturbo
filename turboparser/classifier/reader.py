class Reader(object):
    '''An abstract reader.'''
    def __init__(self, path):
        self.path = path
        self.file = None

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()
