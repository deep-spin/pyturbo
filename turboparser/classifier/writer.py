class Writer(object):
    '''An abstract writer.'''
    def __init__(self):
        pass

    def open(self, path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def write(self, instance):
        raise NotImplementedError
