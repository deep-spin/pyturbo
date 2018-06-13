# Abstract class for a dictionary. Task-specific dictionaries should derive
# from this class and implement the pure virtual methods.

class Dictionary(object):
    '''An abstract dictionary.'''
    def __init__(self):
        pass

    def clear(self):
        raise NotImplementedError

    def save(self, file):
        raise NotImplementedError

    def load(self, file):
        raise NotImplementedError

    def allow_growth(self):
        raise NotImplementedError

    def stop_growth(self):
        raise NotImplementedError

