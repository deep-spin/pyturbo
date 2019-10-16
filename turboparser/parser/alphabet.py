'''This implements a dictionary of labels.'''

from .constants import EMPTY


class Alphabet(dict):
    '''This class implements a dictionary of labels. Labels as mapped to
    integers, and it is efficient to retrieve the label name from its
    integer representation, and vice-versa.'''
    def __init__(self, label_names=None):
        dict.__init__(self)
        self.locked = False
        self.names = []
        if label_names is not None:
            for name in label_names:
                self.insert(name)

    def clear(self):
        dict.clear(self)
        self.names = []

    def insert(self, name):
        '''Add new label.'''
        if name in self:
            return
        if self.locked:
            raise ValueError('Attempted to insert in locked alphabet')
        else:
            label_id = len(self.names)
            self[name] = label_id
            self.names.append(name)

    def lookup(self, name):
        '''Lookup label.'''
        if name in self:
            return self[name]
        else:
            return -1

    def stop_growth(self):
        self.locked = True

    def allow_growth(self):
        self.locked = False

    def get_label_name(self, label_id):
        '''Get label name from id.'''
        if label_id == EMPTY:
            return '_'
        return self.names[label_id]
