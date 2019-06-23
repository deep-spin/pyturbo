'''This implements a dictionary of labels.'''

from collections import OrderedDict

from .constants import NONE, UNKNOWN


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
        return self.names[label_id]


class MultiAlphabet(object):
    """
    Class to store a multitype mapping, such as UFeats.

    It may contain several types of keys, such as tense, number, case, etc. Each
    key has its own set of possible values, such as past/present/future,
    plural/sing, etc.
    """
    def __init__(self):
        self.alphabets = OrderedDict()

    def allow_growth(self):
        for alphabet in self.alphabets:
            alphabet.allow_growth()

    def stop_growth(self):
        for alphabet in self.alphabets:
            alphabet.stop_growth()

    def insert(self, attributes):
        """
        Insert a dictionary of attributes, such that each key refers to an
        alphabet.
        """
        for key in attributes:
            if key not in self.alphabets:
                self.alphabets[key] = Alphabet()

            alphabet = self.alphabets[key]
            value = attributes[key]
            alphabet.insert(value)

    def sort(self):
        """
        Sort the order in which alphabets are stored. This is useful for
        consistency in multiple runs.
        """
        new_alphabets = OrderedDict()
        sorted_keys = sorted(self.alphabets)
        for key in sorted_keys:
            new_alphabets[key] = self.alphabets[key]

        self.alphabets = new_alphabets

    def lookup(self, feature_dict, special_symbol=NONE):
        """
        Lookup feature dictionary.

        :param feature_dict: dictionary mapping feature names to values
        :param special_symbol: symbol to use for missing features
        :return: a list of numeric ids
        """
        ids = [None] * len(self.alphabets)
        for i, feature_name in enumerate(self.alphabets):
            alphabet = self.alphabets[feature_name]
            if feature_name in feature_dict:
                label = feature_dict[feature_name]
            else:
                # no available value for this attribute; e.g. tense in nouns
                label = special_symbol
            id_ = alphabet.lookup(label)
            if id_ < 0:
                id_ = alphabet.lookup(UNKNOWN)
            ids[i] = id_

        return ids

    def get_label_names(self, label_ids):
        """
        Return a dictionary mapping keys to values.

        :param label_ids: a list in the same order as the alphabets used by
            this object
        :return: dictionary mapping key names to value names
        """
        name_dict = {}
        for i, alphabet_name in enumerate(self.alphabets):
            id_ = label_ids[i]
            alphabet = self.alphabets[alphabet_name]
            label_name = alphabet.get_label_name(id_)
            name_dict[alphabet_name] = label_name

        return name_dict
