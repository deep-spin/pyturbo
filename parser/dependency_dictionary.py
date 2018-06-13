# Abstract class for a dictionary. Task-specific dictionaries should derive
# from this class and implement the pure virtual methods.

from classifier.alphabet import Alphabet
import numpy as np

class DependencyDictionary(Dictionary):
    '''An abstract dictionary.'''
    def __init__(self, classifier=None):
        Dictionary.__init__(self)
        self.classifier = classifier
        self.token_dictionary = None
        self.label_alphabet = Alphabet()
        self.maximum_left_distances = None
        self.maximum_right_distances = None

    def save(self, file):
        raise NotImplementedError

    def load(self, file):
        raise NotImplementedError

    def allow_growth(self):
        raise NotImplementedError

    def stop_growth(self):
        raise NotImplementedError

    def create_label_dictionary(self, reader):
        raise NotImplementedError

    def get_label_name(self, label):
        return self.label_alphabet.get_label_name(label)
