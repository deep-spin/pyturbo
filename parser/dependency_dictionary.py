from classifier.alphabet import Alphabet
from classifier.dictionary import Dictionary
import numpy as np

class DependencyDictionary(Dictionary):
    '''An abstract dictionary.'''
    def __init__(self, classifier=None):
        Dictionary.__init__(self)
        self.classifier = classifier
        self.relation_alphabet = Alphabet()
        self.existing_relations = None
        self.maximum_left_distances = None
        self.maximum_right_distances = None

    def save(self, file):
        raise NotImplementedError

    def load(self, file):
        raise NotImplementedError

    def allow_growth(self):
        self.classifier.token_dictionary.allow_growth()

    def stop_growth(self):
        self.classifier.token_dictionary.stop_growth()

    def create_relation_dictionary(self, reader):
        raise NotImplementedError

    def get_relation_name(self, relation):
        return self.relation_alphabet.get_relation_name(relation)

    def get_existing_relations(self, modifier_tag, head_tag):
        return self.existing_relations[modifier_tag][head_tag]

    def get_maximum_left_distance(self, modifier_tag, head_tag):
        return self.maximum_left_distances[modifier_tag][head_tag]

    def get_maximum_right_distance(self, modifier_tag, head_tag):
        return self.maximum_right_distances[modifier_tag][head_tag]
