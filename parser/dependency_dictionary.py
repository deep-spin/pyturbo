from classifier.alphabet import Alphabet
from classifier.dictionary import Dictionary
import numpy as np
import logging

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
        logging.info('Creating relation dictionary...')

        relation_counts = []

        # Go through the corpus and build the relation dictionary,
        # counting the frequencies.
        reader.open(self.classifier.options.training_path)
        instance = reader.next()
        while instance is not None:
            for i in range(1, len(instance)):
                # Add dependency relaion to alphabet.
                relation = instance.get_relation(i)
                id = self.relation_alphabet.insert(relation)
                if id >= len(relation_counts):
                    relation_counts.append(1)
                else:
                    relation_counts[id] += 1
            instance = reader.next()

        reader.close()
        self.relation_alphabet.stop_growth()

        # Go through the corpus and build the existing relation for each
        # head-modifier POS pair.
        num_tags = self.classifier.token_dictionary.get_num_tags()
        self.existing_relations = [[[] for i in range(num_tags)]
                                   for j in range(num_tags)]
        self.maximum_left_distances = [[0 for i in range(num_tags)]
                                       for j in range(num_tags)]
        self.maximum_right_distances = [[0 for i in range(num_tags)]
                                        for j in range(num_tags)]

        reader.open(self.classifier.options.training_path)
        instance = reader.next()
        while instance is not None:
            for i in range(1, len(instance)):
                head = instance.get_head(i)
                assert 0 <= head < len(instance)
                modifier_tag = self.classifier.token_dictionary.get_tag_id(
                    instance.get_tag(i))
                head_tag = self.classifier.token_dictionary.get_tag_id(
                    instance.get_tag(head))
                if modifier_tag < 0:
                    modifier_tag = \
                        self.classifier.token_dictionary.token_unknown
                if head_tag < 0:
                    head_tag = self.classifier.token_dictionary.token_unknown

                id = self.relation_alphabet.lookup(instance.get_relation(i))
                assert id >= 0

                # Insert new relation in the set of existing relations, if it
                # is not there already. NOTE: this is inefficient, maybe we
                # should be using a different data structure.
                if id not in self.existing_relations[modifier_tag][head_tag]:
                    self.existing_relations[modifier_tag][head_tag].append(id)

                # Update the maximum distances if necessary.
                if head:
                    if head < i:
                        # Right attachment.
                        if i - head > \
                           self.maximum_right_distances[modifier_tag][head_tag]:
                            self.maximum_right_distances \
                                [modifier_tag][head_tag] = i - head
                    else:
                        # Left attachment.
                        if head - i > \
                           self.maximum_left_distances[modifier_tag][head_tag]:
                            self.maximum_left_distances \
                                [modifier_tag][head_tag] = head - i

            instance = reader.next()

        reader.close()
        logging.info('Number of relations: %d' % len(self.relation_alphabet))

    def get_relation_name(self, relation):
        return self.relation_alphabet.get_relation_name(relation)

    def get_existing_relations(self, modifier_tag, head_tag):
        return self.existing_relations[modifier_tag][head_tag]

    def get_maximum_left_distance(self, modifier_tag, head_tag):
        return self.maximum_left_distances[modifier_tag][head_tag]

    def get_maximum_right_distance(self, modifier_tag, head_tag):
        return self.maximum_right_distances[modifier_tag][head_tag]
