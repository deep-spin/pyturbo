from .alphabet import Alphabet, MultiAlphabet
from ..classifier.dictionary import Dictionary
from .dependency_reader import ConllReader
from .constants import ROOT, UNKNOWN, NONE
import pickle
import logging
from collections import Counter


class TokenDictionary(Dictionary):
    '''A dictionary for storing token information.'''
    def __init__(self):
        Dictionary.__init__(self)
        self.character_alphabet = Alphabet()
        self.pretrain_alphabet = Alphabet()
        self.form_alphabet = Alphabet()
        self.lemma_alphabet = Alphabet()
        self.morph_tag_alphabet = MultiAlphabet()
        self.morph_singleton_alphabet = Alphabet()
        self.upos_alphabet = Alphabet()
        self.xpos_alphabet = Alphabet()
        self.deprel_alphabet = Alphabet()

        # keep all alphabets ordered
        self.alphabets = [self.character_alphabet,
                          self.pretrain_alphabet,
                          self.form_alphabet,
                          self.lemma_alphabet,
                          self.morph_tag_alphabet,
                          self.morph_singleton_alphabet,
                          self.upos_alphabet,
                          self.xpos_alphabet,
                          self.deprel_alphabet]

    def save(self, file):
        for alphabet in self.alphabets:
            pickle.dump(alphabet, file)

    def load(self, file):
        # TODO: avoid repeating code somehow
        self.character_alphabet = pickle.load(file)
        self.pretrain_alphabet = pickle.load(file)
        self.form_alphabet = pickle.load(file)
        self.lemma_alphabet = pickle.load(file)
        self.morph_tag_alphabet = pickle.load(file)
        self.morph_singleton_alphabet = pickle.load(file)
        self.upos_alphabet = pickle.load(file)
        self.xpos_alphabet = pickle.load(file)
        self.deprel_alphabet = pickle.load(file)

    def allow_growth(self):
        for alphabet in self.alphabets:
            alphabet.allow_growth()

    def stop_growth(self):
        for alphabet in self.alphabets:
            alphabet.stop_growth()

    def get_upos_tags(self):
        """Return the set of UPOS tags"""
        return self.upos_alphabet.keys()

    def get_xpos_tags(self):
        """Return the set of XPOS tags"""
        return self.xpos_alphabet.keys()

    def get_deprel_tags(self):
        return self.deprel_alphabet.keys()

    def get_num_characters(self):
        return len(self.character_alphabet)

    def get_num_embeddings(self):
        return len(self.pretrain_alphabet)

    def get_num_forms(self):
        return len(self.form_alphabet)

    def get_num_lemmas(self):
        return len(self.lemma_alphabet)

    def get_num_upos_tags(self):
        return len(self.upos_alphabet)

    def get_num_xpos_tags(self):
        return len(self.xpos_alphabet)

    def get_num_morph_attributes(self):
        return len(self.morph_tag_alphabet.alphabets)

    def get_num_morph_values(self, i):
        """Return the number of values for the i-th morph attribute"""
        return len(self.morph_tag_alphabet.alphabets[i])

    def get_num_morph_singletons(self):
        return len(self.morph_singleton_alphabet)

    def get_num_deprels(self):
        return len(self.deprel_alphabet)

    def get_embedding_id(self, form):
        id_ = self.pretrain_alphabet.lookup(form)
        if id_ >= 0:
            return id_
        return self.pretrain_alphabet.lookup(UNKNOWN)

    def get_character_id(self, character):
        id_ = self.character_alphabet.lookup(character)
        if id_ >= 0:
            return id_
        return self.character_alphabet.lookup(UNKNOWN)

    def get_form_id(self, form):
        id_ = self.form_alphabet.lookup(form)
        if id_ >= 0:
            return id_
        return self.form_alphabet.lookup(UNKNOWN)

    def get_lemma_id(self, lemma):
        id_ = self.lemma_alphabet.lookup(lemma)
        if id_ >= 0:
            return id_
        return self.lemma_alphabet.lookup(UNKNOWN)

    def get_morph_ids(self, morph_dict):
        return self.morph_tag_alphabet.lookup(morph_dict)

    def get_upos_id(self, tag):
        id_ = self.upos_alphabet.lookup(tag)
        if id_ >= 0:
            return id_
        return self.upos_alphabet.lookup(UNKNOWN)

    def get_xpos_id(self, tag):
        id_ = self.xpos_alphabet.lookup(tag)
        if id_ >= 0:
            return id_
        return self.xpos_alphabet.lookup(UNKNOWN)

    def get_morph_tag_id(self, morph_tag):
        id_ = self.morph_tag_alphabet.lookup(morph_tag)
        if id_ >= 0:
            return id_
        return self.morph_tag_alphabet.lookup(UNKNOWN)

    def get_morph_singleton_id(self, morph_singleton):
        id_ = self.morph_singleton_alphabet.lookup(morph_singleton)
        if id_ >= 0:
            return id_
        return self.morph_singleton_alphabet.lookup(UNKNOWN)

    def get_deprel_id(self, deprel):
        id_ = self.deprel_alphabet.lookup(deprel)
        if id_ >= 0:
            return id_
        return self.deprel_alphabet.lookup(UNKNOWN)

    def initialize(self, input_path, case_sensitive, word_list=None,
                   char_cutoff=1, form_cutoff=7, lemma_cutoff=7):
        """
        Initializes the dictionary with indices for word forms and tags.

        If a word dictionary with words having pretrained embeddings is given,
        new words found in the training data are added in the beginning of it

        :param input_path: path to an input CoNLL file
        :param word_list: optional list with words in pre-trained embeddings
        """
        logging.info('Creating token dictionary...')

        char_counts = Counter()
        form_counts = Counter()
        lemma_counts = Counter()

        # embeddings not included here to keep the same ordering
        for alphabet in [self.upos_alphabet,
                         self.xpos_alphabet,
                         self.morph_singleton_alphabet,
                         self.deprel_alphabet]:
            alphabet.insert(UNKNOWN)
            if alphabet is not self.deprel_alphabet:
                alphabet.insert(ROOT)

        # Go through the corpus and build the dictionaries,
        # counting the frequencies.
        reader = ConllReader()
        with reader.open(input_path) as r:
            for instance in r:
                for i in range(len(instance)):
                    # Add form to alphabet.
                    form = instance.get_form(i)
                    if not case_sensitive:
                        form_counts[form.lower()] += 1
                    else:
                        form_counts[form] += 1

                    for char in form:
                        char_counts[char] += 1

                    # Add lemma to alphabet.
                    lemma = instance.get_lemma(i)
                    lemma_counts[lemma] += 1

                    # Dependency relation
                    deprel = instance.get_relation(i)
                    self.deprel_alphabet.insert(deprel)

                    # Add tags to alphabet.
                    tag = instance.get_upos(i)
                    self.upos_alphabet.insert(tag)

                    tag = instance.get_xpos(i)
                    self.xpos_alphabet.insert(tag)

                    # Add morph tags to alphabet.
                    morph_singleton = instance.get_morph_singleton(i)
                    self.morph_singleton_alphabet.insert(morph_singleton)

                    # Add each key/value UFeats pair
                    morph_tags = instance.get_morph_tags(i)
                    self.morph_tag_alphabet.insert(morph_tags)

        self.morph_tag_alphabet.sort()
        for sub_alphabet_name in self.morph_tag_alphabet.alphabets:
            sub_alphabet = self.morph_tag_alphabet.alphabets[sub_alphabet_name]
            sub_alphabet.insert(UNKNOWN)
            sub_alphabet.insert(ROOT)
            sub_alphabet.insert(NONE)

        # Now adjust the cutoffs if necessary.
        # (only using cutoffs for words and lemmas)
        for alphabet, counter, cutoff in \
            zip([self.character_alphabet, self.form_alphabet,
                 self.lemma_alphabet],
                [char_counts, form_counts, lemma_counts],
                [char_cutoff, form_cutoff, lemma_cutoff]):

            alphabet.clear()
            alphabet.insert(UNKNOWN)
            if alphabet != 'char':
                alphabet.insert(ROOT)

            for name, count in counter.most_common():
                if count >= cutoff:
                    alphabet.insert(name)
                else:
                    break

            alphabet.stop_growth()

        if word_list is not None:
            for word in word_list:
                self.pretrain_alphabet.insert(word)

        # # update the embedding vocabulary with new words found in training data
        # dataset_words = self.form_alphabet.keys()
        # embedding_words = self.embedding_alphabet.keys()

        # # get a deterministic ordering
        # new_words = sorted(dataset_words - embedding_words)
        # for new_word in new_words:
        #     self.embedding_alphabet.insert(new_word)
        self.pretrain_alphabet.stop_growth()

        logging.info('Number of characters: %d' % len(self.character_alphabet))
        logging.info('Number of pretrained embedding forms: %d' %
                     len(self.pretrain_alphabet))
        logging.info('Number of forms: %d' % len(self.form_alphabet))
        logging.info('Number of lemmas: %d' % len(self.lemma_alphabet))
        logging.info('Number of coarse POS tags: %d' % len(self.upos_alphabet))
        logging.info('Number of fine POS tags: %d' %
                     len(self.xpos_alphabet))
        logging.info('Number of morph singletons (combination of morph tags '
                     'seen in data): %d' % len(self.morph_singleton_alphabet))
        logging.info('Number of dependency relations: %d' %
                     len(self.deprel_alphabet))
