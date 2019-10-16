from .alphabet import Alphabet
from ..classifier.utils import get_logger
from ..classifier.dictionary import Dictionary
from .dependency_reader import ConllReader
from .constants import UNKNOWN, SPECIAL_SYMBOLS, EMPTY, EOS, BOS
import pickle
from collections import Counter, OrderedDict


logger = get_logger()


class TokenDictionary(Dictionary):
    '''A dictionary for storing token information.'''
    def __init__(self):
        Dictionary.__init__(self)
        self.character_alphabet = Alphabet()
        self.pretrain_alphabet = Alphabet()
        self.form_alphabet = Alphabet()
        self.lemma_alphabet = Alphabet()
        self.morph_tag_alphabets = OrderedDict()
        self.morph_singleton_alphabet = Alphabet()
        self.upos_alphabet = Alphabet()
        self.xpos_alphabet = Alphabet()
        self.deprel_alphabet = Alphabet()

        # keep all alphabets ordered
        self.alphabets = [self.character_alphabet,
                          self.pretrain_alphabet,
                          self.form_alphabet,
                          self.lemma_alphabet,
                          self.morph_tag_alphabets,
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
        self.morph_tag_alphabets = pickle.load(file)
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

    def get_num_morph_features(self):
        return len(self.morph_tag_alphabets)

    def get_num_morph_values(self, i):
        """Return the number of values for the i-th morph attribute"""
        return len(self.morph_tag_alphabets[i])

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

    def get_morph_ids(self, morph_dict, special_symbol=EMPTY):
        ids = [None] * len(self.morph_tag_alphabets)
        for i, feature_name in enumerate(self.morph_tag_alphabets):
            alphabet = self.morph_tag_alphabets[feature_name]

            if feature_name in morph_dict:
                label = morph_dict[feature_name]
            else:
                # no available value for this attribute; e.g. tense in nouns
                label = special_symbol

            id_ = alphabet.lookup(label)
            if id_ < 0:
                id_ = alphabet.lookup(UNKNOWN)
            ids[i] = id_

        return ids

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
                   char_cutoff=1, form_cutoff=7, lemma_cutoff=7,
                   morph_cutoff=1):
        """
        Initializes the dictionary with indices for word forms and tags.

        If a word dictionary with words having pretrained embeddings is given,
        new words found in the training data are added in the beginning of it

        :param input_path: path to an input CoNLL file
        :param word_list: optional list with words in pre-trained embeddings
        """
        logger.info('Creating token dictionary...')

        char_counts = Counter()
        form_counts = Counter()
        lemma_counts = Counter()
        deprel_counts = Counter()
        upos_counts = Counter()
        xpos_counts = Counter()

        # this stores a Counter for each morph feature (tense, number, gender..)
        morph_counts = {}

        # embeddings not included here to keep the same ordering
        for alphabet in [self.upos_alphabet,
                         self.xpos_alphabet,
                         self.morph_singleton_alphabet,
                         self.deprel_alphabet]:
            for symbol in SPECIAL_SYMBOLS:
                alphabet.insert(symbol)

        # Go through the corpus and build the dictionaries,
        # counting the frequencies.
        reader = ConllReader()
        with reader.open(input_path) as r:
            for instance in r:
                # start from 1 to skip root
                for i in range(1, len(instance)):
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
                    deprel_counts[deprel] += 1

                    # POS tags
                    tag = instance.get_upos(i)
                    if tag != '_':
                        # tag _ is represented by EMPTY and added anyway
                        upos_counts[tag] += 1

                    tag = instance.get_xpos(i)
                    if tag != '_':
                        xpos_counts[tag] += 1

                    # Morph features
                    morph_singleton = instance.get_morph_singleton(i)
                    if morph_singleton != '_':
                        self.morph_singleton_alphabet.insert(morph_singleton)

                        # Add each key/value UFeats pair
                        morph_tags = instance.get_morph_tags(i)
                        for feature in morph_tags:
                            if feature not in morph_counts:
                                morph_counts[feature] = Counter()
                            value = morph_tags[feature]
                            morph_counts[feature][value] += 1

        # sort morph features to ensure deterministic behavior
        morph_features = sorted(morph_counts)
        for feature in morph_features:
            # feature_counts is a Counter mapping value to counts
            feature_counts = morph_counts[feature]
            total = sum(feature_counts.values())
            if total < morph_cutoff:
                continue

            feature_alphabet = Alphabet()
            for symbol in SPECIAL_SYMBOLS:
                feature_alphabet.insert(symbol)

            # now sort all values within this feature
            values = sorted(feature_counts)
            for value in values:
                count = feature_counts[value]
                if count >= morph_cutoff:
                    feature_alphabet.insert(value)

            self.morph_tag_alphabets[feature] = feature_alphabet

        # Now adjust the cutoffs if necessary.
        # (only using cutoffs for words and lemmas)
        for alphabet, counter, cutoff in \
            zip([self.character_alphabet, self.form_alphabet,
                 self.lemma_alphabet, self.deprel_alphabet,
                 self.upos_alphabet, self.xpos_alphabet],
                [char_counts, form_counts, lemma_counts, deprel_counts,
                 upos_counts, xpos_counts],
                [char_cutoff, form_cutoff, lemma_cutoff, 1, 1, 1]):

            alphabet.clear()
            for symbol in SPECIAL_SYMBOLS:
                alphabet.insert(symbol)

            if alphabet == self.character_alphabet:
                # EOS and BOS for characters only
                self.character_alphabet.insert(EOS)
                self.character_alphabet.insert(BOS)

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

        logger.debug('Number of characters: %d' % len(self.character_alphabet))
        logger.debug('Number of pretrained embedding forms: %d' %
                     len(self.pretrain_alphabet))
        logger.debug('Number of forms: %d' % len(self.form_alphabet))
        logger.debug('Number of lemmas: %d' % len(self.lemma_alphabet))
        logger.debug('Number of coarse POS tags: %d' % len(self.upos_alphabet))
        logger.debug('Number of fine POS tags: %d' %
                      len(self.xpos_alphabet))
        logger.debug('Number of morph singletons (combination of morph tags '
                     'seen in data): %d' % len(self.morph_singleton_alphabet))
        logger.debug('Number of dependency relations: %d' %
                     len(self.deprel_alphabet))
