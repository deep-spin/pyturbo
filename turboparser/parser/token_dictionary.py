from ..classifier.alphabet import Alphabet
from ..classifier.dictionary import Dictionary
from .dependency_reader import ConllReader
import pickle
import logging
from collections import Counter

UNKNOWN = '_UNKNOWN_'


class TokenDictionary(Dictionary):
    '''A dictionary for storing token information.'''
    def __init__(self, classifier=None):
        Dictionary.__init__(self)
        self.classifier = classifier
        self.character_alphabet = Alphabet()
        self.embedding_alphabet = Alphabet()
        self.form_alphabet = Alphabet()
        self.form_lower_alphabet = Alphabet()
        self.lemma_alphabet = Alphabet()
        self.prefix_alphabet = Alphabet()
        self.suffix_alphabet = Alphabet()
        self.morph_tag_alphabet = Alphabet()
        self.morph_singleton_alphabet = Alphabet()
        self.upos_alphabet = Alphabet()
        self.xpos_alphabet = Alphabet()
        self.shape_alphabet = Alphabet()
        self.deprel_alphabet = Alphabet()

        self.max_forms = 0xffff

        # keep all alphabets ordered
        self.alphabets = [self.character_alphabet,
                          self.embedding_alphabet,
                          self.form_alphabet,
                          self.form_lower_alphabet,
                          self.lemma_alphabet,
                          self.prefix_alphabet,
                          self.suffix_alphabet,
                          self.morph_tag_alphabet,
                          self.morph_singleton_alphabet,
                          self.upos_alphabet,
                          self.xpos_alphabet,
                          self.shape_alphabet,
                          self.deprel_alphabet]

    def save(self, file):
        for alphabet in self.alphabets:
            pickle.dump(alphabet, file)

    def load(self, file):
        # TODO: avoid repeating code somehow
        self.character_alphabet = pickle.load(file)
        self.embedding_alphabet = pickle.load(file)
        self.form_alphabet = pickle.load(file)
        self.form_lower_alphabet = pickle.load(file)
        self.lemma_alphabet = pickle.load(file)
        self.prefix_alphabet = pickle.load(file)
        self.suffix_alphabet = pickle.load(file)
        self.morph_tag_alphabet = pickle.load(file)
        self.morph_singleton_alphabet = pickle.load(file)
        self.upos_alphabet = pickle.load(file)
        self.xpos_alphabet = pickle.load(file)
        self.shape_alphabet = pickle.load(file)
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
        return len(self.embedding_alphabet)

    def get_num_forms(self):
        return len(self.form_alphabet)

    def get_num_lemmas(self):
        return len(self.form_alphabet)

    def get_num_upos_tags(self):
        return len(self.upos_alphabet)

    def get_num_xpos_tags(self):
        return len(self.xpos_alphabet)

    def get_num_morph_tags(self):
        return len(self.morph_tag_alphabet)

    def get_num_morph_singletons(self):
        return len(self.morph_singleton_alphabet)

    def get_num_deprels(self):
        return len(self.deprel_alphabet)

    def get_embedding_id(self, form):
        id_ = self.embedding_alphabet.lookup(form)
        if id_ >= 0:
            return id_
        return self.embedding_alphabet.lookup(UNKNOWN)

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

    def get_form_lower_id(self, form_lower):
        id_ = self.form_lower_alphabet.lookup(form_lower)
        if id_ >= 0:
            return id_
        return self.form_lower_alphabet.lookup(UNKNOWN)

    def get_lemma_id(self, lemma):
        id_ = self.lemma_alphabet.lookup(lemma)
        if id_ >= 0:
            return id_
        return self.lemma_alphabet.lookup(UNKNOWN)

    def get_prefix_id(self, prefix):
        id_ = self.prefix_alphabet.lookup(prefix)
        if id_ >= 0:
            return id_
        return self.prefix_alphabet.lookup(UNKNOWN)

    def get_suffix_id(self, suffix):
        id_ = self.suffix_alphabet.lookup(suffix)
        if id_ >= 0:
            return id_
        return self.suffix_alphabet.lookup(UNKNOWN)

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

    def get_shape_id(self, shape):
        id_ = self.shape_alphabet.lookup(shape)
        if id_ >= 0:
            return id_
        return self.shape_alphabet.lookup(UNKNOWN)

    def get_deprel_id(self, deprel):
        id_ = self.deprel_alphabet.lookup(deprel)
        if id_ >= 0:
            return id_
        return self.deprel_alphabet.lookup(UNKNOWN)

    def initialize(self, input_path, case_sensitive, word_dict=None):
        """
        Initializes the dictionary with indices for word forms and tags.

        If a word dictionary with words having pretrained embeddings is given,
        new words found in the training data are added in the beginning of it

        :param input_path: path to an input CoNLL file
        :param word_dict: optional dictionary mapping words to indices, in case
            pre-trained embeddings are used.
        """
        logging.info('Creating token dictionary...')

        char_counts = Counter()
        form_counts = Counter()
        form_lower_counts = Counter()
        lemma_counts = Counter()

        # embeddings not included here to keep the same ordering
        for alphabet in [self.form_alphabet,
                         self.form_lower_alphabet,
                         self.lemma_alphabet,
                         self.character_alphabet,
                         self.upos_alphabet,
                         self.xpos_alphabet,
                         self.morph_tag_alphabet,
                         self.morph_singleton_alphabet,
                         self.deprel_alphabet]:
            alphabet.insert(UNKNOWN)

        # Go through the corpus and build the dictionaries,
        # counting the frequencies.
        reader = ConllReader()
        with reader.open(input_path) as r:
            for instance in r:
                for i in range(len(instance)):
                    # Add form to alphabet.
                    form = instance.get_form(i)
                    form_lower = form.lower()
                    if not case_sensitive:
                        form = form_lower
                    form_counts[form] += 1

                    for char in form:
                        char_counts[char] += 1

                    # Add lower-case form to alphabet.
                    form_lower_counts[form_lower] += 1

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
                    for j in range(instance.get_num_morph_tags(i)):
                        morph_tag = instance.get_morph_tag(i, j)
                        self.morph_tag_alphabet.insert(morph_tag)

        # Now adjust the cutoffs if necessary.
        # (only using cutoffs for words and chars)
        for label, alphabet, counter, cutoff, max_length in \
            zip(['char', 'form', 'form_lower', 'lemma'],
                [self.character_alphabet, self.form_alphabet,
                 self.form_lower_alphabet, self.lemma_alphabet],
                [char_counts, form_counts, form_lower_counts, lemma_counts],
                [self.classifier.options.char_cutoff,
                 self.classifier.options.form_cutoff,
                 self.classifier.options.form_cutoff,
                 self.classifier.options.lemma_cutoff],
                [int(10e6), self.max_forms, self.max_forms, self.max_forms]):

            alphabet.clear()
            alphabet.insert(UNKNOWN)
            max_length -= 1  # -1 for the unknown symbol
            for name, count in counter.most_common(max_length):
                if count >= cutoff:
                    alphabet.insert(name)
            alphabet.stop_growth()

        if word_dict is not None:
            word_list = sorted(word_dict, key=lambda w: word_dict[w])
            for word in word_list:
                self.embedding_alphabet.insert(word)

        # update the embedding vocabulary with new words found in training data
        dataset_words = self.form_alphabet.keys()
        embedding_words = self.embedding_alphabet.keys()

        # get a deterministic ordering
        new_words = sorted(dataset_words - embedding_words)
        for new_word in new_words:
            self.embedding_alphabet.insert(new_word)
        self.embedding_alphabet.stop_growth()

        logging.info('Number of characters: %d' % len(self.character_alphabet))
        logging.info('Number of embedding forms: %d' %
                     len(self.embedding_alphabet))
        logging.info('Number of forms: %d' % len(self.form_alphabet))
        logging.info('Number of lower-case forms: %d' %
                     len(self.form_lower_alphabet))
        logging.info('Number of lemmas: %d' % len(self.lemma_alphabet))
        logging.info('Number of prefixes: %d' % len(self.prefix_alphabet))
        logging.info('Number of suffixes: %d' % len(self.suffix_alphabet))
        logging.info('Number of coarse POS tags: %d' % len(self.upos_alphabet))
        logging.info('Number of fine POS tags: %d' %
                     len(self.xpos_alphabet))
        logging.info('Number of morph tags: %d' % len(self.morph_tag_alphabet))
        logging.info('Number of morph singletons (combination of morph tags '
                     'seen in data): %d' % len(self.morph_singleton_alphabet))
        logging.info('Number of dependency relations: %d' %
                     len(self.deprel_alphabet))
