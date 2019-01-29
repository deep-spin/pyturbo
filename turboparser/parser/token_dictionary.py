from ..classifier.alphabet import Alphabet
from ..classifier.dictionary import Dictionary
import pickle
import logging
from collections import Counter

UNKNOWN = '_UNKNOWN_'
START = '<s>'
STOP = '</s>'
PADDING = '_PADDING_'


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
        self.upos_alphabet = Alphabet()
        self.xpos_alphabet = Alphabet()
        self.shape_alphabet = Alphabet()

        # keep all alphabets ordered
        self.alphabets = [self.character_alphabet,
                          self.embedding_alphabet,
                          self.form_alphabet,
                          self.form_lower_alphabet,
                          self.lemma_alphabet,
                          self.prefix_alphabet,
                          self.suffix_alphabet,
                          self.morph_tag_alphabet,
                          self.upos_alphabet,
                          self.xpos_alphabet,
                          self.shape_alphabet]

        # Special symbols.
        self.special_symbols = Alphabet()
        self.token_unknown = self.special_symbols.insert(UNKNOWN)
        self.token_start = self.special_symbols.insert(START)
        self.token_stop = self.special_symbols.insert(STOP)
        self.token_padding = self.special_symbols.insert(PADDING)

        # Maximum alphabet sizes.
        self.max_forms = 0xffff
        self.max_lemmas = 0xffff
        self.max_shapes = 0xffff
        self.max_tags = 0xff
        self.max_morph_tags = 0xfff

    def add_special_symbol(self, symbol):
        """
        Add special symbols to the dictionary.

        Any calls to this method must be made before `initialize`.
        """
        self.special_symbols.insert(symbol)

    def save(self, file):
        for alphabet in self.alphabets:
            pickle.dump(alphabet, file)
        pickle.dump(self.special_symbols, file)
        pickle.dump(self.token_unknown, file)
        pickle.dump(self.token_start, file)
        pickle.dump(self.token_stop, file)
        pickle.dump(self.max_forms, file)
        pickle.dump(self.max_lemmas, file)
        pickle.dump(self.max_shapes, file)
        pickle.dump(self.max_tags, file)
        pickle.dump(self.max_morph_tags, file)

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
        self.upos_alphabet = pickle.load(file)
        self.xpos_alphabet = pickle.load(file)
        self.shape_alphabet = pickle.load(file)
        self.special_symbols = pickle.load(file)
        self.token_unknown = pickle.load(file)
        self.token_start = pickle.load(file)
        self.token_stop = pickle.load(file)
        self.max_forms = pickle.load(file)
        self.max_lemmas = pickle.load(file)
        self.max_shapes = pickle.load(file)
        self.max_tags = pickle.load(file)
        self.max_morph_tags = pickle.load(file)

    def allow_growth(self):
        for alphabet in self.alphabets:
            alphabet.allow_growth()

    def stop_growth(self):
        for alphabet in self.alphabets:
            alphabet.stop_growth()

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

    def get_shape_id(self, shape):
        id_ = self.shape_alphabet.lookup(shape)
        if id_ >= 0:
            return id_
        return self.shape_alphabet.lookup(UNKNOWN)

    def initialize(self, reader, case_sensitive, word_dict=None):
        """
        Initializes the dictionary with indices for word forms and tags.

        If a word dictionary with words having pretrained embeddings is given,
        new words found in the training data are added in the beginning of it

        :param reader: a subclass of Reader
        :param word_dict: optional dictionary mapping words to indices, in case
            pre-trained embeddings are used.
        """
        logging.info('Creating token dictionary...')

        char_counts = Counter()
        form_counts = Counter()
        form_lower_counts = Counter()
        lemma_counts = Counter()
        upos_counts = Counter()
        xpos_counts = Counter()
        morph_tag_counts = Counter()

        for name in self.special_symbols.names:
            # embeddings not included here to keep the same ordering
            for alphabet in [self.form_alphabet,
                             self.form_lower_alphabet,
                             self.lemma_alphabet,
                             self.prefix_alphabet,
                             self.suffix_alphabet,
                             self.upos_alphabet,
                             self.xpos_alphabet,
                             self.morph_tag_alphabet]:
                alphabet.insert(name)

        # Go through the corpus and build the dictionaries,
        # counting the frequencies.
        with reader.open(self.classifier.options.training_path) as r:
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

                    # Add prefix/suffix to alphabet.
                    # TODO: add varying lengths.
                    prefix = form[:self.classifier.options.prefix_length]
                    suffix = form[-self.classifier.options.suffix_length:]

                    # Add tags to alphabet.
                    tag = instance.get_upos(i)
                    upos_counts[tag] += 1

                    tag = instance.get_xpos(i)
                    xpos_counts[tag] += 1

                    # Add morph tags to alphabet.
                    for j in range(instance.get_num_morph_tags(i)):
                        morph_tag = instance.get_morph_tag(i, j)
                        morph_tag_counts[morph_tag] += 1

        # Now adjust the cutoffs if necessary.
        for label, alphabet, counter, cutoff, max_length in \
            zip(['char', 'form', 'form_lower', 'lemma', 'tag', 'morph_tag'],
                [self.character_alphabet, self.form_alphabet,
                 self.form_lower_alphabet, self.lemma_alphabet,
                 self.upos_alphabet, self.xpos_alphabet,
                 self.morph_tag_alphabet],
                [char_counts, form_counts, form_lower_counts, lemma_counts,
                 upos_counts, xpos_counts, morph_tag_counts],
                [self.classifier.options.char_cutoff,
                 self.classifier.options.form_cutoff,
                 self.classifier.options.form_cutoff,
                 self.classifier.options.lemma_cutoff,
                 self.classifier.options.tag_cutoff,
                 self.classifier.options.tag_cutoff,
                 self.classifier.options.morph_tag_cutoff],
                [int(10e6), self.max_forms, self.max_forms, self.max_lemmas,
                 self.max_tags, self.max_tags, self.max_morph_tags]):

            alphabet.clear()
            for name in self.special_symbols.names:
                alphabet.insert(name)
            max_length = max_length - len(self.special_symbols)
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
        new_words = dataset_words - embedding_words
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
