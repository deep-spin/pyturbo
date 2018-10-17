from ..classifier.alphabet import Alphabet
from ..classifier.dictionary import Dictionary
import pickle
import logging

class TokenDictionary(Dictionary):
    '''A dictionary for storing token information.'''
    def __init__(self, classifier=None):
        Dictionary.__init__(self)
        self.classifier = classifier
        self.form_alphabet = Alphabet()
        self.form_lower_alphabet = Alphabet()
        self.lemma_alphabet = Alphabet()
        self.prefix_alphabet = Alphabet()
        self.suffix_alphabet = Alphabet()
        self.morph_tag_alphabet = Alphabet()
        self.tag_alphabet = Alphabet()
        self.shape_alphabet = Alphabet()

        # Special symbols.
        self.special_symbols = Alphabet()
        self.token_unknown = self.special_symbols.insert('_UNKNOWN_')
        self.token_start = self.special_symbols.insert('_START_')
        self.token_stop = self.special_symbols.insert('_STOP_')
        self.token_padding = self.special_symbols.insert('_PADDING_')

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
        pickle.dump(self.form_alphabet, file)
        pickle.dump(self.form_lower_alphabet, file)
        pickle.dump(self.lemma_alphabet, file)
        pickle.dump(self.prefix_alphabet, file)
        pickle.dump(self.suffix_alphabet, file)
        pickle.dump(self.morph_tag_alphabet, file)
        pickle.dump(self.tag_alphabet, file)
        pickle.dump(self.shape_alphabet, file)
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
        self.form_alphabet = pickle.load(file)
        self.form_lower_alphabet = pickle.load(file)
        self.lemma_alphabet = pickle.load(file)
        self.prefix_alphabet = pickle.load(file)
        self.suffix_alphabet = pickle.load(file)
        self.morph_tag_alphabet = pickle.load(file)
        self.tag_alphabet = pickle.load(file)
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
        self.form_alphabet.allow_growth()
        self.form_lower_alphabet.allow_growth()
        self.lemma_alphabet.allow_growth()
        self.prefix_alphabet.allow_growth()
        self.suffix_alphabet.allow_growth()
        self.morph_tag_alphabet.allow_growth()
        self.tag_alphabet.allow_growth()
        self.shape_alphabet.allow_growth()

    def stop_growth(self):
        self.form_alphabet.stop_growth()
        self.form_lower_alphabet.stop_growth()
        self.lemma_alphabet.stop_growth()
        self.prefix_alphabet.stop_growth()
        self.suffix_alphabet.stop_growth()
        self.morph_tag_alphabet.stop_growth()
        self.tag_alphabet.stop_growth()
        self.shape_alphabet.stop_growth()

    def get_num_forms(self):
        return len(self.form_alphabet)

    def get_num_lemmas(self):
        return len(self.form_alphabet)

    def get_num_tags(self):
        return len(self.tag_alphabet)

    def get_form_id(self, form):
        return self.form_alphabet.lookup(form)

    def get_form_lower_id(self, form_lower):
        return self.form_lower_alphabet.lookup(form_lower)

    def get_lemma_id(self, lemma):
        return self.lemma_alphabet.lookup(lemma)

    def get_prefix_id(self, prefix):
        return self.prefix_alphabet.lookup(prefix)

    def get_suffix_id(self, suffix):
        return self.suffix_alphabet.lookup(suffix)

    def get_tag_id(self, tag):
        return self.tag_alphabet.lookup(tag)

    def get_morph_tag_id(self, morph_tag):
        return self.morph_tag_alphabet.lookup(morph_tag)

    def get_shape_id(self, shape):
        return self.shape_alphabet.lookup(shape)

    def initialize(self, reader):
        """
        Initializes the dictionary with indices for word forms and tags.
        """
        logging.info('Creating token dictionary...')

        form_counts = []
        form_lower_counts = []
        lemma_counts = []
        tag_counts = []
        morph_tag_counts = []

        for name in self.special_symbols.names:
            for alphabet in [self.form_alphabet,
                             self.form_lower_alphabet,
                             self.lemma_alphabet,
                             self.prefix_alphabet,
                             self.suffix_alphabet,
                             self.tag_alphabet,
                             self.morph_tag_alphabet]:
                alphabet.insert(name)
            for counts in [form_counts,
                           form_lower_counts,
                           lemma_counts,
                           tag_counts,
                           morph_tag_counts]:
                counts.append(-1)

        # Go through the corpus and build the dictionaries,
        # counting the frequencies.
        with reader.open(self.classifier.options.training_path) as r:
            for instance in r:
                for i in range(len(instance)):
                    # Add form to alphabet.
                    form = instance.get_form(i)
                    form_lower = form.lower()
                    if not self.classifier.options.form_case_sensitive:
                        form = form_lower
                    id = self.form_alphabet.insert(form)
                    if id >= len(form_counts):
                        form_counts.append(1)
                    else:
                        form_counts[id] += 1

                    # Add lower-case form to alphabet.
                    id = self.form_lower_alphabet.insert(form_lower)
                    if id >= len(form_lower_counts):
                        form_lower_counts.append(1)
                    else:
                        form_lower_counts[id] += 1

                    # Add lemma to alphabet.
                    id = self.lemma_alphabet.insert(instance.get_lemma(i))
                    if id >= len(lemma_counts):
                        lemma_counts.append(1)
                    else:
                        lemma_counts[id] += 1

                    # Add prefix/suffix to alphabet.
                    # TODO: add varying lengths.
                    prefix = form[:self.classifier.options.prefix_length]
                    id = self.prefix_alphabet.insert(prefix)
                    suffix = form[-self.classifier.options.suffix_length:]
                    id = self.suffix_alphabet.insert(suffix)

                    # Add tags to alphabet.
                    id = self.tag_alphabet.insert(instance.get_tag(i))
                    if id >= len(tag_counts):
                        tag_counts.append(1)
                    else:
                        tag_counts[id] += 1

                    # Add morph tags to alphabet.
                    for j in range(instance.get_num_morph_tags(i)):
                        id = self.morph_tag_alphabet.insert(
                            instance.get_morph_tag(i, j))
                        if id >= len(morph_tag_counts):
                            morph_tag_counts.append(1)
                        else:
                            morph_tag_counts[id] += 1

        # Now adjust the cutoffs if necessary.
        for label, alphabet, counts, cutoff, max_length in \
            zip(['form', 'form_lower', 'lemma', 'tag', 'morph_tag'],
                [self.form_alphabet, self.form_lower_alphabet,
                 self.lemma_alphabet, self.tag_alphabet,
                 self.morph_tag_alphabet],
                [form_counts, form_lower_counts, lemma_counts, tag_counts,
                 morph_tag_counts],
                [self.classifier.options.form_cutoff,
                 self.classifier.options.form_cutoff,
                 self.classifier.options.lemma_cutoff,
                 self.classifier.options.tag_cutoff,
                 self.classifier.options.morph_tag_cutoff],
                [self.max_forms, self.max_forms, self.max_lemmas,
                 self.max_tags, self.max_morph_tags]):
            names = alphabet.names.copy()
            while True:
                alphabet.clear()
                for name in self.special_symbols.names:
                    alphabet.insert(name)
                for name, count in zip(names, counts):
                    if count > cutoff:
                        alphabet.insert(name)
                if len(alphabet) < max_length:
                    break
                cutoff += 1
                logging.info('Incrementing %s cutoff to %d...' % (label, cutoff))
            alphabet.stop_growth()

        logging.info('Number of forms: %d' % len(self.form_alphabet))
        logging.info('Number of lower-case forms: %d' %
                     len(self.form_lower_alphabet))
        logging.info('Number of lemmas: %d' % len(self.lemma_alphabet))
        logging.info('Number of prefixes: %d' % len(self.prefix_alphabet))
        logging.info('Number of suffixes: %d' % len(self.suffix_alphabet))
        logging.info('Number of tags: %d' % len(self.tag_alphabet))
        logging.info('Number of morph tags: %d' % len(self.morph_tag_alphabet))
