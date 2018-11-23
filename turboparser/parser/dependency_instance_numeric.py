from .dependency_instance import DependencyInstance, \
    DependencyInstanceInput, DependencyInstanceOutput
from .token_dictionary import UNKNOWN


class DependencyInstanceNumericInput(DependencyInstanceInput):
    def __init__(self, input, dictionary):
        self.embedding_ids = [-1] * len(input.forms)
        self.forms = [-1] * len(input.forms)
        self.forms_lower = [-1] * len(input.forms)
        self.lemmas = [-1] * len(input.lemmas)
        self.prefixes = [-1] * len(input.forms)
        self.suffixes = [-1] * len(input.forms)
        self.tags = [-1] * len(input.tags)
        self.morph_tags = [[-1] * len(morph_tags)
                           for morph_tags in input.morph_tags]
        self.is_noun = [False] * len(input.forms)
        self.is_verb = [False] * len(input.forms)
        self.is_punc = [False] * len(input.forms)
        self.is_coord = [False] * len(input.forms)

        token_dictionary = dictionary.classifier.token_dictionary

        for i in range(len(input.forms)):
            # Form and lower-case form.
            form = input.forms[i]
            form_lower = form.lower()
            if not token_dictionary.classifier.options.form_case_sensitive:
                form = form_lower
            id = token_dictionary.get_form_id(form)
            assert id < 0xffff
            self.forms[i] = id

            id = token_dictionary.get_form_lower_id(form_lower)
            assert id < 0xffff
            self.forms_lower[i] = id

            id = token_dictionary.get_embedding_id(form)
            self.embedding_ids[i] = id

            # Lemma.
            lemma = input.lemmas[i]
            id = token_dictionary.get_form_id(form)
            assert id < 0xffff
            self.lemmas[i] = id

            # Prefix.
            prefix = form[:token_dictionary.classifier.options.prefix_length]
            id = token_dictionary.get_prefix_id(prefix)
            assert id < 0xffff
            self.prefixes[i] = id

            # Suffix.
            suffix = form[-token_dictionary.classifier.options.suffix_length:]
            id = token_dictionary.get_suffix_id(suffix)
            assert id < 0xffff
            self.suffixes[i] = id

            # POS tag.
            tag = input.tags[i]
            id = token_dictionary.get_tag_id(tag)
            assert id < 0xff
            self.tags[i] = id

            # Morphological tags.
            morph_tags = input.morph_tags[i]
            for j in range(len(morph_tags)):
                morph_tag = morph_tags[j]
                id = token_dictionary.get_morph_tag_id(morph_tag)
                assert id < 0xffff
                self.morph_tags[i][j] = id

            # Check whether the word is a noun, verb, punctuation or
            # coordination. Note: this depends on the POS tag string.
            # This procedure is taken from EGSTRA
            # (http://groups.csail.mit.edu/nlp/egstra/).
            self.is_noun[i] = input.tags[i][0] in ['n', 'N']
            self.is_verb[i] = input.tags[i][0] in ['v', 'V']
            self.is_punc[i] = input.tags[i] in ['Punc', '$,', '$.', 'PUNC',
                                             'punc', 'F', 'IK', 'XP', ',', ';']
            self.is_coord[i] = input.tags[i] in ['Conj', 'KON', 'conj',
                                                 'Conjunction', 'CC', 'cc']

class DependencyInstanceNumericOutput(DependencyInstanceOutput):
    def __init__(self, output, dictionary):
        self.heads = [-1] * len(output.heads)
        self.relations = [-1] * len(output.relations)

        token_dictionary = dictionary.classifier.token_dictionary

        for i in range(len(output.heads)):
            self.heads[i] = output.heads[i]
            relation = output.relations[i]
            id = dictionary.relation_alphabet.lookup(relation)
            assert id < 0xff # Check this.
            if id < 0:
                id = token_dictionary.token_unknown
            self.relations[i] = id

class DependencyInstanceNumeric(DependencyInstance):
    '''An dependency parsing instance with numeric fields.'''
    def __init__(self, instance, dictionary):
        if instance.output is None:
            output = None
        else:
            output = DependencyInstanceNumericOutput(instance.output,
                                                     dictionary)
        DependencyInstance.__init__(
            self, DependencyInstanceNumericInput(instance.input, dictionary),
            output)

    def __len__(self):
        return len(self.input.forms)

    def get_form_lower(self, i):
        return self.input.forms_lower[i]

    def get_prefix(self, i):
        return self.input.prefixes[i]

    def get_suffix(self, i):
        return self.input.suffixes[i]

    def is_verb(self, i):
        return self.input.is_verb[i]

    def is_punc(self, i):
        return self.input.is_punc[i]

    def is_coord(self, i):
        return self.input.is_coord[i]
