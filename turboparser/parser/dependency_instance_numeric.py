from .dependency_instance import DependencyInstance, \
    DependencyInstanceInput, DependencyInstanceOutput

import numpy as np


class DependencyInstanceNumericInput(DependencyInstanceInput):
    def __init__(self, input, token_dictionary):
        length = len(input.forms)

        self.characters = [None for _ in range(len(input.forms))]
        self.forms = np.full(length, -1, np.int32)
        self.forms_lower = self.forms.copy()
        self.embedding_ids = self.forms.copy()
        self.lemmas = self.forms.copy()
        self.prefixes = self.forms.copy()
        self.suffixes = self.forms.copy()
        self.upos = self.forms.copy()
        self.xpos = self.forms.copy()
        self.morph_tags = [[-1] * len(morph_tags)
                           for morph_tags in input.morph_tags]
        self.morph_singletons = self.forms.copy()
        # self.is_noun = [False] * len(input.forms)
        # self.is_verb = [False] * len(input.forms)
        # self.is_punc = [False] * len(input.forms)
        # self.is_coord = [False] * len(input.forms)

        for i in range(len(input.forms)):
            # Form and lower-case form.
            form = input.forms[i]
            form_lower = form.lower()
            if not token_dictionary.classifier.options.form_case_sensitive:
                form = form_lower
            id_ = token_dictionary.get_form_id(form)
            assert id_ < 0xffff
            self.forms[i] = id_

            id_ = token_dictionary.get_form_lower_id(form_lower)
            assert id_ < 0xffff
            self.forms_lower[i] = id_

            id_ = token_dictionary.get_embedding_id(form)
            self.embedding_ids[i] = id_

            # characters
            self.characters[i] = [token_dictionary.get_character_id(c)
                                  for c in form]

            # Lemma.
            lemma = input.lemmas[i]
            id_ = token_dictionary.get_lemma_id(lemma)
            assert id_ < 0xffff
            self.lemmas[i] = id_

            # Prefix.
            prefix = form[:token_dictionary.classifier.options.prefix_length]
            id_ = token_dictionary.get_prefix_id(prefix)
            assert id_ < 0xffff
            self.prefixes[i] = id_

            # Suffix.
            suffix = form[-token_dictionary.classifier.options.suffix_length:]
            id_ = token_dictionary.get_suffix_id(suffix)
            assert id_ < 0xffff
            self.suffixes[i] = id_

            # POS tag.
            tag = input.upos[i]
            id_ = token_dictionary.get_upos_id(tag)
            assert id_ < 0xff
            self.upos[i] = id_

            tag = input.xpos[i]
            id_ = token_dictionary.get_xpos_id(tag)
            self.xpos[i] = id_

            # Morphological tags.
            morph_tags = input.morph_tags[i]
            for j in range(len(morph_tags)):
                morph_tag = morph_tags[j]
                id_ = token_dictionary.get_morph_tag_id(morph_tag)
                assert id_ < 0xffff
                self.morph_tags[i][j] = id_

            morph_singleton = input.morph_singletons[i]
            self.morph_singletons[i] = token_dictionary.\
                get_morph_singleton_id(morph_singleton)

            # # Check whether the word is a noun, verb, punctuation or
            # # coordination. Note: this depends on the POS tag string.
            # # This procedure is taken from EGSTRA
            # # (http://groups.csail.mit.edu/nlp/egstra/).
            # self.is_noun[i] = input.upos[i][0] in ['n', 'N']
            # self.is_verb[i] = input.upos[i][0] in ['v', 'V']
            # self.is_punc[i] = input.upos[i] in ['Punc', '$,', '$.', 'PUNC',
            #                                  'punc', 'F', 'IK', 'XP', ',', ';']
            # self.is_coord[i] = input.upos[i] in ['Conj', 'KON', 'conj',
            #                                      'Conjunction', 'CC', 'cc']


#TODO: merge input and output in a single class
class DependencyInstanceNumericOutput(DependencyInstanceOutput):
    def __init__(self, output, token_dictionary, relation_dictionary):
        length = len(output.heads)
        self.heads = np.full(length, -1, np.int32)
        self.relations = self.heads.copy()
        self.upos = self.heads.copy()
        self.xpos = self.heads.copy()
        self.morph_singletons = self.heads.copy()

        for i in range(len(output.heads)):
            self.heads[i] = output.heads[i]
            relation = output.relations[i]
            self.relations[i] = relation_dictionary.get_relation_id(relation)
            if output.upos is not None:
                tag = output.upos[i]
                self.upos[i] = token_dictionary.get_upos_id(tag)
            if output.xpos is not None:
                tag = output.xpos[i]
                self.xpos[i] = token_dictionary.get_xpos_id(tag)
            if output.morph_singletons is not None:
                tag = output.morph_singletons[i]
                self.morph_singletons[i] = token_dictionary.\
                    get_morph_singleton_id(tag)


class DependencyInstanceNumeric(DependencyInstance):
    '''An dependency parsing instance with numeric fields.'''
    def __init__(self, instance, token_dictionary, relation_dictionary):
        """
        :param instance: DependencyInstance
        :param token_dictionary: TokenDictionary
        :param relation_dictionary: DependencyDictionary
        """
        if instance.output is None:
            output = None
        else:
            output = DependencyInstanceNumericOutput(instance.output,
                                                     token_dictionary,
                                                     relation_dictionary)
        DependencyInstance.__init__(
            self, DependencyInstanceNumericInput(instance.input,
                                                 token_dictionary),
            output)

    def __len__(self):
        return len(self.input.forms)

    def get_characters(self, i):
        return self.input.characters[i]

    def get_embedding_id(self, i):
        return self.input.embedding_ids[i]

    def get_all_embedding_ids(self):
        return self.input.embedding_ids

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
