from .dependency_instance import DependencyInstance, \
    DependencyInstanceInput, DependencyInstanceOutput


class DependencyInstanceNumericInput(DependencyInstanceInput):
    def __init__(self, input, token_dictionary):
        self.characters = [None for _ in range(len(input.forms))]
        self.embedding_ids = [-1] * len(input.forms)
        self.forms = [-1] * len(input.forms)
        self.forms_lower = [-1] * len(input.forms)
        self.lemmas = [-1] * len(input.lemmas)
        self.prefixes = [-1] * len(input.forms)
        self.suffixes = [-1] * len(input.forms)
        self.tags = [-1] * len(input.tags)
        self.fine_tags = [-1] * len(input.fine_tags)
        self.morph_tags = [[-1] * len(morph_tags)
                           for morph_tags in input.morph_tags]
        self.is_noun = [False] * len(input.forms)
        self.is_verb = [False] * len(input.forms)
        self.is_punc = [False] * len(input.forms)
        self.is_coord = [False] * len(input.forms)

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
            tag = input.tags[i]
            id_ = token_dictionary.get_tag_id(tag)
            assert id_ < 0xff
            self.tags[i] = id_

            tag = input.fine_tags[i]
            id_ = token_dictionary.get_fine_tag_id(tag)
            self.fine_tags[i] = id_

            # Morphological tags.
            morph_tags = input.morph_tags[i]
            for j in range(len(morph_tags)):
                morph_tag = morph_tags[j]
                id_ = token_dictionary.get_morph_tag_id(morph_tag)
                assert id_ < 0xffff
                self.morph_tags[i][j] = id_

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
    def __init__(self, output, token_dictionary, relation_dictionary):
        self.heads = [-1] * len(output.heads)
        self.relations = [-1] * len(output.relations)
        if output.tags is None:
            self.tags = None
        else:
            self.tags = [-1] * len(output.tags)

        for i in range(len(output.heads)):
            self.heads[i] = output.heads[i]
            relation = output.relations[i]
            self.relations[i] = relation_dictionary.get_relation_id(relation)
            if self.tags is not None:
                tag = output.tags[i]
                self.tags[i] = token_dictionary.get_tag_id(tag)


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
