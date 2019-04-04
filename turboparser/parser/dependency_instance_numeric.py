from .dependency_instance import DependencyInstance

import numpy as np


class DependencyInstanceNumeric(DependencyInstance):
    """An dependency parsing instance with numeric fields."""
    def __init__(self, instance, token_dictionary):
        """
        :param instance: DependencyInstance
        :param token_dictionary: TokenDictionary
        :param relation_dictionary: DependencyDictionary
        """
        length = len(instance)

        self.characters = [None for _ in range(length)]
        self.forms = np.full(length, -1, np.int32)
        self.forms_lower = self.forms.copy()
        self.embedding_ids = self.forms.copy()
        self.lemmas = self.forms.copy()
        self.prefixes = self.forms.copy()
        self.suffixes = self.forms.copy()
        self.upos = self.forms.copy()
        self.xpos = self.forms.copy()
        self.morph_tags = [[-1] * len(morph_tags)
                           for morph_tags in instance.morph_tags]
        self.morph_singletons = self.forms.copy()

        self.heads = np.full(length, -1, np.int32)
        self.relations = self.heads.copy()
        self.multiwords = instance.multiwords

        for i in range(length):
            # Form and lower-case form.
            form = instance.forms[i]
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
            lemma = instance.lemmas[i]
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
            tag = instance.upos[i]
            id_ = token_dictionary.get_upos_id(tag)
            assert id_ < 0xff
            self.upos[i] = id_

            tag = instance.xpos[i]
            id_ = token_dictionary.get_xpos_id(tag)
            self.xpos[i] = id_

            # Morphological tags.
            morph_tags = instance.morph_tags[i]
            for j in range(len(morph_tags)):
                morph_tag = morph_tags[j]
                id_ = token_dictionary.get_morph_tag_id(morph_tag)
                assert id_ < 0xffff
                self.morph_tags[i][j] = id_

            morph_singleton = instance.morph_singletons[i]
            self.morph_singletons[i] = token_dictionary.\
                get_morph_singleton_id(morph_singleton)

            self.heads[i] = instance.heads[i]
            relation = instance.relations[i]
            self.relations[i] = token_dictionary.get_deprel_id(relation)

    def get_characters(self, i):
        return self.characters[i]

    def get_embedding_id(self, i):
        return self.embedding_ids[i]

    def get_all_embedding_ids(self):
        return self.embedding_ids

    def get_form_lower(self, i):
        return self.forms_lower[i]

    def get_prefix(self, i):
        return self.prefixes[i]

    def get_suffix(self, i):
        return self.suffixes[i]
