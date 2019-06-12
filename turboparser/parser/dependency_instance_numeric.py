from .dependency_instance import DependencyInstance
from .token_dictionary import TokenDictionary
import numpy as np


class DependencyInstanceNumeric(DependencyInstance):
    """An dependency parsing instance with numeric fields."""
    def __init__(self, instance, token_dictionary, case_sensitive):
        """
        :param instance: DependencyInstance
        :param token_dictionary: TokenDictionary
        :type token_dictionary: TokenDictionary
        :param case_sensitive: bool
        """
        length = len(instance)

        self.characters = [None for _ in range(length)]
        self.forms = np.full(length, -1, np.int32)
        self.embedding_ids = self.forms.copy()
        self.lemmas = self.forms.copy()
        self.upos = self.forms.copy()
        self.xpos = self.forms.copy()
        self.morph_tags = self.characters.copy()
        self.morph_singletons = self.forms.copy()

        self.heads = np.full(length, -1, np.int32)
        self.relations = self.heads.copy()
        self.multiwords = instance.multiwords

        for i in range(length):
            # Form and lower-case form.
            form = instance.forms[i]
            lower_form = form.lower()
            if not case_sensitive:
                id_ = token_dictionary.get_form_id(lower_form)
            else:
                id_ = token_dictionary.get_form_id(form)
            self.forms[i] = id_

            id_ = token_dictionary.get_embedding_id(lower_form)
            self.embedding_ids[i] = id_

            # characters
            self.characters[i] = [token_dictionary.get_character_id(c)
                                  for c in form]

            # Lemma.
            lemma = instance.lemmas[i]
            id_ = token_dictionary.get_lemma_id(lemma)
            self.lemmas[i] = id_

            # POS tag.
            tag = instance.upos[i]
            id_ = token_dictionary.get_upos_id(tag)
            self.upos[i] = id_

            tag = instance.xpos[i]
            id_ = token_dictionary.get_xpos_id(tag)
            self.xpos[i] = id_

            # Morphological tags.
            morph_tags = instance.morph_tags[i]
            self.morph_tags[i] = token_dictionary.get_morph_ids(morph_tags)

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
