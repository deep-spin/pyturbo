from .dependency_instance import DependencyInstance
from .token_dictionary import TokenDictionary
from .constants import ROOT, EMPTY
import numpy as np


class DependencyInstanceNumeric(DependencyInstance):
    """An dependency parsing instance with numeric fields."""
    def __init__(self, instance, token_dictionary, case_sensitive,
                 bert_tokenizer):
        """
        Store numpy array containing the instance's words, lemmas, POS, morph
        tags and dependencies.

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
        self.lemma_characters = self.characters.copy()
        self.upos = self.forms.copy()
        self.xpos = self.forms.copy()
        self.morph_tags = self.characters.copy()
        self.morph_singletons = self.forms.copy()

        self.heads = np.full(length, -1, np.int32)
        self.relations = self.heads.copy()
        self.multiwords = instance.multiwords

        # join to tokenize faster everything
        sentence = ' '.join(self.forms)
        bert_tokens = bert_tokenizer.tokenize(sentence)
        self.bert_token_starts = np.array([not token.startswith('##')
                                           for token in bert_tokens], np.bool)
        cls_id = bert_tokenizer.cls_token_id
        sep_id = bert_tokenizer.sep_token_id
        token_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
        bert_ids = [cls_id] + token_ids + [sep_id]
        self.bert_ids = np.array(bert_ids)

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

            # Lemma.
            lemma = instance.lemmas[i]
            id_ = token_dictionary.get_lemma_id(lemma)
            self.lemmas[i] = id_

            # characters and morph tags
            morph_tags = instance.morph_tags[i]
            if i == 0:
                characters = [token_dictionary.get_character_id(ROOT)]
                morph_tags = token_dictionary.get_morph_ids(morph_tags, ROOT)
                lemma_characters = [token_dictionary.get_character_id(ROOT)]
            else:
                characters = [token_dictionary.get_character_id(c)
                              for c in form]
                morph_tags = token_dictionary.get_morph_ids(morph_tags)
                lemma_characters = [token_dictionary.get_character_id(c)
                                    for c in lemma]

            self.characters[i] = np.array(characters)
            self.morph_tags[i] = np.array(morph_tags)
            self.lemma_characters[i] = np.array(lemma_characters)

            # POS tag.
            tag = instance.upos[i]
            if tag == '_':
                tag = EMPTY
            id_ = token_dictionary.get_upos_id(tag)
            self.upos[i] = id_

            tag = instance.xpos[i]
            if tag == '_':
                tag = EMPTY
            id_ = token_dictionary.get_xpos_id(tag)
            self.xpos[i] = id_

            # Morphological tags.
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
