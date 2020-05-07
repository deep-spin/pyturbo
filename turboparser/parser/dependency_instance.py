from .constants import ROOT


num_conllu_fields = 10
multiword_blanks = '\t'.join(['_'] * 8)


class MultiwordSpan(object):
    """
    Class for storing a multiword token, including its text form and text span.
    """
    def __init__(self, first, last, form):
        """
        Create a multiword token representation. For example, a token as

        9-10	ao	_

        should be instantiated as MultiwordSpan(9, 10, 'ao')

        :param first: The first position of the words included in the multiword
            token
        :param last: Same as above, for the last position
        :param form: The token form as appears in the text
        """
        self.first = first
        self.last = last
        self.form = form


class DependencyInstance(object):
    """A dependency parsing instance."""
    def __init__(self, forms, lemmas, upos, xpos, morph_tags, morph_singletons,
                 heads, relations, multiwords):
        self.forms = forms
        self.lemmas = lemmas
        self.upos = upos
        self.xpos = xpos
        self.morph_tags = morph_tags
        self.morph_singletons = morph_singletons
        self.heads = heads
        self.relations = relations
        self.multiwords = multiwords

    @classmethod
    def from_tokens(cls, tokens):
        """Create an instance from tokens only, without any annotation"""
        tokens = [ROOT] + tokens
        r = range(len(tokens))
        empty_list = ['_' for _ in r]
        empty_list[0] = ROOT
        heads = [-1 for _ in r]
        morph_tags = [{} for _ in r]

        instance = DependencyInstance(
            tokens, empty_list, empty_list.copy(), empty_list.copy(),
            morph_tags, empty_list.copy(), heads, empty_list.copy(), [])

        return instance

    def __len__(self):
        return len(self.forms)

    def get_form(self, i):
        return self.forms[i]

    def get_lemma(self, i):
        return self.lemmas[i]

    def get_upos(self, i):
        return self.upos[i]

    def get_all_upos(self):
        return self.upos

    def get_xpos(self, i):
        return self.xpos[i]

    def get_all_xpos(self):
        return self.xpos

    def get_all_morph_singletons(self):
        return self.morph_singletons

    def get_all_morph_tags(self):
        return self.morph_tags

    def get_morph_tags(self, i):
        """
        Return the morphological features for token i
        """
        return self.morph_tags[i]

    def get_morph_singleton(self, i):
        """
        Return the singleton morphological data of the i-th word
        """
        return self.morph_singletons[i]

    def get_head(self, i):
        return self.heads[i]

    def get_all_heads(self):
        """Return all heads in the sentence, including root"""
        return self.heads

    def get_relation(self, i):
        return self.relations[i]

    def get_all_relations(self):
        return self.relations

    def get_all_forms(self):
        return self.forms

    def get_all_lemmas(self):
        return self.lemmas

    def to_conll(self) -> str:
        """Return a string in CONLLU format"""
        # keep track of multiword tokens
        multiword = self.multiwords[0] if len(self.multiwords) else None
        multiword_idx = 0
        lines = []

        for i in range(1, len(self)):
            if multiword and i == multiword.first:
                span = '%d-%d' % (multiword.first, multiword.last)
                line = '%s\t%s\t%s\n' % (span, multiword.form, multiword_blanks)
                lines.append(line)

                multiword_idx += 1
                if multiword_idx >= len(self.multiwords):
                    multiword = None
                else:
                    multiword = self.multiwords[multiword_idx]

            line = '\t'.join([str(i), self.forms[i], self.lemmas[i],
                              self.upos[i], self.xpos[i],
                              self.morph_singletons[i], str(self.heads[i]),
                              self.relations[i], '_', '_'])
            lines.append(line)

        return '\n'.join(lines)
