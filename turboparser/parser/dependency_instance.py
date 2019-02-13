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


class DependencyInstance():
    '''An dependency parsing instance.'''
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

    def get_num_morph_tags(self, i):
        return len(self.morph_tags[i])

    def get_morph_tag(self, i, j):
        """
        Return the j-th morphological attribute of the i-th word
        """
        return self.morph_tags[i][j]

    def get_morph_singleton(self, i):
        """
        Return the singleton morphological data of the i-th word
        """
        return self.morph_singletons[i]

    def get_head(self, i):
        return self.heads[i]

    def get_relation(self, i):
        return self.relations[i]
