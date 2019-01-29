from ..classifier.instance import Instance


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


class DependencyInstanceInput(object):
    def __init__(self, forms, lemmas, upos, xpos, morph_tags, multiwords):
        self.forms = forms
        self.lemmas = lemmas
        self.upos = upos
        self.xpos = xpos
        self.morph_tags = morph_tags
        self.multiwords = multiwords


class DependencyInstanceOutput(object):
    def __init__(self, heads, relations, upos=None, xpos=None, morph_tags=None):
        self.heads = heads
        self.relations = relations
        self.upos = upos
        self.xpos = xpos
        self.morph_tags = morph_tags


class DependencyInstance(Instance):
    '''An dependency parsing instance.'''
    def __init__(self, input, output=None):
        Instance.__init__(self, input, output)

    def __len__(self):
        return len(self.input.forms)

    def get_form(self, i):
        return self.input.forms[i]

    def get_lemma(self, i):
        return self.input.lemmas[i]

    def get_upos(self, i):
        return self.input.upos[i]

    def get_all_upos(self):
        return self.input.upos

    def get_xpos(self, i):
        return self.input.xpos[i]

    def get_all_xpos(self):
        return self.input.xpos

    def get_num_morph_tags(self, i):
        return len(self.input.morph_tags[i])

    def get_morph_tag(self, i, j):
        """
        Return the j-th morphological attribute of the i-th word
        """
        return self.input.morph_tags[i][j]

    def get_head(self, i):
        return self.output.heads[i]

    def get_relation(self, i):
        return self.output.relations[i]
