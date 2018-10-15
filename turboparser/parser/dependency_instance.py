from ..classifier.instance import Instance


class DependencyInstanceInput(object):
    def __init__(self, forms, lemmas, tags, morph_tags):
        self.forms = forms
        self.lemmas = lemmas
        self.tags = tags
        self.morph_tags = morph_tags


class DependencyInstanceOutput(object):
    def __init__(self, heads, relations):
        self.heads = heads
        self.relations = relations


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

    def get_tag(self, i):
        return self.input.tags[i]

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
