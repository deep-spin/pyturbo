from classifier.instance import Instance

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
    def __init__(self, input, output):
        Instance.__init__(input, output)
