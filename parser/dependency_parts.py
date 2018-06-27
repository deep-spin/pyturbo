class DependencyPartArc(object):
    def __init__(self, head=-1, modifier=-1):
        self.head = head
        self.modifier = modifier

class DependencyPartLabeledArc(object):
    def __init__(self, head=-1, modifier=-1, label=-1):
        self.head = head
        self.modifier = modifier
        self.label = label

class DependencyParts(list):
    def __init__(self):
        self.index = None
        self.index_labeled = None
        self.offsets = {}

    def get_offset(self, type):
        return self.offsets[type]

    def set_offset(self, type, offset, size):
        self.offsets[type] = (offset, size)
