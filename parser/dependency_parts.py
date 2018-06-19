

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
        self.offsets = []

    def create_part_arc(self, head, modifier):
        return DependencyPartArc(head, modifier)

    def create_part_labeled_arc(self, head, modifier, label):
        return DependencyPartLabeledArc(head, modifier, label)

