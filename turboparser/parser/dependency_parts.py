class DependencyPart(object):
    """
    Base class for Dependency Parts
    """
    def __init__(self):
        raise NotImplementedError('Abstract class')

    def __str__(self):
        nodes = [('h', self.head), ('m', self.modifier)]
        if hasattr(self, 'grandparent'):
            nodes.append(('g', self.grandparent))
        if hasattr(self, 'sibling'):
            nodes.append(('s', self.sibling))
        if hasattr(self, 'label'):
            nodes.append(('l', self.label))

        nodes_strings = ['{}={}'.format(node[0], node[1]) for node in nodes]
        str_ = self.__class__.__name__ + '(' + ', '.join(nodes_strings) + ')'
        return str_


class Arc(DependencyPart):
    def __init__(self, head=-1, modifier=-1):
        self.head = head
        self.modifier = modifier


class LabeledArc(DependencyPart):
    def __init__(self, head=-1, modifier=-1, label=-1):
        self.head = head
        self.modifier = modifier
        self.label = label


class Grandparent(DependencyPart):
    def __init__(self, head=-1, modifier=-1, grandparent=-1):
        self.head = head
        self.modifier = modifier
        self.grandparent = grandparent


class NextSibling(DependencyPart):
    def __init__(self, head=-1, modifier=-1, sibling=-1):
        self.head = head
        self.modifier = modifier
        self.sibling = sibling


class GrandSibling(DependencyPart):
    def __init__(self, head=-1, modifier=-1, grandparent=-1, sibling=-1):
        self.head = head
        self.modifier = modifier
        self.sibling = sibling
        self.grandparent = grandparent


class DependencyParts(list):
    def __init__(self):
        self.index = None
        self.index_labeled = None
        self.offsets = {}
        self.arc_index = {}
        self.labeled_indices = {}
        self.arc_labels = {}

        # the i-th position stores the best label found for arc i
        self.best_labels = []

    def has_type(self, type_):
        """
        Return whether this object stores parts of a particular type.

        :param type_: a class such as NextSibling or
            Grandparent
        :return: boolean
        """
        return type_ in self.offsets and self.offsets[type_][1] > 0

    def append(self, part):
        """
        Append the object to the internal list. If it's a first order arc, also
        update the position index.

        :param part: a DependencyPart
        """
        super(DependencyParts, self).append(part)
        if isinstance(part, Arc):
            if part.head not in self.arc_index:
                self.arc_index[part.head] = {}

            self.arc_index[part.head][part.modifier] = len(self) - 1

        elif isinstance(part, LabeledArc):
            if part.head not in self.labeled_indices:
                self.labeled_indices[part.head] = {}

            if part.head not in self.arc_labels:
                self.arc_labels[part.head] = {}

            head_indices = self.labeled_indices[part.head]
            head_labels = self.arc_labels[part.head]
            if part.modifier not in head_indices:
                head_indices[part.modifier] = []

            if part.modifier not in head_labels:
                head_labels[part.modifier] = []

            position = len(self) - 1
            head_indices[part.modifier].append(position)
            head_labels[part.modifier].append(part.label)

    def iterate_over_type(self, type_, return_index=False):
        """
        Iterates over the parts (arcs) of a particular type.

        If return_index is True, also return the index of the part in the
        list.
        """
        offset, size = self.get_offset(type_)
        for i in range(offset, offset + size):
            if return_index:
                yield i, self[i]
            else:
                yield self[i]

    def get_offset(self, type_):
        return self.offsets[type_]

    def set_offset(self, type_, offset, size):
        self.offsets[type_] = (offset, size)

    def find_arc_index(self, head, modifier):
        """
        Return the position of the arc connecting `head` and `modifier`. If no
        such arc exists, return -1.

        :param head: integer
        :param modifier: integer
        :return: integer
        """
        if head not in self.arc_index or modifier not in self.arc_index[head]:
            return -1

        return self.arc_index[head][modifier]

    def find_labeled_arc_indices(self, head, modifier):
        """
        Return the list of positions of the labeled arcs connecting `head` and
        `modifier`. If no such label exists, return an empty list.
        """
        if head not in self.labeled_indices:
            return []

        head_dict = self.labeled_indices[head]
        if modifier not in head_dict:
            return []

        return head_dict[modifier]

    def find_arc_labels(self, head, modifier):
        """
        Return a list of all possible labels for the arc between `head` and
        `modifier`. If there is none, return an empty list.
        """
        if head not in self.arc_labels:
            return []

        head_dict = self.arc_labels[head]
        if modifier not in head_dict:
            return []

        return head_dict[modifier]
