from collections import OrderedDict
import numpy as np


class DependencyPart(object):
    """
    Base class for Dependency Parts
    """
    __slots__ = 'head', 'modifier'

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
    __slots__ = 'label',

    def __init__(self, head=-1, modifier=-1, label=-1):
        self.head = head
        self.modifier = modifier
        self.label = label


class Grandparent(DependencyPart):
    __slots__ = 'grandparent',

    def __init__(self, head=-1, modifier=-1, grandparent=-1):
        self.head = head
        self.modifier = modifier
        self.grandparent = grandparent


class NextSibling(DependencyPart):
    __slots__ = 'sibling'

    def __init__(self, head=-1, modifier=-1, sibling=-1):
        self.head = head
        self.modifier = modifier
        self.sibling = sibling


class GrandSibling(DependencyPart):
    __slots__ = 'grandparent', 'sibling'

    def __init__(self, head=-1, modifier=-1, grandparent=-1, sibling=-1):
        self.head = head
        self.modifier = modifier
        self.sibling = sibling
        self.grandparent = grandparent


class DependencyParts(object):
    def __init__(self, instance, model_type):
        """
        A DependencyParts object stores all the parts into which a dependency
        tree is factored.

        This class has an attribute arc_mask, a matrix with shape (n, n) in
        which cell (m, h) indicates if the arc from h to m is considered, if
        non-zero, or pruned out, if zero.

        In principle, all labels are considered possible for all arcs.

        For higher order parts, it stores OrderedDict's that map the class
        (i.e., a class object, not an instance) to DependencyPart objects such
        as Grandparent, NextSibling, etc.

        :param instance: a DependencyInstanceNumeric object
        :param model_type: a ModelType object, indicating which type of parts
            should be created (siblings, grandparents, etc)
        """
        self.index = None
        self.index_labeled = None

        length = len(instance)
        self.arc_mask = np.ones([length, length], np.int)
        self.gold_arcs = None
        self._set_gold_arcs(instance)

        # part_lists[Arc] contains the list of Arc objects; same for others
        self.part_lists = OrderedDict()

        # part_gold[Arc] contains the gold labels for Arc objects
        self.part_gold = OrderedDict()

        # the i-th position stores the best label found for arc i
        self.best_labels = []

    def _set_gold_arcs(self, instance):
        """
        If the instance has gold heads, store the gold arcs in arc matrix.
        """
        heads = instance.get_all_heads()
        if heads[1] == -1:
            # check [-1] because [0] is the root
            return

        self.gold_arcs = np.ones_like(self.arc_mask)
        for m, h in enumerate(heads[1:], 1):
            self.gold_arcs[m, h] = 1

    def has_type(self, type_):
        """
        Return whether this object stores parts of a particular type.

        :param type_: a class such as NextSibling or
            Grandparent
        :return: boolean
        """
        return type_ in self.part_lists and len(self.part_lists[type_]) > 0

    def __len__(self):
        return sum(len(part_list) for part_list in self.part_lists.values())

    def get_num_type(self, type_):
        """
        Return the number of parts of the given type
        """
        if type_ not in self.part_lists:
            return 0
        return len(self.part_lists[type_])

    def get_type_offset(self, type_):
        """
        Return the offset of the given type in the ordered array with gold data.
        """
        if type_ not in self.part_lists:
            return -1

        offset = 0
        for type_i in self.part_lists:
            if type_i == type_:
                return offset
            else:
                offset += len(self.part_lists[type_i])

    def get_gold_output(self):
        """
        Return a single list with all gold values, in the order that parts were
        added.

        If first Arc parts were added, then some Sibling, then more Arcs, the
        gold list will have all Arcs and then all Siblings.

        :return: a list if any output exists, None otherwise
        """
        all_gold = []
        for type_ in self.part_gold:
            all_gold.extend(self.part_gold[type_])

        if len(all_gold) == 0:
            return None

        return all_gold

    def append(self, part, gold=None):
        """
        Append the object to the internal list. If it's a first order arc, also
        update the position index.

        :param part: a DependencyPart
        :param gold: either 1 or 0
        """
        part_type = type(part)
        if part_type not in self.part_lists:
            self.part_lists[part_type] = []
            if gold is not None:
                self.part_gold[part_type] = []

        self.part_lists[part_type].append(part)
        if gold is not None:
            self.part_gold[part_type].append(gold)

        if isinstance(part, Arc):
            if part.head not in self.arc_index:
                self.arc_index[part.head] = {}

            index = len(self.part_lists[Arc]) - 1
            self.arc_index[part.head][part.modifier] = index

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

            index = len(self.part_lists[LabeledArc]) - 1
            head_indices[part.modifier].append(index)
            head_labels[part.modifier].append(part.label)

    def iterate_over_type(self, type_):
        """
        Iterates over the parts (arcs) of a particular type.
        """
        if type_ not in self.part_lists:
            raise StopIteration

        for part in self.part_lists[type_]:
            yield part

    def get_parts_of_type(self, type_):
        """
        Return a sublist of this object containing parts of the requested type.
        """
        return self.part_lists[type_]

    def get_gold_for_type(self, type_):
        """
        Return a list with gold values for parts of the given type
        """
        return self.part_gold[type_]

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
        Return the list of positions (within the list of labeled arcs) of the
        labeled arcs connecting `head` and `modifier`. If no such label exists,
        return an empty list.
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
