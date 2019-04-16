from collections import OrderedDict
import numpy as np
from .dependency_instance_numeric import DependencyInstanceNumeric
from .constants import Target

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


type2target = {NextSibling: Target.NEXT_SIBLINGS,
               Grandparent: Target.GRANDPARENTS,
               GrandSibling: Target.GRANDSIBLINGS}

target2type = {Target.NEXT_SIBLINGS: NextSibling,
               Target.GRANDPARENTS: Grandparent,
               Target.GRANDSIBLINGS: GrandSibling}


class DependencyParts(object):
    def __init__(self, instance, model_type, mask=None, labeled=True,
                 num_relations=None):
        """
        A DependencyParts object stores all the parts into which a dependency
        tree is factored.

        This class has an attribute arc_mask to indicate which arcs are
        considered possible. In principle, all labels are considered possible
        for possible arcs.

        For higher order parts, it stores OrderedDict's that map the class
        (i.e., a class object, not an instance) to DependencyPart objects such
        as Grandparent, NextSibling, etc.

        :param instance: a DependencyInstanceNumeric object
        :param model_type: a ModelType object, indicating which type of parts
            should be created (siblings, grandparents, etc)
        :param labeled: whether LabeledArc parts should be used
        :param mask: either None (no prune) or a bool numpy matrix with shape
            (n, n+1) -- n is number of words without root. Cell (m, h) indicates
            if the arc from h to m is considered, if True, or pruned out, if
            False.
        :param num_relations: number of dependency relations, if used
        """
        self.index = None
        self.index_labeled = None
        self.num_parts = 0
        self.arc_mask = mask
        self.labeled = labeled
        self.num_relations = num_relations
        self.gold_arcs = None

        # store the order in which part types are used
        self.type_order = []

        # part_lists[Type] contains the list of Type parts
        self.part_lists = OrderedDict()

        # part_gold[Type] contains the gold labels for Type objects
        self.part_gold = OrderedDict()

        self._make_parts(instance, model_type)

        self.best_labels = {}

    def get_num_expected_scores(self, target):
        """
        Return the expected number of scores for a given target.

        This is useful to get the number of meaningful scores from a padded
        batch. It can return the number of arcs to be scored, or grandparents,
        tags.

        :param target: a value in Target
        :return: int
        """
        if target == Target.HEADS:
            return self.num_arcs
        elif target == Target.RELATIONS:
            return self.num_labeled_arcs
        elif target in target2type:
            # this covers higher-order features such as grandparent, siblings
            type_ = target2type[target]
            return len(self.part_lists[type_])
        else:
            # assume it is some tagging task
            return len(self.arc_mask)

    def save_best_labels(self, best_labels, arcs):
        """
        Save the best labels for each arc in a dictionary.

        :param best_labels: array with the best label for each arc
        :param arcs: list of tuples (h, m)
        """
        self.best_labels = {}
        for arc, label in zip(arcs, best_labels):
            self.best_labels[arc] = label

    def concatenate_part_scores(self, scores):
        """
        Concatenate all the vectors of part scores in the given dictionary to a
        single vector in the same order used in parts.

        :param scores: dictionary mapping target names to arrays
        :return: a single numpy array
        """
        score_list = [scores[type_] for type_ in self.type_order]
        return np.concatenate(score_list)

    def get_labels(self, heads):
        """
        Return the labels associated with the given head attachments for the
        words.

        :param heads: list or array with the head of each word in the sentence
            (root not included)
        :return: a list of predicted labels
        """
        pred_labels = []
        for m, h in enumerate(heads, 1):
            label = self.best_labels[(h, m)]
            pred_labels.append(label)

        return pred_labels

    def _make_parts(self, instance, model_type):
        """
        Create all the parts to represent the instance
        """
        # if no mask was given, create an all-True mask with a False diagonal
        if self.arc_mask is None:
            length = len(instance)
            self.arc_mask = np.ones([length - 1, length], dtype=np.bool)
            self.arc_mask[np.arange(length - 1), np.arange(1, length)] = False

        # if there are gold labels, store them
        self.gold_parts = self._make_gold_arcs(instance)
        self.type_order.append(Target.HEADS)

        # all non-masked arcs count as a part
        possible_arcs = self.arc_mask.size
        num_masked = np.sum(self.arc_mask == 0)
        self.num_arcs = possible_arcs - num_masked
        if self.labeled:
            self.num_labeled_arcs = self.num_arcs * self.num_relations
            self.type_order.append(Target.RELATIONS)
        else:
            self.num_labeled_arcs = 0

        self.num_parts = self.num_arcs + self.num_labeled_arcs

        if model_type.consecutive_siblings:
            raise NotImplemented
        if model_type.grandparents:
            raise NotImplemented
        if model_type.grandsiblings:
            raise NotImplemented

        self.gold_parts = np.array(self.gold_parts, dtype=np.float32)
        assert self.num_parts == len(self.gold_parts)

    def _make_gold_arcs(self, instance):
        """
        If the instance has gold heads, create a list with the gold arcs and
        gold relations.

        :type instance: DependencyInstanceNumeric
        :return: a list of 0s and 1s
        """
        # skip root
        heads = instance.get_all_heads()[1:]
        if heads[0] == -1:
            return

        relations = instance.get_all_relations()[1:]
        gold_parts = []
        gold_relations = []
        length = len(instance)

        for m in range(length - 1):
            # -1 to skip root
            for h in range(length):
                if not self.arc_mask[m, h]:
                    continue

                gold_head = heads[m] == h
                if gold_head:
                    gold_parts.append(1)
                else:
                    gold_parts.append(0)

                if not self.labeled:
                    continue

                for rel in range(self.num_relations):
                    if gold_head and relations[m] == rel:
                        gold_relations.append(1)
                    else:
                        gold_relations.append(0)

        gold_parts.extend(gold_relations)
        return gold_parts

    def has_type(self, type_):
        """
        Return whether this object stores parts of a particular type.

        :param type_: a class such as NextSibling or
            Grandparent
        :return: boolean
        """
        return type_ in self.part_lists and len(self.part_lists[type_]) > 0

    def __len__(self):
        return self.num_parts

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
