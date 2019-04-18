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

        self.make_parts(instance, model_type)

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

    def make_parts(self, instance, model_type):
        """
        Create all the parts to represent the instance
        """
        # if no mask was given, create an all-True mask with a False diagonal
        if self.arc_mask is None:
            length = len(instance)
            self.arc_mask = np.ones([length - 1, length], dtype=np.bool)
            self.arc_mask[np.arange(length - 1), np.arange(1, length)] = False

        # TODO: enforce connectedness (necessary if pruning by tag or distance)

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

        if model_type.consecutive_siblings:
            self.make_consecutive_siblings(instance)

        if model_type.grandparents:
            self.make_grandparents(instance)

        if model_type.grandsiblings:
            self.make_grandsiblings(instance)

        self.num_parts = self.num_arcs + self.num_labeled_arcs + \
            sum(len(parts) for parts in self.part_lists.values())
        for type_ in self.part_lists:
            self.type_order.append(type_)

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
            self.make_gold = False
            return

        self.make_gold = True
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

    def make_grandparents(self, instance):
        """
        Create the parts relative to grandparents.

        Each part means that an arc h -> m and g -> h exist at the same time.

        :type instance: DependencyInstanceNumeric
        """
        gp_parts = []
        for g in range(len(instance)):
            for h in range(1, len(instance)):
                if g == h:
                    continue

                if not self.arc_mask[h - 1, g]:
                    # the arc g -> h has been pruned out
                    continue

                gold_gh = instance.get_head(h) == g

                for m in range(1, len(instance)):
                    if h == m:
                        # g == m is necessary to run the grandparent factor
                        continue

                    if not self.arc_mask[m - 1, h]:
                        # pruned out
                        continue

                    part = Grandparent(h, m, g)
                    if self.make_gold:
                        if gold_gh and instance.get_head(m) == h:
                            gold = 1
                        else:
                            gold = 0
                        self.gold_parts.append(gold)

                    gp_parts.append(part)

        self.part_lists[Target.GRANDPARENTS] = gp_parts

    def make_consecutive_siblings(self, instance):
        """
        Create the parts relative to consecutive siblings.

        Each part means that an arc h -> m and h -> s exist at the same time,
        with both h > m and h > s or both h < m and h < s.

        :param instance: DependencyInstance
        :type instance: DependencyInstanceNumeric
        """
        parts = []
        for h in range(len(instance)):

            # siblings to the right of h
            # when m = h, it signals that s is the first child
            for m in range(h, len(instance)):

                if h != m and not self.arc_mask[m - 1, h]:
                    # pruned out
                    continue

                gold_hm = m == h or instance.get_head(m) == h
                arc_between = False

                # when s = length, it signals that m encodes the last child
                for s in range(m + 1, len(instance) + 1):
                    if s < len(instance) and not self.arc_mask[s - 1, h]:
                        # pruned out
                        continue

                    if self.make_gold:
                        gold_hs = s == len(instance) or \
                                    instance.get_head(s) == h

                        if gold_hm and gold_hs and not arc_between:
                            gold = 1
                            arc_between = True
                        else:
                            gold = 0

                        self.gold_parts.append(gold)
                    part = GrandSibling(h, m, s)
                    parts.append(part)

            # siblings to the left of h
            for m in range(h, -1, -1):
                if h != m and not self.arc_mask[m - 1, h]:
                    # pruned out
                    continue

                gold_hm = m == h or instance.get_head(m) == h
                arc_between = False

                # when s = 0, it signals that m encoded the leftmost child
                for s in range(m - 1, -2, -1):
                    if s == 0 or (s != -1 and not self.arc_mask[s - 1, h]):
                        # pruned out
                        continue

                    if self.make_gold:
                        gold_hs = s == -1 or instance.get_head(s) == h

                        if gold_hm and gold_hs and not arc_between:
                            gold = 1
                            arc_between = True
                        else:
                            gold = 0

                        self.gold_parts.append(gold)
                    part = GrandSibling(h, m, s)
                    parts.append(part)

        self.part_lists[Target.NEXT_SIBLINGS] = parts

    def make_grandsiblings(self, instance):
        """
        Create the parts relative to grandsibling nodes.

        Each part means that arcs g -> h, h -> m, and h ->s exist at the same
        time.
        :type instance: DependencyInstanceNumeric
        """
        parts = []
        for g in range(len(instance)):
            for h in range(1, len(instance)):
                if g == h:
                    continue

                if not self.arc_mask[h - 1, g]:
                    # pruned
                    continue

                gold_gh = instance.get_head(h) == g

                # check modifiers to the right
                for m in range(h, len(instance)):
                    if h != m and not self.arc_mask[m - 1, h]:
                        # pruned; h == m signals first child
                        continue

                    gold_hm = m == h or instance.get_head(m) == h
                    arc_between = False

                    for s in range(m + 1, len(instance) + 1):
                        if s < len(instance) and not self.arc_mask[s - 1, h]:
                            # pruned; s == len signals last child
                            continue

                        gold_hs = s == len(instance) or \
                            instance.get_head(s) == h

                        if self.make_gold:
                            gold = 0
                            if gold_hm and gold_hs and not arc_between:
                                if gold_gh:
                                    gold = 1

                                arc_between = True
                            self.gold_parts.append(gold)

                        part = GrandSibling(h, m, g, s)
                        parts.append(part)

                # check modifiers to the left
                for m in range(h, 0, -1):
                    if h != m and not self.arc_mask[m - 1, h]:
                        # pruned; h == m signals last child
                        continue

                    gold_hm = m == h or instance.get_head(m) == h
                    arc_between = False

                    for s in range(m - 1, -2, -1):
                        if s == 0 or (s != -1 and not self.arc_mask[s - 1, h]):
                            # pruned out
                            # s = -1 signals leftmost child; 0 should be ignored
                            continue

                        gold_hs = s == -1 or instance.get_head(s) == h
                        if self.make_gold:
                            gold = 0
                            if gold_hm and gold_hs and not arc_between:
                                if gold_gh:
                                    gold = 1

                                arc_between = True
                            self.gold_parts.append(gold)

                        part = GrandSibling(h, m, g, s)
                        parts.append(part)

        self.part_lists[Target.GRANDSIBLINGS] = parts

    def create_arc_index(self):
        """
        Create a matrix such that cell (h, m) has the position of the given arc
        in the arc list of -1 if it doesn't exist.

        The matrix shape is (n, n), where n includes the dummy root.
        """
        # first, invert the arc_mask which is (m, h)
        mask = self.arc_mask.T
        mask = mask.astype(np.int)

        # add the root
        length = len(mask)
        root_col = np.zeros([length, 1], dtype=np.int)
        mask = np.concatenate([root_col, mask], axis=1)

        # replace 1's and 0's with their positions
        mask[mask == 0] = -1
        mask[mask == 1] = np.arange(np.sum(mask == 1))

        return mask

    def get_arc_indices(self):
        """
        Return a tuple with indices for heads and modifiers of valid arcs, such
        that they are ordered first by head and then by modifier.

        Modifier words are numbered from 1; 0 is reserved for the root.

        This ensures that all conversions from arc_mask to arcs will have the
        same ordering.

        :return: a tuple (heads, modifiers)
        """
        head_indices, modifier_indices = np.where(self.arc_mask.T)
        # modifiers in the mask consider the first real word as 0
        modifier_indices += 1

        return head_indices, modifier_indices

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
        if type_ == Target.HEADS:
            return 0

        if type_ == Target.RELATIONS:
            return self.num_arcs

        if type_ not in self.part_lists:
            return -1

        offset = self.num_arcs + self.num_labeled_arcs
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
