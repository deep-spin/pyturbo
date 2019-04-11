from collections import defaultdict
import ad3.factor_graph as fg
from ad3.extensions import PFactorTree, PFactorHeadAutomaton, \
    decode_matrix_tree, PFactorGrandparentHeadAutomaton
import numpy as np

from ..classifier.structured_decoder import StructuredDecoder
from .dependency_instance import DependencyInstance
from .dependency_parts import Arc, LabeledArc, NextSibling, DependencyParts, \
    Grandparent, GrandSibling
from .constants import Target


class PartStructure(object):
    """
    Class to store a list of dependency parts relative to a given head, as well
    as their scores and indices.
    """
    def __init__(self):
        self.parts = []
        self.scores = []
        self.indices = []

    def append(self, part, score, index):
        self.parts.append(part)
        self.scores.append(score)
        self.indices.append(index)

    def get_arcs(self, sort_decreasing=False, head='head', modifier='modifier',
                 sort_by_head=False):
        """
        Return a list of (h, m) tuples in the structure.
        """
        arc_set = set([(getattr(part, head), getattr(part, modifier))
                       for part in self.parts
                       if part.head != part.modifier])

        idx = 0 if sort_by_head else 1
        arc_list = sorted(arc_set, key=lambda arc: arc[idx],
                          reverse=sort_decreasing)
        return arc_list


class DependencyDecoder(StructuredDecoder):

    def __init__(self):
        StructuredDecoder.__init__(self)
        self.left_siblings = None
        self.right_siblings = None
        self.left_grandparents = None
        self.right_grandparents = None
        self.left_grandsiblings = None
        self.right_grandsiblings = None

        self.arc_indices = None
        self.additional_indices = None
        self.use_grandsiblings = None
        self.use_grandparents = None
        self.use_siblings = None

        # best_label_indices stores the position in the gold vector of the
        # best LabeledArc corresponding to each arc
        self.best_label_indices = None

    def decode(self, instance, parts, scores):
        """
        Decode the scores to the dependency parts under the necessary
        contraints, yielding a valid dependency tree.

        :param instance: DependencyInstance
        :param parts: a DependencyParts objects holding all the parts included
            in the scoring functions; usually arcs, siblings and grandparents
        :type parts: DependencyParts
        :param scores: dictionary mapping target names (such as arcs, relations,
            grandparents etc) to arrays of scores
        :return:
        """
        # this keeps track of the index of each part added to the
        # graph. The i-th part added to the graph will have its index to the
        # parts list stored in additiona_index[i] or arc_index[i]
        self.arc_indices = []
        self.additional_indices = []

        if parts.has_type(LabeledArc):
            self.decode_labels(parts, scores)

        # this keeps track of the arcs in the order they were added to the graph
        self.arcs = []

        graph = fg.PFactorGraph()
        variables = self.create_tree_factor(instance, parts, scores, graph)

        # self.use_siblings = parts.has_type(NextSibling)
        # self.use_grandparents = parts.has_type(Grandparent)
        # self.use_grandsiblings = parts.has_type(GrandSibling)
        #
        # self._index_parts_by_head(parts, instance, scores)
        #
        # if self.use_grandsiblings or \
        #         (self.use_siblings and self.use_grandparents):
        #     self.create_gp_head_automata(parts, graph, variables)
        # elif self.use_grandparents:
        #     self.create_grandparent_factors(parts, scores, graph, variables)
        # elif self.use_siblings:
        #     self.create_head_automata(parts, graph, variables)

        graph.set_eta_ad3(.05)
        graph.adapt_eta_ad3(True)
        graph.set_max_iterations_ad3(500)
        graph.set_residual_threshold_ad3(1e-3)

        value, posteriors, additional_posteriors, status = \
            graph.solve_lp_map_ad3()
        predicted_output = self.get_predicted_output(parts, posteriors,
                                                     additional_posteriors)

        return predicted_output

    def decode_labels(self, parts, scores):
        """
        Find the most likely label for each arc.

        :type parts: DependencyParts
        :param scores: list or array of scores for each part
        :return: the scores vector modified such that each unlabeled arc has
            its score increased by the best labeled score
        """
        relation_scores = scores[Target.RELATIONS]
        relation_scores[~parts.arc_mask] = -np.inf
        self.best_labels = relation_scores.argmax(-1)
        parts.best_labels = self.best_labels

        # # store the position of the best label for each arc
        # # best_label_indices[i] contains the position in the gold vector
        # # corresponding to the best LabeledArc part for Arc i
        # self.best_label_indices = []
        #
        # # copied scores
        # new_scores = np.array(scores)
        # offset_arc = parts.get_type_offset(Arc)
        # offset_label = parts.get_type_offset(LabeledArc)
        #
        # # we find which label has the highest score for each arc and sum
        # # the arc score with the label score
        # for i, arc in enumerate(parts.iterate_over_type(Arc), offset_arc):
        #     labeled_indices = parts.find_labeled_arc_indices(arc.head,
        #                                                      arc.modifier)
        #     labels = parts.find_arc_labels(arc.head, arc.modifier)
        #
        #     best_label_index = -1
        #     best_label = -1
        #     best_score = -np.inf
        #     for index, label in zip(labeled_indices, labels):
        #         index += offset_label
        #         score = scores[index]
        #
        #         if score > best_score:
        #             best_score = score
        #             best_label_index = index
        #             best_label = label
        #
        #     assert best_label_index >= 0
        #     self.best_label_indices.append(best_label_index)
        #     new_scores[i] += best_score
        #     parts.best_labels.append(best_label)
        #
        # return new_scores

    def copy_unlabeled_predictions(self, parts, predicted_output):
        """
        Copy the (unlabeled) arc prediction values found by the decoder to the
        labeled part with the highest score.

        :param predicted_output: array of predictions found by the decoder
        :type parts: DependencyParts
        """
        offset_arcs = parts.get_offset(Arc)[0]
        for i, arc in parts.iterate_over_type(Arc, return_index=True):
            label_index = i - offset_arcs
            label = self.best_label_indices[label_index]
            predicted_output[label] = predicted_output[i]

    def _index_parts_by_head(self, parts, instance, scores):
        """
        Create data structures mapping heads to lists of dependency parts,
        such as siblings or grandparents. The data strutctures are member
        variables.

        :type parts: DependencyParts
        :type instance: DependencyInstance
        :type scores: np.ndarray
        """
        n = len(instance)

        self.left_siblings = create_empty_structures(n)
        self.right_siblings = create_empty_structures(n)
        if self.use_siblings:
            _populate_structure_list(
                self.left_siblings, self.right_siblings, parts, scores,
                NextSibling)

        self.left_grandparents = create_empty_structures(n)
        self.right_grandparents = create_empty_structures(n)
        if self.use_grandparents:
            _populate_structure_list(
                self.left_grandparents, self.right_grandparents, parts, scores,
                Grandparent)

        self.left_grandsiblings = create_empty_structures(n)
        self.right_grandsiblings = create_empty_structures(n)
        if self.use_grandsiblings:
            _populate_structure_list(
                self.left_grandsiblings, self.right_grandsiblings, parts,
                scores, GrandSibling)

    def decode_matrix_tree(self, length, arc_index, parts, scores,
                           max_heads, threshold=0):
        """
        Decode the scores generated by the pruner constrained to be a
        non-projective tree.

        :param arc_index: a dictionary mapping [head] to another dictionary,
            which maps [modifier] to either a position in the arcs list or -1
            if the arc doesn't exist.
        :param parts: list of dependency parts
        :type parts: DependencyParts
        :param scores: 1d array with the score for each arc
        :return: a new parts object and a corresponding new gold scores. If
            gold_scores was None, the second element will be None.
        """
        # if there are scores for labeled parts, add the highest label score
        # of each arc to the arc score itself
        if parts.has_type(LabeledArc):
            scores = self.decode_labels(parts, scores)

        arcs = []
        offset = parts.get_type_offset(Arc)
        num_arcs = parts.get_num_type(Arc)
        for part in parts.iterate_over_type(Arc):
            arcs.append((part.head, part.modifier))
        arc_scores = scores[offset:offset + num_arcs]

        marginals, log_partition, entropy = decode_matrix_tree(
            length, arc_index, arcs, arc_scores)

        # now, we can treat the marginals as scores for each arc and run the
        # naive decoder algorithm. The resulting configurations ensures at
        # least one non-projective tree

        new_parts = self.decode_pruner_naive(parts, marginals,
                                             max_heads, threshold)

        return new_parts

    def decode_pruner_naive(self, parts, scores, max_heads,
                            threshold=-np.inf):
        """
        Decode the scores generated by the pruner without any constraint on the
        tree structure.

        :param parts: DependencyParts holding arcs
        :type parts: DependencyParts
        :param scores: the scores for each part in `parts`
        :param max_heads: maximum allowed head candidates for modifier
        :param threshold: prune parts with a scorer lower than this value. It
            should only be used when scores have been normalized to
            probabilities, since log potential values have an arbitrary scale.
        :return: a new parts object and a corresponding new gold scores. If
            gold_scores was None, the second element will be None.
        """
        candidate_heads = defaultdict(list)
        new_parts = DependencyParts()

        offset = parts.get_type_offset(Arc)
        offset_labeled = parts.get_type_offset(LabeledArc)
        gold_output = parts.get_gold_output()
        for i, arc in enumerate(parts.iterate_over_type(Arc), offset):
            score = scores[i]
            head = arc.head
            modifier = arc.modifier
            if score >= threshold:
                candidate_heads[modifier].append((head, score))

        for modifier in candidate_heads:
            heads_and_scores = candidate_heads[modifier]
            heads_and_scores.sort(key=lambda x: x[1], reverse=True)
            heads_and_scores = heads_and_scores[:max_heads]
            for head, score in heads_and_scores:
                if gold_output is not None:
                    # index is >= 0 because the arc was already in parts
                    index = parts.find_arc_index(head, modifier)
                    gold = gold_output[index + offset]
                else:
                    gold = None
                arc = Arc(head, modifier)
                new_parts.append(arc, gold)

                if not parts.has_type(LabeledArc):
                    continue

                # add labeled parts for (h, m)
                labels = parts.find_arc_labels(head, modifier)
                indices = parts.find_labeled_arc_indices(head, modifier)
                for index, label in zip(indices, labels):
                    index += offset_labeled

                    if gold_output is not None:
                        gold = gold_output[index]
                    else:
                        gold = None

                    part = LabeledArc(head, modifier, label)
                    new_parts.append(part, gold)

        return new_parts

    def _get_margin(self, parts):
        """
        Compute and return a margin vector to be used in the loss and a
        normalization term to be added to it.

        It only affects Arcs or LabeledArcs (in case the latter are used).

        :param parts: DependencyParts object
        :return: a margin array to be added to the model scores and a
            normalization constant
        """
        p = np.zeros_like(len(parts), dtype=np.float)
        if parts.labeled:
            # place the margin on LabeledArcs scores
            # their offset in the gold vector is immediately after Arcs
            offset = parts.num_arcs
            num_parts = parts.num_labeled_arcs
        else:
            # place the margin on Arc scores
            offset = 0
            num_parts = parts.num_arcs

        gold_values = parts.gold_arcs[offset:offset + num_parts]
        p[offset:offset + num_parts] = 0.5 - gold_values
        q = 0.5 * gold_values.sum()

        return p, q

    def get_predicted_output(self, parts, posteriors, additional_posteriors):
        """
        Create a numpy array with the predicted output for each part.

        :param parts: a DependencyParts object
        :param posteriors: list of posterior probabilities of the binary
            variables in the graph
        :param additional_posteriors: list of posterior probabilities of the
            variables introduced by factors
        :return: a numpy array with the same size as parts
        """
        predicted_output = np.zeros(len(parts), np.float)
        num_arcs = parts.num_arcs

        assert len(posteriors) == num_arcs
        assert len(additional_posteriors) == (len(parts) - num_arcs
                                              - parts.num_labeled_arcs)

        predicted_output[:num_arcs] = posteriors

        # if doing labeled parsing, set the score of the best label for each
        # arc to be the same as the score of the arc
        offset = num_arcs
        num_relations = parts.num_relations

        #TODO: set additional_posteriors

        for posterior, (m, h) in zip(posteriors, self.arcs):
            label = self.best_labels[m, h]
            predicted_output[offset + label] = posterior
            offset += num_relations

        return predicted_output

    def create_tree_factor(self, instance, parts, scores, graph):
        """
        Include factors to constrain the graph to a valid dependency tree.

        :type instance: DependencyInstance
        :type parts: DependencyParts
        :param scores: 1d np.array with model scores for each part
        :type graph: fg.PFactorGraph
        :return: a list of arc variables. The i-th variable corresponds to the
            i-th arc in parts.
        """
        # length is the number of tokens in the instance, including root
        length = len(instance)

        arc_scores = scores[Target.HEADS]
        if parts.labeled:
            # take the best relation score for each arc
            label_scores = scores[Target.RELATIONS]
            inds = np.expand_dims(self.best_labels, 2)
            best_label_scores = np.take_along_axis(label_scores, inds, 2)

            # make a copy of arc scores so the originals are unchanged
            arc_scores = arc_scores.copy()
            arc_scores += best_label_scores.squeeze(2)

        assert (length - 1) == len(arc_scores)
        assert length == len(arc_scores[0])

        tree_factor = PFactorTree()
        variables = []
        self.arcs = list(*zip(np.where(parts.arc_mask)))
        for m, h in self.arcs:
            arc_variable = graph.create_binary_variable()
            arc_variable.set_log_potential(arc_scores[m, h])
            variables.append(arc_variable)

        # for r, part in enumerate(parts.iterate_over_type(Arc), offset_arcs):
        #     arcs.append((part.head, part.modifier))
        #     arc_variable = graph.create_binary_variable()
        #     score = scores[r]
        #     arc_variable.set_log_potential(score)
        #     variables.append(arc_variable)
        #     self.arc_indices.append(r)

        # owned_by_graph makes the factor persist after calling this function
        # if left as False, the factor is garbage collected
        graph.declare_factor(tree_factor, variables, owned_by_graph=True)
        tree_factor.initialize(length, self.arcs)

        return variables

    def create_gp_head_automata(self, parts, graph, variables):
        """
        Include grandsibling factors (grandparent head automata) in the graph.

        :type parts: DependencyParts
        :param graph: the graph
        :param variables: list of binary variables denoting arcs
        """
        offset_arcs = parts.get_type_offset(Arc)
        # n is the number of tokens including the root
        n = len(self.left_siblings)

        def create_gp_head_automaton(structures, decreasing):
            """
            Create and sets the grandparent head automaton for either or right
            siblings and grandparents.
            """
            for head_structure in structures:
                siblings_structure = head_structure[0]
                sib_indices = siblings_structure.indices
                sib_tuples = [(p.head, p.modifier, p.sibling)
                              for p in siblings_structure.parts]

                grandparent_structure = head_structure[1]
                gp_indices = grandparent_structure.indices
                gp_tuples = [(p.grandparent, p.head, p.modifier)
                             for p in grandparent_structure.parts]

                # (g, h) arcs must always be in increasing order
                # we must include (g, h) even if there is no grandparent part
                # this happens when the only sibling part is with null siblings
                h = sib_tuples[0][0]
                incoming_arcs = []
                incoming_var_inds = []
                for g in range(n):
                    index = parts.find_arc_index(g, h)
                    if index >= 0:
                        incoming_var_inds.append(index)
                        incoming_arcs.append((g, h))

                # get arcs from siblings because we must include even outgoing
                # arcs that would make a cycle with the grandparent
                outgoing_arcs = siblings_structure.get_arcs(decreasing)

                if len(incoming_arcs) == 0:
                    # no grandparent structure; create simple head automaton
                    self._create_head_automata(
                        [siblings_structure], parts, graph, variables,
                        decreasing, offset_arcs)
                    continue

                outgoing_var_inds = [parts.find_arc_index(arc[0],
                                                          arc[1]) - offset_arcs
                                     for arc in outgoing_arcs]

                incoming_vars = [variables[i] for i in incoming_var_inds]
                outgoing_vars = [variables[i] for i in outgoing_var_inds]
                local_variables = incoming_vars + outgoing_vars

                indices = gp_indices + sib_indices
                scores = grandparent_structure.scores + \
                    siblings_structure.scores

                if len(head_structure) == 3:
                    grandsibling_structure = head_structure[2]
                    gsib_tuples = [(p.grandparent, p.head,
                                    p.modifier, p.sibling)
                                   for p in grandsibling_structure.parts]
                    scores += grandsibling_structure.scores
                    indices += grandsibling_structure.indices
                else:
                    gsib_tuples = None

                factor = PFactorGrandparentHeadAutomaton()
                graph.declare_factor(factor, local_variables,
                                     owned_by_graph=True)
                factor.initialize(incoming_arcs, outgoing_arcs, gp_tuples,
                                  sib_tuples, gsib_tuples)
                factor.set_additional_log_potentials(scores)
                self.additional_indices.extend(indices)

        if self.use_grandsiblings:
            left_structures = zip(self.left_siblings,
                                  self.left_grandparents,
                                  self.left_grandsiblings)
            right_structures = zip(self.right_siblings,
                                   self.right_grandparents,
                                   self.right_grandsiblings)
        else:
            left_structures = zip(self.left_siblings,
                                  self.left_grandparents)
            right_structures = zip(self.right_siblings,
                                   self.right_grandparents)

        create_gp_head_automaton(left_structures, decreasing=True)
        create_gp_head_automaton(right_structures, decreasing=False)

    def create_grandparent_factors(self, parts, scores, graph, variables):
        """
        Include grandparent factors for constraining grandparents in the graph.

        :param parts: DependencyParts
        :param scores: np.array
        :param graph: the graph
        :param variables: list of binary variables denoting arcs
        """
        offset_arcs = parts.get_type_offset(Arc)
        offset_gp = parts.get_type_offset(Grandparent)

        for i, part in enumerate(parts.iterate_over_type(Grandparent),
                                 offset_gp):
            head = part.head
            modifier = part.modifier
            grandparent = part.grandparent

            index_hm = parts.find_arc_index(head, modifier) - offset_arcs
            index_gh = parts.find_arc_index(grandparent, head) - offset_arcs

            var_hm = variables[index_hm]
            var_gh = variables[index_gh]

            score = scores[i]
            graph.create_factor_pair([var_hm, var_gh], score)
            self.additional_indices.append(i)

    def _create_head_automata(self, structures, parts, graph, variables,
                              decreasing, offset_arcs=0):
        """
        Creates and sets the head automaton factors for either left or
        right siblings.

        :param structures: a list of PartStructure objects containing
            next sibling parts
        :param parts: DependencyParts
        :param graph: the AD3 graph
        :param variables: list of variables constrained by the automata
        :param decreasing: whether to sort modifiers in decreasing order.
            It should be True for left hand side automata and False for
            right hand side.
        """
        for head_structure in structures:
            if len(head_structure.parts) == 0:
                continue

            indices = head_structure.indices
            arcs = head_structure.get_arcs(decreasing)
            var_inds = [parts.find_arc_index(arc[0], arc[1]) - offset_arcs
                        for arc in arcs]
            local_variables = [variables[i] for i in var_inds]
            siblings = [(p.head, p.modifier, p.sibling)
                        for p in head_structure.parts]

            # important: first declare the factor in the graph,
            # then initialize
            factor = PFactorHeadAutomaton()
            graph.declare_factor(factor, local_variables,
                                 owned_by_graph=True)
            factor.initialize(arcs, siblings, validate=False)
            factor.set_additional_log_potentials(head_structure.scores)

            self.additional_indices.extend(indices)

    def create_head_automata(self, parts, graph, variables):
        """
        Include head automata for constraining consecutive siblings in the
        graph.

        :type parts: DependencyParts
        :param graph: the graph
        :param variables: list of binary variables denoting arcs
        """
        # needed to map indices in parts to indices in variables
        offset_arcs = parts.get_type_offset(Arc)
        self._create_head_automata(
            self.left_siblings, parts, graph, variables, decreasing=True,
            offset_arcs=offset_arcs)
        self._create_head_automata(
            self.right_siblings, parts, graph, variables, decreasing=False,
            offset_arcs=offset_arcs)


def create_empty_lists(n):
    """
    Create a list with n empty lists
    """
    return [[] for _ in range(n)]


def create_empty_structures(n):
    """
    Create a list with n empty PartStructures
    """
    return [PartStructure() for _ in range(n)]


def _populate_structure_list(left_list, right_list, parts, scores,
                             type_):
    """
    Populate structure lists left_list and right_list with the dependency
    parts that appear to the left and right of each head.

    :param left_list: a list with empty PartStructure objects, one for each
        head. It will be filled with the structures occurring left to each
        head.
    :param parts: a DependencyParts object
    :param scores: a list or array of scores
    :param right_list: same as above, for the right hand side.
    :param type_: a type of dependency part
    """
    offset = parts.get_type_offset(type_)
    for i, part in enumerate(parts.iterate_over_type(type_), offset):

        # make this check because modifier == head has a special meaning for
        # sibling parts
        if isinstance(part, (NextSibling, GrandSibling)):
            is_right = part.sibling > part.head
        else:
            is_right = part.modifier > part.head

        if is_right:
            # right sibling
            right_list[part.head].append(part, scores[i], i)
        else:
            # left sibling
            left_list[part.head].append(part, scores[i], i)


def make_score_matrix(length, arcs, scores):
    """
    Makes a score matrix from an array of scores ordered in the same way as a
    list of DependencyPartArcs. Positions [m, h] corresponding to non-existing
    arcs have score of -inf.

    :param length: length of the sentence, including the root pseudo-token
    :param arcs: list of candidate Arc parts
    :param scores: array with score of each arc (ordered in the same way as
        arcs)
    :return: a 2d numpy array
    """
    score_matrix = np.full([length, length], -np.inf, np.float32)
    for arc, score in zip(arcs, scores):
        h = arc.head
        m = arc.modifier
        score_matrix[m, h] = score

    return score_matrix


def chu_liu_edmonds(score_matrix):
    """
    Run the Chu-Liu-Edmonds' algorithm to find the maximum spanning tree.

    :param score_matrix: a matrix such that cell [m, h] has the score for the
        arc (h, m).
    :return: an array heads, such that heads[m] contains the head of token m.
        The root is in position 0 and has head -1.
    """
    # avoid loops to self
    np.fill_diagonal(score_matrix, -np.inf)

    # no token points to the root but the root itself
    score_matrix[0] = -np.inf
    score_matrix[0, 0] = 0

    # pick the highest score head for each modifier and look for cycles
    heads = score_matrix.argmax(1)
    cycles = tarjan(heads)

    if cycles:
        # t = len(tree); c = len(cycle); n = len(noncycle)
        # locations of cycle; (t) in [0,1]
        cycle = cycles.pop()
        # indices of cycle in original tree; (c) in t
        cycle_locs = np.where(cycle)[0]
        # heads of cycle in original tree; (c) in t
        cycle_subtree = heads[cycle]
        # scores of cycle in original tree; (c) in R
        cycle_scores = score_matrix[cycle, cycle_subtree]
        # total score of cycle; () in R
        cycle_score = cycle_scores.sum()

        # locations of noncycle; (t) in [0,1]
        noncycle = np.logical_not(cycle)
        # indices of noncycle in original tree; (n) in t
        noncycle_locs = np.where(noncycle)[0]

        # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
        metanode_head_scores = score_matrix[cycle][:, noncycle] - \
            cycle_scores[:, None] + cycle_score
        # scores of cycle's potential dependents; (n x c) in R
        metanode_dep_scores = score_matrix[noncycle][:,cycle]
        # best noncycle head for each cycle dependent; (n) in c
        metanode_heads = np.argmax(metanode_head_scores, axis=0)
        # best cycle head for each noncycle dependent; (n) in c
        metanode_deps = np.argmax(metanode_dep_scores, axis=1)

        # scores of noncycle graph; (n x n) in R
        subscores = score_matrix[noncycle][:,noncycle]
        # pad to contracted graph; (n+1 x n+1) in R
        subscores = np.pad(subscores, ((0, 1), (0, 1)), 'constant')
        # set the contracted graph scores of cycle's potential heads;
        # (c x n)[:, (n) in n] in R -> (n) in R
        subscores[-1, :-1] = metanode_head_scores[metanode_heads,
                                                  np.arange(len(noncycle_locs))]
        # set the contracted graph scores of cycle's potential dependents;
        # (n x c)[(n) in n] in R-> (n) in R
        subscores[:-1, -1] = metanode_dep_scores[np.arange(len(noncycle_locs)),
                                                 metanode_deps]

        # MST with contraction; (n+1) in n+1
        contracted_tree = chu_liu_edmonds(subscores)
        # head of the cycle; () in n
        cycle_head = contracted_tree[-1]
        # fixed tree: (n) in n+1
        contracted_tree = contracted_tree[:-1]
        # initialize new tree; (t) in 0
        new_heads = -np.ones_like(heads)

        # fixed tree with no heads coming from the cycle: (n) in [0,1]
        contracted_subtree = contracted_tree < len(contracted_tree)
        # add the nodes to the new tree (t)
        # [(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
        new_heads[noncycle_locs[contracted_subtree]] = \
            noncycle_locs[contracted_tree[contracted_subtree]]

        # fixed tree with heads coming from the cycle: (n) in [0,1]
        contracted_subtree = np.logical_not(contracted_subtree)
        # add the nodes to the tree (t)
        # [(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
        new_heads[noncycle_locs[contracted_subtree]] = \
            cycle_locs[metanode_deps[contracted_subtree]]
        # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
        new_heads[cycle_locs] = heads[cycle_locs]
        # root of the cycle; (n)[() in n] in c = () in c
        cycle_root = metanode_heads[cycle_head]
        # add the root of the cycle to the new tree;
        # (t)[(c)[() in c] in t] = (c)[() in c]
        new_heads[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]

        heads = new_heads

    return heads


def tarjan(heads):
    """Tarjan's algorithm for finding cycles"""
    indices = -np.ones_like(heads)
    lowlinks = -np.ones_like(heads)
    onstack = np.zeros_like(heads, dtype=bool)
    stack = []
    _index = [0]
    cycles = []

    def strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True
        dependents = np.where(np.equal(heads, i))[0]
        for j in dependents:
            if indices[j] == -1:
                strong_connect(j)
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            elif onstack[j]:
                lowlinks[i] = min(lowlinks[i], indices[j])

        # There's a cycle!
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)
        return

    for i in range(len(heads)):
        if indices[i] == -1:
            strong_connect(i)

    return cycles
