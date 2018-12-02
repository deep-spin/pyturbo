from collections import defaultdict
import ad3.factor_graph as fg
from ad3.extensions import PFactorTree, PFactorHeadAutomaton, \
    decode_matrix_tree, PFactorGrandparentHeadAutomaton
import numpy as np

from ..classifier.structured_decoder import StructuredDecoder
from .dependency_instance import DependencyInstance
from .dependency_parts import Arc, LabeledArc, NextSibling, DependencyParts, \
    Grandparent, GrandSibling


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

        self.best_label_indices = None

    def decode(self, instance, parts, scores):
        """
        Decode the scores to the dependency parts under the necessary
        contraints, yielding a valid dependency tree.

        :param instance: DependencyInstance
        :param parts: a DependencyParts objects holding all the parts included
            in the scoring functions; usually arcs, siblings and grandparents
        :type parts: DependencyParts
        :param scores: array or tensor with scores for each part, produced by
            the model. It should be a 1d array.
        :return:
        """
        # this keeps track of the index of each part added to the
        # graph. The i-th part added to the graph will have its index to the
        # parts list stored in additiona_index[i] or arc_index[i]
        self.arc_indices = []
        self.additional_indices = []

        if parts.has_type(LabeledArc):
            scores = self.decode_labels(parts, scores)

        graph = fg.PFactorGraph()
        variables = self.create_tree_factor(instance, parts, scores, graph)

        self.use_siblings = parts.has_type(NextSibling)
        self.use_grandparents = parts.has_type(Grandparent)
        self.use_grandsiblings = parts.has_type(GrandSibling)

        self._index_parts_by_head(parts, instance, scores)

        if self.use_siblings and self.use_grandparents:
            self.create_gp_head_automata(parts, graph, variables)
        elif self.use_grandparents:
            self.create_grandparent_factors(parts, scores, graph, variables)
        elif self.use_siblings:
            self.create_head_automata(parts, graph, variables)

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
        # store the position of the best label for each arc
        self.best_label_indices = []

        # copied scores
        new_scores = np.array(scores)

        for i, arc in parts.iterate_over_type(Arc, return_index=True):
            labeled_indices = parts.find_labeled_arc_indices(arc.head,
                                                             arc.modifier)
            labels = parts.find_arc_labels(arc.head, arc.modifier)

            best_label_index = -1
            best_label = -1
            best_score = -np.inf
            for index, label in zip(labeled_indices, labels):
                score = scores[index]

                if score > best_score:
                    best_score = score
                    best_label_index = index
                    best_label = label

            assert best_label_index >= 0
            self.best_label_indices.append(best_label_index)
            new_scores[i] += best_score
            parts.best_labels.append(best_label)

        return new_scores

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

        if self.use_siblings:
            self.left_siblings = create_empty_structures(n)
            self.right_siblings = create_empty_structures(n)
            _populate_structure_list(
                self.left_siblings, self.right_siblings, parts, scores,
                NextSibling)

        if self.use_grandparents:
            self.left_grandparents = create_empty_structures(n)
            self.right_grandparents = create_empty_structures(n)
            _populate_structure_list(
                self.left_grandparents, self.right_grandparents, parts, scores,
                Grandparent)

        if self.use_grandsiblings:
            self.left_grandsiblings = create_empty_structures(n)
            self.right_grandsiblings = create_empty_structures(n)
            _populate_structure_list(
                self.left_grandsiblings, self.right_grandsiblings, parts,
                scores, GrandSibling)

    def decode_matrix_tree(self, length, arc_index, parts, scores, gold_output,
                           max_heads, threshold=0):
        """
        Decode the scores generated by the pruner constrained to be a
        non-projective tree.

        :param arc_index: a 2d array mapping pairs (head, modifier) to a
            position in the arcs list; -1 if the combination doesn't exist.
        :param parts: list of Arc objects
        :param scores: 1d array with the score for each arc
        :return: a new parts object and a corresponding new gold scores. If
            gold_scores was None, the second element will be None.
        """
        arcs = []
        for part in parts:
            arcs.append((part.head, part.modifier))

        marginals, log_partition, entropy = decode_matrix_tree(
            length, arc_index, arcs, scores)

        # now, we can treat the marginals as scores for each arc and run the
        # naive decoder algorithm. The resulting configurations ensures at
        # least one non-projective tree

        new_parts, new_gold = self.decode_pruner_naive(parts, marginals,
                                                       gold_output, max_heads,
                                                       threshold)

        return new_parts, new_gold

    def decode_pruner_naive(self, parts, scores, gold_output, max_heads,
                            threshold=-np.inf):
        """
        Decode the scores generated by the pruner without any constraint on the
        tree structure.

        :param parts: DependencyParts holding arcs
        :param scores: the scores for each part in `parts`
        :param gold_output: either None or gold binary labels for each part
        :param max_heads: maximum allowed head candidates for modifier
        :param threshold: prune parts with a scorer lower than this value. It
            should only be used when scores have been normalized to
            probabilities, since log potential values have an arbitrary scale.
        :return: a new parts object and a corresponding new gold scores. If
            gold_scores was None, the second element will be None.
        """
        candidate_heads = defaultdict(list)
        new_parts = DependencyParts()
        new_gold = None if gold_output is None else []

        for part, score in zip(parts, scores):
            head = part.head
            modifier = part.modifier
            if score >= threshold:
                candidate_heads[modifier].append((head, score))

        for modifier in candidate_heads:
            heads_and_scores = candidate_heads[modifier]
            heads_and_scores.sort(key=lambda x: x[1], reverse=True)
            heads_and_scores = heads_and_scores[:max_heads]
            for head, score in heads_and_scores:
                new_parts.append(Arc(head, modifier))
                if gold_output is not None:
                    # index is >= 0 because the arc was already in parts
                    index = parts.find_arc_index(head, modifier)
                    new_gold.append(gold_output[index])

        new_parts.set_offset(Arc, 0, len(new_parts))

        return new_parts, new_gold

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

        assert len(posteriors) == len(self.arc_indices)
        assert len(additional_posteriors) == len(self.additional_indices)

        all_posteriors = posteriors + additional_posteriors
        all_indices = self.arc_indices + self.additional_indices
        for value, index in zip(all_posteriors, all_indices):
            predicted_output[index] = value

        # Copy the (unlabeled) arc prediction values found to the
        # labeled part with the highest score.
        if self.best_label_indices:
            offset_arcs = parts.get_offset(Arc)[0]
            for i, arc in parts.iterate_over_type(Arc, return_index=True):
                label_index = i - offset_arcs
                label = self.best_label_indices[label_index]
                predicted_output[label] = predicted_output[i]

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

        tree_factor = PFactorTree()
        arcs = []
        variables = []
        offset_arcs = parts.get_offset(Arc)[0]
        for r, part in parts.iterate_over_type(Arc, True):
            arcs.append((part.head, part.modifier))
            arc_variable = graph.create_binary_variable()

            score = scores[r]
            if self.best_label_indices:
                index = self.best_label_indices[r - offset_arcs]
                score += scores[index]
            arc_variable.set_log_potential(score)

            variables.append(arc_variable)
            self.arc_indices.append(r)

        # owned_by_graph makes the factor persist after calling this function
        # if left as False, the factor is garbage collected
        graph.declare_factor(tree_factor, variables, owned_by_graph=True)
        tree_factor.initialize(length, arcs)

        return variables

    def create_gp_head_automata(self, parts, graph, variables):
        """
        Include grandsibling factors (grandparent head automata) in the graph.

        :type parts: DependencyParts
        :param graph: the graph
        :param variables: list of binary variables denoting arcs
        """
        offset_arcs, _ = parts.get_offset(Arc)

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
                incoming_arcs = grandparent_structure.get_arcs(
                    sort_decreasing=False, head='grandparent', modifier='head',
                    sort_by_head=True)

                # get arcs from siblings because we must include even outgoing
                # arcs that would make a cycle with the grandparent
                outgoing_arcs = siblings_structure.get_arcs(decreasing)

                if len(incoming_arcs) == 0 or len(outgoing_arcs) == 0:
                    # no grandparent structure; create simple head automaton
                    # TODO: maybe change the function to receive a single
                    # structure instead of a list?
                    self._create_head_automata(
                        [siblings_structure], parts, graph, variables,
                        decreasing, offset_arcs)
                    continue

                in_var_inds = [parts.find_arc_index(arc[0],
                                                    arc[1]) - offset_arcs
                               for arc in incoming_arcs]
                out_var_inds = [parts.find_arc_index(arc[0],
                                                     arc[1]) - offset_arcs
                                for arc in outgoing_arcs]

                incoming_vars = [variables[i] for i in in_var_inds]
                outgoing_vars = [variables[i] for i in out_var_inds]
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
        offset_arcs, num_arcs = parts.get_offset(Arc)

        for i, part in parts.iterate_over_type(Grandparent,
                                               return_index=True):
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
        offset_arcs, _ = parts.get_offset(Arc)
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
    for i, part in parts.iterate_over_type(type_, return_index=True):

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
