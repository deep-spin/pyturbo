from collections import defaultdict
import ad3.factor_graph as fg
from ad3.extensions import PFactorTree, PFactorHeadAutomaton, decode_matrix_tree
import numpy as np

from classifier.structured_decoder import StructuredDecoder
from parser.dependency_instance import DependencyInstance
from parser.dependency_parts import DependencyPartArc, \
    DependencyPartLabeledArc, DependencyPartNextSibling, \
    DependencyParts, DependencyPartGrandparent, DependencyPartGrandSibling


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

    def get_arcs(self, sort_decreasing=False):
        """
        Return a list of (h, m) tuples in the structure (only considers the head
        and modifier compoinent of each part).
        """
        arc_set = set([(part.head, part.modifier)
                       for part in self.parts
                       if part.head != part.modifier])
        arc_list = sorted(arc_set, key=lambda arc: arc[1],
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

        self.additional_indices = None
        self.use_grandsiblings = None
        self.use_grandparents = None
        self.use_siblings = None

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
        graph = fg.PFactorGraph()
        variables = self.create_tree_factor(instance, parts, scores, graph)

        # this keeps track of the index of each additional part added to the
        # graph. The i-th part added to the graph will have its index to the
        # parts list stored in additiona_index[i]
        self.additional_indices = []

        self.use_siblings = parts.has_type(DependencyPartNextSibling)
        self.use_grandparents = parts.has_type(DependencyPartGrandparent)
        self.use_grandsiblings = parts.has_type(DependencyPartGrandSibling)

        self._index_parts_by_head(parts, instance, scores)

        if self.use_grandparents:
            self.create_grandparent_factors(parts, scores, graph, variables)
        if self.use_siblings:
            self.create_next_sibling_factors(parts, graph, variables)

        graph.set_eta_ad3(.05)
        graph.adapt_eta_ad3(True)
        graph.set_max_iterations_ad3(500)
        graph.set_residual_threshold_ad3(1e-3)

        value, posteriors, additional_posteriors, status = \
            graph.solve_lp_map_ad3()
        predicted_output = self.get_predicted_output(parts, posteriors,
                                                     additional_posteriors)

        return predicted_output

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
                DependencyPartNextSibling)

        if self.use_grandparents:
            self.left_grandparents = create_empty_structures(n)
            self.right_grandparents = create_empty_structures(n)
            _populate_structure_list(
                self.left_grandparents, self.right_grandparents, parts, scores,
                DependencyPartGrandparent)

        if self.use_grandsiblings:
            self.left_grandsiblings = create_empty_structures(n)
            self.right_grandsiblings = create_empty_structures(n)
            _populate_structure_list(
                self.left_grandsiblings, self.right_grandsiblings, parts,
                scores, DependencyPartGrandSibling)

    def decode_matrix_tree(self, length, arc_index, parts, scores, gold_output,
                           max_heads, threshold=0):
        """
        Decode the scores generated by the pruner constrained to be a
        non-projective tree.

        :param arc_index: a 2d array mapping pairs (head, modifier) to a
            position in the arcs list; -1 if the combination doesn't exist.
        :param parts: list of DependencyPartArc objects
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
            sorted(heads_and_scores, key=lambda x: x[1], reverse=True)
            heads_and_scores = heads_and_scores[:max_heads]
            for head, score in heads_and_scores:
                new_parts.append(DependencyPartArc(head, modifier))
                if gold_output is not None:
                    # index is >= 0 because the arc was already in parts
                    index = parts.find_arc_index(head, modifier)
                    new_gold.append(gold_output[index])

        new_parts.set_offset(DependencyPartArc, 0, len(new_parts))

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
        posteriors = np.array(posteriors)
        predicted_output = np.zeros(len(parts), posteriors.dtype)

        # first, copy the posteriors to the appropriate place
        offset_arcs, num_arcs = parts.get_offset(DependencyPartArc)
        predicted_output[offset_arcs:offset_arcs + num_arcs] = posteriors

        for value, index in zip(additional_posteriors, self.additional_indices):
            predicted_output[index] = value

        # copy the posteriors and additional to predicted_output in the same
        # order they were created

        #
        #
        #     offset_type, num_this_type = parts.get_offset(type_)
        #
        #     from_posteriors = offset_type - num_arcs
        #     until_posteriors = from_posteriors + num_this_type
        #
        #     predicted_output[offset_type:offset_type + num_this_type] = \
        #         additional_posteriors[from_posteriors:until_posteriors]
        #
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
        offset_arcs, num_arcs = parts.get_offset(DependencyPartArc)

        tree_factor = PFactorTree()
        arc_indices = []
        variables = []
        for r in range(offset_arcs, offset_arcs + num_arcs):
            arc_indices.append((parts[r].head, parts[r].modifier))
            arc_variable = graph.create_binary_variable()
            arc_variable.set_log_potential(scores[r])
            variables.append(arc_variable)

        # owned_by_graph makes the factor persist after calling this function
        # if left as False, the factor is garbage collected
        graph.declare_factor(tree_factor, variables, owned_by_graph=True)
        tree_factor.initialize(length, arc_indices)

        return variables

    def create_grandsibling_factors(self, instance, parts, scores, graph,
                                    variables):
        """
        Include grandsibling factors (grandparent head automata) in the graph.

        :type parts: DependencyParts
        :param scores: np.array
        :param graph: the graph
        :param variables: list of binary variables denoting arcs
        """
        # TODO: try to avoid repeitition with the head_automaton function
        offset_arcs, _ = parts.get_offset(DependencyPartArc)

        n = len(instance)
        offset_gsib, num_gsib = parts.get_offset(
            DependencyPartNextSibling)

        # loop through all parts and organize them according to the head
        left_siblings = create_empty_lists(n)
        right_siblings = create_empty_lists(n)
        left_scores = create_empty_lists(n)
        right_scores = create_empty_lists(n)

        for r, part in parts.iterate_over_type(DependencyPartGrandSibling,
                                               True):
            h = part.head
            m = part.modifier
            g = part.grandparent
            s = part.sibling

            if s > h:
                # right sibling
                right_siblings[h].append((g, h, m, s))
                right_scores[h].append(scores[r])
            else:
                # left sibling
                left_siblings[h].append((g, h, m, s))
                left_scores[h].append(scores[r])

        # create right and left automata for each head
        for h in range(n):

            # left hand side
            # these are the variables constrained by the factor and their arcs
            local_variables = []

            # tuples (g, h)
            incoming_arcs = []

            # tuples (h, m)
            outgoing_arcs = []

            for part in left_siblings[h]:
                h = part.head
                m = part.modifier
                g = part.grandparent
                s = part.sibling

                index_gh = parts.find_arc_index(g, h)
                var_index_gh = index_gh - offset_arcs

                incoming_arcs.append((g, h))
                outgoing_arcs.append((h, m))

    def create_grandparent_factors(self, parts, scores, graph, variables):
        """
        Include grandparent factors for constraining grandparents in the graph.

        :param parts: DependencyParts
        :param scores: np.array
        :param graph: the graph
        :param variables: list of binary variables denoting arcs
        """
        offset_arcs, num_arcs = parts.get_offset(DependencyPartArc)

        for i, part in parts.iterate_over_type(DependencyPartGrandparent,
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

    def create_next_sibling_factors(self, parts, graph, variables):
        """
        Include head automata for constraining consecutive siblings in the
        graph.

        :type parts: DependencyParts
        :param graph: the graph
        :param variables: list of binary variables denoting arcs
        """
        # needed to map indices in parts to indices in variables
        offset_arcs, _ = parts.get_offset(DependencyPartArc)

        def create_head_automaton(structures, decreasing):
            """
            Creates and sets the head automaton factors for either left or
            right siblings.
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

        create_head_automaton(self.left_siblings, decreasing=True)
        create_head_automaton(self.right_siblings, decreasing=False)


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
        if isinstance(part, DependencyPartNextSibling):
            is_right = part.sibling > part.head
        else:
            is_right = part.modifier > part.head

        if is_right:
            # right sibling
            right_list[part.head].append(part, scores[i], i)
        else:
            # left sibling
            left_list[part.head].append(part, scores[i], i)
