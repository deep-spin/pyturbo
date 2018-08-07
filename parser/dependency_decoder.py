from classifier.structured_decoder import StructuredDecoder
from parser.dependency_instance import DependencyInstance
from parser.dependency_parts import DependencyPartArc, \
    DependencyPartLabeledArc, DependencyPartConsecutiveSibling, \
    DependencyParts
import ad3.factor_graph as fg
from ad3.extensions import PFactorTree, PFactorHeadAutomaton
import numpy as np


class DependencyDecoder(StructuredDecoder):
    def __init__(self):
        StructuredDecoder.__init__(self)

    def decode(self, instance, parts, scores):
        """
        Decode the scores to the dependency parts under the necessary
        contraints, yielding a valid dependency tree.

        :param instance: DependencyInstance
        :type parts: DependencyParts
        :param scores: array or tensor with scores for each part
        :return:
        """
        length = len(instance)
        offset_arcs, num_arcs = parts.get_offset(DependencyPartArc)

        predicted_output = np.zeros(len(parts))
        graph = fg.PFactorGraph()
        tree_factor = PFactorTree()
        arc_indices = []
        variables = []
        for r in range(offset_arcs, offset_arcs + num_arcs):
            arc_indices.append((parts[r].head, parts[r].modifier))
            arc_variable = graph.create_binary_variable()
            arc_variable.set_log_potential(scores[r])
            variables.append(arc_variable)

        graph.declare_factor(tree_factor, variables)
        tree_factor.initialize(length, arc_indices)
        graph.set_eta_ad3(.05)
        graph.adapt_eta_ad3(True)
        graph.set_max_iterations_ad3(500)
        graph.set_residual_threshold_ad3(1e-3)

        value, posteriors, additional_posteriors, status = \
            graph.solve_lp_map_ad3()

        predicted_output = posteriors
        return predicted_output

    def set_next_sibling_factors(self, instance, parts, scores, graph,
                                 variables):
        """
        Include head automata for constraining consecutive siblings in the
        graph.

        :type parts: DependencyParts
        :type instance: DependencyInstance
        :type scores: np.array
        :param graph: the graph
        :param variables: list of binary variables denoting arcs
        """
        # needed to map indices in parts to indices in variables
        offset_arcs, _ = parts.get_offset(DependencyPartArc)

        def add_variables(local_variables, h, m):
            """
            Add the AD3 binary variable representing the arc from h to m if
            it exists in parts.
            """
            parts_index = parts.find_arc_index(h, m)
            if parts_index < 0:
                return

            var_index = parts_index - offset_arcs
            arc_variable = variables[var_index]
            local_variables.append(arc_variable)

        n = len(instance)
        offset, num_siblings = parts.get_offset(
            DependencyPartConsecutiveSibling)

        # loop through all parts and organize them according to the head
        left_siblings = create_empty_lists(n)
        right_siblings = create_empty_lists(n)
        left_scores = create_empty_lists(n)
        right_scores = create_empty_lists(n)

        for r in range(offset, offset + num_siblings):
            part = parts[r]
            h = part.head
            s = part.sibling

            if s > h:
                # right sibling
                right_siblings[h].append(part)
                right_scores[h].append(scores[offset + r])
            else:
                # left sibling
                left_siblings[h].append(part)
                left_scores[h].append(scores[offset + r])

        # create right and left automata for each head
        for h in range(n):
            # these are the variables constrained by the factor
            local_variables = []

            # left hand side
            for m in range(h - 1, 0, -1):
                add_variables(local_variables, h, m)

            # important: first declare the factor in the graph, then initialize
            factor = PFactorHeadAutomaton()
            graph.declare_factor(factor, local_variables)
            factor.initialize(h, left_siblings[h])
            factor.set_additional_log_potentials(left_scores[h])

            # right hand side
            local_variables = []

            for m in range(h + 1, n):
                add_variables(local_variables, h, m)

            factor = PFactorHeadAutomaton()
            graph.declare_factor(factor, local_variables)
            factor.initialize(n - h, right_siblings[h])
            factor.set_additional_log_potentials(right_scores[h])


def create_empty_lists(n):
    """
    Create a list with n empty lists
    """
    return [[] for _ in range(n)]
