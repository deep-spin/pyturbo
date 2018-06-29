from classifier.structured_decoder import StructuredDecoder
from parser.dependency_instance import DependencyInstance
from parser.dependency_parts import DependencyPartArc, DependencyPartLabeledArc
import ad3.factor_graph as fg
from ad3.extensions import PFactorTree
import numpy as np

class DependencyDecoder(StructuredDecoder):
    def __init__(self):
        StructuredDecoder.__init__(self)

    def decode(self, instance, parts, scores):
        length = len(instance)
        offset_arcs, num_arcs = parts.get_offset(DependencyPartArc)

        predicted_output = np.zeros(len(parts))
        graph = fg.PFactorGraph()
        tree_factor = PFactorTree()
        arc_indices = []
        variables = []
        for r in range(offset_arcs, offset_arcs + num_arcs):
            arc_indices.append((parts[r].head, parts[r].modifier))
            arc = graph.create_binary_variable()
            arc.set_log_potential(scores[r])
            variables.append(arc)
        graph.declare_factor(tree_factor, variables)
        tree_factor.initialize(length, arc_indices)
        graph.set_eta_ad3(.05)
        graph.adapt_eta_ad3(True)
        graph.set_max_iterations_ad3(500)
        graph.set_residual_threshold_ad3(1e-3)

        value, posteriors, additional_posteriors, status = \
            graph.solve_lp_map_ad3()
        #print(status)
        #print(value)
        #predicted_arcs = \
        #    [index for index, posterior in zip(arc_indices, posteriors)
        #     if posterior > 0.1]
        #print(predicted_arcs)
        predicted_output = posteriors
        return predicted_output
