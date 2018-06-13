import ad3.factor_graph as fg
import numpy as np

graph = fg.PFactorGraph()
tree_factor = fg.PFactorTree()
arc_indices = []
variables = []
for m in range(1, 5):
    for h in range(5):
        if m == h: continue
        arc_indices.append((h, m))
        arc = graph.create_binary_variable()
        arc.set_log_potential(np.random.rand(1))
        variables.append(arc)
graph.declare_factor(tree_factor, variables)
tree_factor.initialize(5, arc_indices)
graph.set_eta_ad3(.1)
graph.adapt_eta_ad3(True)
graph.set_max_iterations_ad3(1000)
value, marginals, edge_marginals, status = graph.solve_lp_map_ad3()
print(status)
print(value)
predicted_arcs = [index for index, posterior in zip(arc_indices, marginals)
                  if posterior > 0.1]
print(predicted_arcs)
