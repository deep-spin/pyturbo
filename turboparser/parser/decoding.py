from collections import defaultdict
import ad3.factor_graph as fg
from ad3.extensions import PFactorTree, PFactorHeadAutomaton, \
    PFactorGrandparentHeadAutomaton
import numpy as np
import os
from joblib import Parallel, delayed
from scipy.special import logsumexp, softmax

from ..classifier.utils import get_logger
from .dependency_instance import DependencyInstance
from ..classifier.instance import InstanceData
from .dependency_parts import NextSibling, DependencyParts, GrandSibling
from .constants import Target


logger = get_logger()
os.environ['LOKY_PICKLER'] = 'pickle'


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


def decode_predictions(predicted_parts=None, parts=None, head_score_matrix=None,
                       label_matrix=None, single_root=True):
    """
    Decode the predicted heads and labels over the output of the AD3 decoder
    or just score matrices.

    This function takes care of the cases when the variable assignments by
    the decoder does not produce a valid tree running the Chu-Liu-Edmonds
    algorithm.

    :param predicted_parts: indicator array of predicted dependency parts
        (with values between 0 and 1). Either this or head_scores and
        label_matrix must be provided.
    :param parts: the dependency parts. Only needed if predicted_parts is given.
    :param single_root: whether to enforce a single sentence root
    :param head_score_matrix: matrix of scores with shape (modifiers, heads)
        there must be one head more than modifiers -- the root
    :param label_matrix: matrix with shape (modifiers, heads) containing the
        most likely label for each arc. Ignored if predicted_parts is given.
    :return: a tuple (pred_heads, pred_labels)
        The first is an array such that position heads[m] contains the head
        for token m; it starts from the first actual word, not the root.
        If the model is not trained for predicting labels, the second item
        is None.
    """
    if head_score_matrix is None:
        length = len(parts.arc_mask)
        arc_scores = predicted_parts[:parts.num_arcs]
        score_matrix = make_score_matrix(length, parts.arc_mask, arc_scores)
    else:
        # TODO: provide the matrix already (n x n)
        zeros = np.zeros_like(head_score_matrix[0]).reshape([1, -1])
        score_matrix = np.concatenate([zeros, head_score_matrix], 0)

    if single_root:
        pred_heads = chu_liu_edmonds_one_root(score_matrix)
    else:
        pred_heads = chu_liu_edmonds(score_matrix)

    pred_heads = pred_heads[1:]
    if label_matrix is not None:
        pred_labels = []
        for m, h in enumerate(pred_heads):
            pred_labels.append(label_matrix[m, h])

        pred_labels = np.array(pred_labels)
    else:
        offset_labels = parts.get_type_offset(Target.RELATIONS)
        num_labeled = parts.num_labeled_arcs
        labeled_parts = predicted_parts[offset_labels:
                                        offset_labels+num_labeled]
        labeled_parts2d = labeled_parts.reshape([parts.num_arcs, -1])

        # best_labels contains the best label for each arc
        best_labels = labeled_parts2d.argmax(1)
        arc_index = parts.create_arc_index()
        pred_labels = [best_labels[arc_index[h, m]]
                       for (m, h) in enumerate(pred_heads, 1)]

    return pred_heads, pred_labels


def decode_matrix_tree(scores):
    """
    Compute the log-partition function and its gradients (marginal
    probabilities) for non-projective dependency parser given a weighted
    adjacency matrix `scores: head × modifier ↦ log-weight` (note: diagonal
    ignored). Root should be counted as a possible head but not modifier.

    Implementation adapted from Tim Vieira.
    (https://github.com/timvieira/spanning_tree)

    References
    - Koo, Globerson, Carreras and Collins (EMNLP'07)
      Structured Prediction Models via the Matrix-Tree Theorem
      https://www.aclweb.org/anthology/D07-1015
    - Smith and Smith (EMNLP'07)
      Probabilistic Models of Nonprojective Dependency Trees
      https://www.aclweb.org/anthology/D07-1014
    - McDonald & Satta (IWPT'07)
      https://www.aclweb.org/anthology/W07-2216

    :return: a tuple (log_partition_function, marginals).
        log_partition_function is a float
        marginals is a matrix (head, modifier) with root included as head only
    """
    # treat single word sentences as a special case
    # (if the score of the root is low enough, we run into a singular matrix
    # and the general-case code breaks)
    if len(scores) == 2:
        # there is only one valid arc in the tree
        log_z = scores[0][0]
        marginals = np.array([[1.], [0.]], dtype=scores.dtype)

        return log_z, marginals

    # Numerical stability trick: We use an extension of the log-sum-exp and
    # exp-normalize tricks to our log-det-exp setting. [I haven't never seen
    # this trick elsewhere.]
    #
    # Note: The `exp` function below is point-wise exponential, not the matrix
    # exponential!
    #
    # for any value c,
    #
    #    log(det(exp(c) * exp(A - c)))
    #     = log(exp(c)^n * det(exp(A - c)))
    #     = c*n + log(det(exp(A - c)))
    #
    # Furthermore,
    #
    #    ∇ log(det(exp(c) * exp(A - c)))
    #     = ∇ [ c*n + log(det(exp(A - c))) ]
    #     = exp(A - c)⁻ᵀ

    # double precision is important to avoid some overflows
    scores = scores.astype(np.float64)
    c = scores.max()
    scores_plus = scores - scores.min()
    scores_minus = scores - c

    # if c is too high, we may have underflows resulting in a matrix full of
    # zeroes and without an inverse. Add a small term to avoid that and keep
    # the relative ordering
    # (this line may cause underflows in the multiplication, but it is harmless
    # as long as the resulting exp_scores are non-zero)
    exp_scores = np.exp(scores_minus) + 1e-40 * scores_plus

    # split root from real words
    r = exp_scores[0]
    A = exp_scores[1:]

    np.fill_diagonal(A, 0)

    L = np.diag(A.sum(axis=0)) - A   # The Laplacian matrix of a graph
    L[0, :] = r                      # Koo et al.'s efficiency trick

    log_z = np.linalg.slogdet(L)[1] + c * len(r)

    dL = np.linalg.inv(L).T
    dr = r * dL[0, :]

    num_tokens = len(r)
    h = np.arange(num_tokens)
    m = np.arange(num_tokens)
    dA = A * (dL[m, m] * (m != 0) - dL * (h.reshape(-1, 1) != 0))

    marginals = np.concatenate([dr.reshape([1, -1]), dA])

    return log_z, marginals


def decode_labels(parts, scores):
    """
    Take the highest scoring label for each part and their scores.

    :param parts: DependencyParts
    :param scores: dictionary mapping targets to arrays with scores
    :return: a tuple (best_labels, best_label_scores)
    """
    relation_scores = scores[Target.RELATIONS]

    # reshape as (num_arcs, num_relations)
    relation_scores = relation_scores.reshape(-1, parts.num_relations)

    # best_labels[i] has the best label for the i-th arc
    best_labels = relation_scores.argmax(-1)

    inds = np.expand_dims(best_labels, 1)
    best_label_scores = np.take_along_axis(relation_scores, inds, 1)

    return best_labels, best_label_scores.squeeze(1)


def batch_decode(instance_data: InstanceData, scores: list, n_jobs=1):
    """
    Decode the scores of dependency parts in parallel.

    This function uses multiprocessing instead of multithreading.

    :param instance_data: InstanceData object
    :param scores: list of dictionaries mapping target names to scores
    :param n_jobs: number of jobs to run in parallel. 1 avoids parallelization
    :return: list of arrays with the prediction probability of each part
    """
    num_sentences = len(instance_data)
    p = Parallel(n_jobs=n_jobs)
    decoded = p(delayed(decode)(instance_data.instances[i],
                                instance_data.parts[i],
                                scores[i])
                for i in range(num_sentences))

    return decoded


def decode(instance, parts, scores):
    """
    Decode the scores to the dependency parts under the necessary
    contraints, yielding a valid dependency tree.

    :param instance: DependencyInstance
    :param parts: a DependencyParts object holding all the parts included
        in the scoring functions; usually arcs, siblings and grandparents
    :type parts: DependencyParts
    :param scores: dictionary mapping target names (such as arcs, relations,
        grandparents etc) to arrays of scores. Arc scores may have padding;
        it is treated internally.
    :return: an array with the prediction probability of each part.
    """
    graph_wrapper = FactorGraph(instance, parts, scores)
    graph = graph_wrapper.graph

    graph.set_eta_ad3(.05)
    graph.adapt_eta_ad3(True)
    graph.set_max_iterations_ad3(500)
    graph.set_residual_threshold_ad3(1e-3)

    predicted_output = np.zeros(len(parts), np.float)
    num_arcs = parts.num_arcs

    value, posteriors, additional_posteriors, status = \
        graph.solve_lp_map_ad3()

    assert len(posteriors) == num_arcs
    assert len(additional_posteriors) == (len(parts) - num_arcs
                                          - parts.num_labeled_arcs)

    predicted_output[:num_arcs] = posteriors
    for index, score in zip(graph_wrapper.additional_indices,
                            additional_posteriors):
        predicted_output[index] = score

    # if doing labeled parsing, set the score of the best label for each
    # arc to be the same as the score of the arc
    offset = num_arcs
    num_relations = parts.num_relations

    for i in range(len(graph_wrapper.arcs)):
        label = graph_wrapper.best_labels[i]
        posterior = posteriors[i]
        predicted_output[offset + label] = posterior
        offset += num_relations

    return predicted_output


def decode_marginals(scores: dict) -> tuple:
    """
    Decode the scores generated by the pruner constrained to be a
    non-projective tree, yielding marginal probabilities for all parts.

    It considers a factorization into unlabeled and labeled parts.

    :param scores: dictionary mapping target names to arrays. Target.HEADS
        should map to (modifier, head) and Target.RELATIONS to (modifier, head,
        label). Both should only include root as a head.
    :return: a tuple (head_marginals, label_marginals, log_partition, entropy)
        log_partition and entropy are floats, the marginals are arrays with
        the same shape as the scores.
    """
    # (n + 1, n, labels)
    label_scores = scores[Target.RELATIONS].transpose(1, 0, 2)
    # (n + 1, n)
    arc_scores = scores[Target.HEADS].T

    # add the label partition function for each arc to the arc score itself
    label_z = logsumexp(label_scores, -1)
    log_z, arc_marginals = decode_matrix_tree(arc_scores + label_z)

    if np.any(arc_marginals < 0):
        lowest = arc_marginals.min()
        if lowest < -1e-6:
            logger.warning('Negative arc marginal: %.8f' % lowest)

        arc_marginals[arc_marginals < 0] = 0.

    # label conditionals contain P(label | arc)
    # (operation equivalent to softmax)
    # label_conditionals will be > 0 for (i, i), but when we compute the
    # posterior it will be 0
    label_z_3d = np.expand_dims(label_z, 2)
    label_conditionals = np.exp(label_scores - label_z_3d)

    # label marginals are P(label | arc) * P(arc)
    label_marginals = label_conditionals * np.expand_dims(arc_marginals, 2)

    # replace -inf with 0 to avoid nan and numpy warnings
    arc_scores[np.isinf(arc_scores)] = 0.
    entropy = log_z - (label_scores * label_marginals).sum() - \
        (arc_scores * arc_marginals).sum()

    if entropy < 0:
        if entropy < -1e-6:
            logger.warning('Negative marginal entropy: %f' % entropy)
        entropy = 0.

    # same shape as input
    arc_marginals = arc_marginals.T
    label_marginals = label_marginals.transpose(1, 0, 2)

    return arc_marginals, label_marginals, log_z, entropy


def generate_arc_mask(marginals: np.array, max_heads: int,
                      threshold: float = None):
    """
    Decode the scores generated by a pruner to generate an arc mask.

    :param marginals: array (h, m) with the marginal probability of each arc
    :param max_heads: maximum allowed head candidates for modifier
    :param threshold: prune arcs (h, m) with a score lower than this value
        multiplied by the highest scoring arc (h', m) for each word m.
    :return: a tuple (arc_mask, entropy).
        arc_mask is a boolean 2d array masking arcs. It has shape (n, n) where
        n is the instance length including root. Position (h, m) has True
        if the arc is valid, False otherwise.

        entropy is the entropy of the marginal arc probabilities (computed
        with the matrix-tree theorem)
    """
    n = len(marginals)

    # max_marginals contains the highest probability head for each word
    max_marginals = marginals.max(0)

    # in some edge cases, the probabilities for all heads to a word are 0
    invalid = max_marginals <= 0
    if np.any(invalid):
        num_tokens = invalid.sum()
        msg = 'All heads have zero probability for %d token(s)' % num_tokens
        logger.info(msg)
        max_marginals[invalid] = 1

    abs_threshold = 0 if threshold is None else threshold * max_marginals

    if n > max_heads:
        # clip values below the top k
        sorted_inds = marginals.argsort(0)
        np.put_along_axis(marginals, sorted_inds[:-max_heads], 0, 0)

    # marginals → (n + 1, n)
    # max_marginals → (n)
    mask = marginals >= abs_threshold

    # allow all heads where all had 0 probability (anything goes!)
    mask[:, invalid] = True

    # mask is expected to be (n + 1, n + 1)
    mask = np.concatenate([np.zeros([n, 1], dtype=np.bool), mask], 1)

    return mask


class FactorGraph(object):

    def __init__(self, instance, parts, scores):
        self.left_siblings = None
        self.right_siblings = None
        self.left_grandparents = None
        self.right_grandparents = None
        self.left_grandsiblings = None
        self.right_grandsiblings = None

        self.use_siblings = parts.has_type(Target.NEXT_SIBLINGS)
        self.use_grandparents = parts.has_type(Target.GRANDPARENTS)
        self.use_grandsiblings = parts.has_type(Target.GRANDSIBLINGS)

        # arcs is a list of tuples (h, m)
        heads, modifiers = parts.get_arc_indices()
        self.arcs = list(zip(heads, modifiers))
        self.arc_index = parts.create_arc_index()

        # these indices keep track of the higher order parts added to the graph
        self.additional_indices = []

        self.graph = fg.PFactorGraph()
        variables = self.create_tree_factor(instance, parts, scores)

        self._index_parts_by_head(parts, instance, scores)

        if self.use_grandsiblings or \
                (self.use_siblings and self.use_grandparents):
            self.create_gp_head_automata(variables)
        elif self.use_grandparents:
            self.create_grandparent_factors(parts, scores, variables)
        elif self.use_siblings:
            self.create_head_automata(variables)

    def _index_parts_by_head(self, parts, instance, scores):
        """
        Create data structures mapping heads to lists of dependency parts,
        such as siblings or grandparents. The data strutctures are member
        variables.

        :type parts: DependencyParts
        :type instance: DependencyInstance
        :param scores: dictionary mapping target names to scores
        :type scores: dict
        """
        n = len(instance)

        self.left_siblings = create_empty_structures(n)
        self.right_siblings = create_empty_structures(n)
        if self.use_siblings:
            _populate_structure_list(
                self.left_siblings, self.right_siblings, parts, scores,
                Target.NEXT_SIBLINGS)

        self.left_grandparents = create_empty_structures(n)
        self.right_grandparents = create_empty_structures(n)
        if self.use_grandparents:
            _populate_structure_list(
                self.left_grandparents, self.right_grandparents, parts, scores,
                Target.GRANDPARENTS)

        self.left_grandsiblings = create_empty_structures(n)
        self.right_grandsiblings = create_empty_structures(n)
        if self.use_grandsiblings:
            _populate_structure_list(
                self.left_grandsiblings, self.right_grandsiblings, parts,
                scores, Target.GRANDSIBLINGS)

    def _add_margin_vector(self, parts, scores):
        """
        Add the margin to the scores.

        It only affects Arcs or LabeledArcs (in case the latter are used).

        This is used before actually decoding.

        :type parts: DependencyParts
        :param scores: a dictionary mapping target names to scores produced by
            the network. The scores must be 1-d arrays.
        """
        if parts.labeled:
            # place the margin on LabeledArcs scores
            key = Target.RELATIONS
            num_parts = parts.num_labeled_arcs
        else:
            # place the margin on Arc scores
            key = Target.HEADS
            num_parts = parts.num_arcs

        offset = parts.offsets[key]
        gold_values = parts.gold_parts[offset:offset + num_parts]
        scores[key] += 0.5 - gold_values

    def create_tree_factor(self, instance, parts, scores):
        """
        Include factors to constrain the graph to a valid dependency tree.

        :type instance: DependencyInstance
        :type parts: DependencyParts
        :param scores: dictionary mapping target names to scores.
            It should have a key for Target.HEADS and another for
            Target.RELATIONS if labels are used
        :return: a list of arc variables. The i-th variable corresponds to the
            i-th arc in parts.
        """
        # length is the number of tokens in the instance, including root
        length = len(instance)

        best_labels, label_scores = decode_labels(parts, scores)
        arc_scores = scores[Target.HEADS] + label_scores
        self.best_labels = best_labels

        tree_factor = PFactorTree()
        variables = []

        for i in range(len(self.arcs)):
            arc_variable = self.graph.create_binary_variable()
            arc_variable.set_log_potential(arc_scores[i])
            variables.append(arc_variable)

        # owned_by_graph makes the factor persist after calling this function
        # if left as False, the factor is garbage collected
        self.graph.declare_factor(tree_factor, variables, owned_by_graph=True)
        tree_factor.initialize(length, self.arcs)

        return variables

    def create_gp_head_automata(self, variables):
        """
        Include grandsibling factors (grandparent head automata) in the graph.

        :param variables: list of binary variables denoting arcs
        """
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

                # (g, h) arcs must always be in increasing order for AD3
                # we must include (g, h) even if there is no grandparent part
                # this happens when the only sibling part is with null siblings
                h = sib_tuples[0][0]
                incoming_arcs = []
                incoming_var_inds = []
                for g in range(n):
                    index = self.arc_index[g, h]
                    if index >= 0:
                        incoming_var_inds.append(index)
                        incoming_arcs.append((g, h))

                # get arcs from siblings because we must include even outgoing
                # arcs that would make a cycle with the grandparent
                outgoing_arcs = siblings_structure.get_arcs(decreasing)

                if len(incoming_arcs) == 0:
                    # no grandparent structure; create simple head automaton
                    self._create_head_automata(
                        [siblings_structure], variables, decreasing)
                    continue

                outgoing_var_inds = [self.arc_index[arc[0], arc[1]]
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
                self.graph.declare_factor(factor, local_variables,
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

    def create_grandparent_factors(self, parts, scores, variables):
        """
        Include grandparent factors for constraining grandparents in the graph.

        :param parts: DependencyParts
        :param scores: np.array
        :param variables: list of binary variables denoting arcs
        """
        for i, part in parts.part_lists[Target.GRANDPARENTS]:
            head = part.head
            modifier = part.modifier
            grandparent = part.grandparent

            index_hm = self.arc_index[head, modifier]
            index_gh = self.arc_index[grandparent, head]

            var_hm = variables[index_hm]
            var_gh = variables[index_gh]

            score = scores[Target.GRANDPARENTS][i]
            self.graph.create_factor_pair([var_hm, var_gh], score)
            self.additional_indices.append(i)

    def _create_head_automata(self, structures, variables, decreasing):
        """
        Creates and sets the head automaton factors for either left or
        right siblings.

        :param structures: a list of PartStructure objects containing
            next sibling parts
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
            var_inds = [self.arc_index[arc[0], arc[1]] for arc in arcs]
            local_variables = [variables[i] for i in var_inds]
            siblings = [(p.head, p.modifier, p.sibling)
                        for p in head_structure.parts]

            # important: first declare the factor in the graph,
            # then initialize
            factor = PFactorHeadAutomaton()
            self.graph.declare_factor(factor, local_variables,
                                      owned_by_graph=True)
            factor.initialize(arcs, siblings, validate=False)
            factor.set_additional_log_potentials(head_structure.scores)

            self.additional_indices.extend(indices)

    def create_head_automata(self, variables):
        """
        Include head automata for constraining consecutive siblings in the
        graph.

        :param variables: list of binary variables denoting arcs
        """
        # needed to map indices in parts to indices in variables
        self._create_head_automata(
            self.left_siblings, variables, decreasing=True)
        self._create_head_automata(
            self.right_siblings, variables, decreasing=False)


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
    :param scores: dictionary mapping target type to a score array
    :param right_list: same as above, for the right hand side.
    :param type_: a target type denoting some dependency part
    """
    part_list = parts.part_lists[type_]
    scores = scores[type_]
    offset = parts.get_type_offset(type_)

    for i, part in enumerate(part_list):

        # make this check because modifier == head has a special meaning for
        # sibling parts
        if isinstance(part, (NextSibling, GrandSibling)):
            is_right = part.sibling > part.head
        else:
            is_right = part.modifier > part.head

        if is_right:
            # right sibling
            right_list[part.head].append(part, scores[i], i + offset)
        else:
            # left sibling
            left_list[part.head].append(part, scores[i], i + offset)


def make_score_matrix(length, arc_mask, scores):
    """
    Makes a score matrix from an array of scores ordered in the same way as a
    list of DependencyPartArcs. Positions [m, h] corresponding to non-existing
    arcs have score of -inf.

    :param length: length of the sentence, including the root pseudo-token
    :param arc_mask: Arc mask as in DependencyParts, with shape (h, m),
    :param scores: array with score of each arc
    :return: a 2d numpy array (m, h), starting from 0.
    """
    score_matrix = np.full([length, length], -np.inf, np.float32)
    score_matrix[arc_mask] = scores

    return score_matrix.T


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


def chu_liu_edmonds_one_root(score_matrix):
    """
    Run the Chu-Liu-Edmonds' algorithm to find the maximum spanning tree with a
    single root.

    :param score_matrix: a matrix such that cell [m, h] has the score for the
        arc (h, m).
    :return: an array heads, such that heads[m] contains the head of token m.
        The root is in position 0 and has head -1.
    """
    def set_root(scores, root):
        """
        Return a tuple (new_scores, root_score)
        The first one is a new version of the original scores in which the token
        `root` is the new root, and every other token has score -inf for being
        root.
        The root_score is the original score for `root` being the actual root.
        """
        root_score = scores[root, 0]
        scores = np.array(scores)
        scores[1:, 0] = -np.inf
        scores[root] = -np.inf
        scores[root, 0] = 0
        return scores, root_score

    score_matrix = score_matrix.astype(np.float64)
    tree = chu_liu_edmonds(score_matrix)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0] + 1
    if len(roots_to_try) == 1:
        return tree

    best_score = -np.inf
    best_tree = None
    num_tokens = len(score_matrix)
    for root in roots_to_try:
        new_scores, root_score = set_root(score_matrix, root)
        new_tree = chu_liu_edmonds(new_scores)

        # scores are supposed to be the log probabilities of each arc
        tree_log_probs = new_scores[np.arange(num_tokens), new_tree]

        if (tree_log_probs > -np.inf).all():
            tree_score = tree_log_probs.sum() + root_score
            if tree_score > best_score:
                best_score = tree_score
                best_tree = new_tree

    assert best_tree is not None

    return best_tree
