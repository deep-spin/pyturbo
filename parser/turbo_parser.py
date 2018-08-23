from classifier.structured_classifier import StructuredClassifier
from classifier.neural_scorer import NeuralScorer
from parser.dependency_options import DependencyOptions
from parser.dependency_reader import DependencyReader
from parser.dependency_writer import DependencyWriter
from parser.dependency_decoder import DependencyDecoder
from parser.dependency_dictionary import DependencyDictionary
from parser.dependency_instance import DependencyInstanceOutput
from parser.dependency_instance_numeric import DependencyInstanceNumeric
from parser.token_dictionary import TokenDictionary
from parser.dependency_parts import DependencyParts, \
    DependencyPartArc, DependencyPartLabeledArc, DependencyPartGrandparent, \
    DependencyPartNextSibling
from parser.dependency_features import DependencyFeatures
from parser.dependency_neural_model import DependencyNeuralModel, special_tokens
import numpy as np
import pickle
import logging


class TurboParser(StructuredClassifier):
    '''Dependency parser.'''
    def __init__(self, options):
        StructuredClassifier.__init__(self, options)
        self.token_dictionary = TokenDictionary(self)
        self.dictionary = DependencyDictionary(self)
        self.reader = DependencyReader()
        self.writer = DependencyWriter()
        self.decoder = DependencyDecoder()
        self.parameters = None
        self.model_type = options.model_type.split('+')
        self.has_pruner = False

        if self.options.train:
            for token in special_tokens:
                self.token_dictionary.add_special_symbol(token)
            self.token_dictionary.initialize(self.reader)
            self.dictionary.create_relation_dictionary(self.reader)
            if self.options.neural:
                if self.options.prune_basic:
                    pruner_model = DependencyNeuralModel(
                        self.token_dictionary,
                        self.dictionary,
                        word_embedding_size=100,
                        tag_embedding_size=20,
                        distance_embedding_size=20,
                        hidden_size=50,
                        num_layers=1,
                        dropout=0.
                    )
                    self.pruner_neural_scorer = NeuralScorer()
                    self.pruner_neural_scorer.initialize(pruner_model)
                    self.is_training_pruner = None

                self.neural_scorer.initialize(
                    DependencyNeuralModel(self.token_dictionary,
                                          self.dictionary,
                                          word_embedding_size=100,
                                          tag_embedding_size=20,
                                          distance_embedding_size=20,
                                          hidden_size=50,
                                          num_layers=1,
                                          dropout=0.))

    def save(self, model_path=None):
        '''Save the full configuration and model.'''
        if not model_path:
            model_path = self.options.model_path
        with open(model_path, 'wb') as f:
            pickle.dump(self.options, f)
            self.token_dictionary.save(f)
            self.dictionary.save(f)
            pickle.dump(self.parameters, f)
            if self.options.neural:
                pickle.dump(self.neural_scorer.model.word_embedding_size, f)
                pickle.dump(self.neural_scorer.model.tag_embedding_size, f)
                pickle.dump(self.neural_scorer.model.distance_embedding_size, f)
                pickle.dump(self.neural_scorer.model.hidden_size, f)
                pickle.dump(self.neural_scorer.model.num_layers, f)
                pickle.dump(self.neural_scorer.model.dropout, f)
                self.neural_scorer.model.save(f)

    def load(self, model_path=None):
        '''Load the full configuration and model.'''
        if not model_path:
            model_path = self.options.model_path
        with open(model_path, 'rb') as f:
            model_options = pickle.load(f)
            self.token_dictionary.load(f)
            self.dictionary.load(f)
            self.parameters = pickle.load(f)
            if model_options.neural:
                word_embedding_size = pickle.load(f)
                tag_embedding_size = pickle.load(f)
                distance_embedding_size = pickle.load(f)
                hidden_size = pickle.load(f)
                num_layers = pickle.load(f)
                dropout = pickle.load(f)
                neural_model = DependencyNeuralModel(
                    self.token_dictionary,
                    self.dictionary,
                    word_embedding_size=word_embedding_size,
                    tag_embedding_size=tag_embedding_size,
                    distance_embedding_size=distance_embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)
                neural_model.load(f)
                self.neural_scorer = NeuralScorer(neural_model)

        self.options.neural = model_options.neural
        self.options.model_type = model_options.model_type
        self.options.unlabeled = model_options.unlabeled
        self.options.projective = model_options.projective

        # prune arcs with label/head POS/modifier POS unseen in training data
        self.options.prune_relations = model_options.prune_relations

        # prune arcs with a distance unseen in training with the given POS tags
        self.options.prune_distances = model_options.prune_distances

        # use a first-order model to prune arcs
        self.options.prune_basic = model_options.prune_basic

        # threshold for the basic pruner, if used
        self.options.pruner_posterior_threshold = \
            model_options.pruner_posterior_threshold

        # maximum candidate heads per word in the basic pruner, if used
        self.options.pruner_max_heads = model_options.pruner_max_heads
    
    def train(self):
        """
        Train the parser and a pruner, if set in the options.
        """
        self.is_training_pruner = True
        super(TurboParser, self).train()
        self.has_pruner = True
        self.is_training_pruner = False
        super(TurboParser, self).train()
    
    def get_formatted_instance(self, instance):
        return DependencyInstanceNumeric(instance, self.dictionary)

    def compute_scores(self, instance, parts, features):
        """
        Compute scores for each part. If the pruner is being trained, use the
        pruner scorer.
        """
        if self.is_training_pruner:
            return self.compute_pruner_scores(instance, parts)
        else:
            return super(TurboParser, self).compute_scores(
                instance, parts, features)

    def compute_pruner_scores(self, instance, parts):
        """
        Compute the scores for every part according to the pruner
        """
        return self.pruner_neural_scorer.compute_scores(instance, parts)

    def prune(self, instance, parts):
        """
        Prune out some arcs according to the the pruner model.

        :param instance: a DependencyInstance object
        :param parts: a DependencyParts object with arcs
        :return: a new DependencyParts object contained the kept arcs
        """
        scores = self.compute_pruner_scores(instance, parts)
        new_parts = self.decoder.decode_pruner_naive(
            parts, scores, self.options.pruner_max_heads)

        # during training, make sure that the gold parts are included
        if self.options.train:
            for m in range(1, len(instance)):
                h = instance.output.heads[m]
                if new_parts.find_arc_index(h, m) < 0:
                    new_parts.append(DependencyPartArc(h, m))

        print('Original parts had len', len(parts), ', pruned has',
              len(new_parts))
        return new_parts

    def make_parts(self, instance):
        """
        Create the parts (arcs) into which the problem is factored.

        :param instance: a DependencyInstance object
        :return: if instances have the expected output, return a tuple
            (parts, gold_output). If they don't, return only the parts.
        """
        parts = DependencyParts()
        include_gold = instance.output is not None
        gold_output = []

        partial_gold = self.make_parts_basic(instance, parts,
                                             add_relation_parts=False)
        gold_output.append(partial_gold)

        if not self.is_training_pruner:
            if self.has_pruner:
                parts = self.prune(instance, parts)

            if 'cs' in self.model_type:
                partial_gold = self.make_parts_consecutive_siblings(instance,
                                                                    parts)
                gold_output.append(partial_gold)
                # partial_gold = self.make_parts_grandparent(instance, parts)
                # gold_output.append(partial_gold)

        if instance.output is not None:
            gold_output = np.concatenate(gold_output)
            return parts, gold_output

        return parts

    def make_parts_consecutive_siblings(self, instance, parts):
        """
        Create the parts relative to consecutive siblings.

        Each part means that an arc h -> m and h -> s exist at the same time,
        with both h > m and h > s or both h < m and h < s.

        :param instance: DependencyInstance
        :param parts: a DependencyParts object. It must already have been
            pruned.
        :type parts: DependencyParts
        :return: if `instances` have the attribute output, return a numpy array
            with the gold output. If it doesn't, return None.
        """
        make_gold = instance.output is not None
        gold_output = [] if make_gold else None

        initial_index = len(parts)
        for h in range(len(instance)):

            # siblings to the right of h
            # when m = h, it signals that s is the first child
            for m in range(h, len(instance)):

                if h != m and 0 > parts.find_arc_index(h, m):
                    # pruned out
                    continue

                gold_hm = m == h or _check_gold_arc(instance, h, m)
                arc_between = False

                # when s = length, it signals that m encodes the last child
                for s in range(m + 1, len(instance) + 1):
                    if s < len(instance) and 0 > parts.find_arc_index(h, s):
                        # pruned out
                        continue

                    parts.append(DependencyPartNextSibling(h, m, s))
                    if make_gold:
                        gold_hs = s == len(instance) or \
                                    _check_gold_arc(instance, h, s)

                        if gold_hm and gold_hs and not arc_between:
                            value = 1
                            arc_between = True
                        else:
                            value = 0
                        gold_output.append(value)

            # siblings to the left of h
            if h == 0:
                # root can't have any child to the left
                continue

            for m in range(h, 0, -1):
                if h != m and 0 > parts.find_arc_index(h, m):
                    # pruned out
                    continue

                gold_hm = m == h or _check_gold_arc(instance, h, m)
                arc_between = False

                # when s = 0, it signals that m encoded the leftmost child
                for s in range(m - 1, -1, -1):
                    if s != 0 and 0 > parts.find_arc_index(h, s):
                        # pruned out
                        continue

                    parts.append(DependencyPartNextSibling(h, m, s))
                    if make_gold:
                        gold_hs = s == 0 or _check_gold_arc(instance, h, s)

                        if gold_hm and gold_hs and not arc_between:
                            value = 1
                            arc_between = True
                        else:
                            value = 0
                        gold_output.append(value)

        parts.set_offset(DependencyPartNextSibling, initial_index,
                         len(parts) - initial_index)
        if make_gold:
            gold_output = np.array(gold_output)

        return gold_output

    def make_parts_grandparent(self, instance, parts):
        """
        Create the parts relative to grandparents.

        Each part means that an arc h -> m and g -> h exist at the same time.

        :param instance: DependencyInstance
        :param parts: a DependencyParts object. It must already have been
            pruned.
        :type parts: DependencyParts
        :return: if `instances` have the attribute output, return a numpy array
            with the gold output. If it doesn't, return None.
        """
        make_gold = instance.output is not None
        gold_output = [] if make_gold else None

        initial_index = len(parts)

        for g in range(len(instance)):
            for h in range(1, len(instance)):
                if g == h:
                    continue

                if 0 > parts.find_arc_index(g, h):
                    # the arc g -> h has been pruned out
                    continue

                gold_gh = _check_gold_arc(instance, g, h)

                for m in range(1, len(instance)):
                    if g == m or h == m:
                        continue

                    if 0 > parts.find_arc_index(h, m):
                        # pruned out
                        continue

                    arc = DependencyPartGrandparent(h, m, g)
                    parts.append(arc)
                    if make_gold:
                        if gold_gh and instance.get_head(m) == h:
                            gold_output.append(1)
                        else:
                            gold_output.append(0)

        parts.set_offset(DependencyPartGrandparent,
                         initial_index, len(parts) - initial_index)
        if make_gold:
            gold_output = np.array(gold_output)

        return gold_output

    def make_parts_global(self, instance):
        """
        Create the parts (arcs) involving global structures: siblings,
        grandparents, etc.

        :param instance: a DependencyInstance object
        :return: if `instances` have the attribute output, return a numpy array
            with the gold output. If it doesn't, return None.
        """
        #TODO: do we need this function?

    def make_parts_basic(self, instance, parts, add_relation_parts=True):
        """
        Create the first-order arcs into which the problem is factored.

        The objects `parts` is modified in-place.

        :param instance: a DependencyInstance object
        :param parts: a DependencyParts object, modified in-place
        :param add_relation_parts: whether to include label information
        :return: if `instances` have the attribute `output`, return a numpy
            array with 1 signaling the presence of an arc in a combination
            (head, modifier) or (head, modifier, label) and 0 otherwise.
            It is a one-dimensional array.

            If `instances` doesn't have the attribute `output` return None.
        """
        make_gold = instance.output is not None
        gold_outputs = [] if make_gold else None

        if add_relation_parts and not self.options.prune_relations:
            allowed_relations = range(len(
                self.dictionary.get_relation_alphabet()))

        num_parts_initial = len(parts)
        for h in range(len(instance)):
            for m in range(1, len(instance)):
                if h == m:
                    continue

                if add_relation_parts:
                    # If no unlabeled arc is there, just skip it.
                    # This happens if that arc was pruned out.
                    if 0 > parts.find_arc_index(h, m):
                        continue
                else:
                    if h and self.options.prune_distances:
                        modifier_tag = instance.get_tag(m)
                        head_tag = instance.get_tag(h)
                        if h < m:
                            # Right attachment.
                            if m - h > \
                               self.dictionary.get_maximum_right_distance(
                                   modifier_tag, head_tag):
                                continue
                        else:
                            # Left attachment.
                            if h - m > \
                               self.dictionary.get_maximum_left_distance(
                                   modifier_tag, head_tag):
                                continue
                if self.options.prune_relations:
                    modifier_tag = instance.get_tag(m)
                    head_tag = instance.get_tag(h)
                    allowed_relations = self.dictionary.get_existing_relations(
                        modifier_tag, head_tag)
                    if not add_relation_parts and not allowed_relations:
                        continue

                # Add parts for labeled/unlabeled arcs.
                if add_relation_parts:
                    # If there is no allowed relation for this arc, but the
                    # unlabeled arc was added, then it was forced to be present
                    # to maintain connectivity of the graph. In that case (which
                    # should be pretty rare) consider all the possible
                    # relations.
                    if not allowed_relations:
                        allowed_relations = range(len(
                            self.dictionary.get_relation_alphabet()))
                    for l in allowed_relations:
                        part = DependencyPartLabeledArc(h, m, l)
                        parts.append(part)
                        if make_gold:
                            if instance.get_head(m) == h and \
                               instance.get_relation(m) == l:
                                gold_outputs.append(1.)
                            else:
                                gold_outputs.append(0.)
                else:
                    part = DependencyPartArc(h, m)
                    parts.append(part)
                    if make_gold:
                        if instance.get_head(m) == h:
                            gold_outputs.append(1.)
                        else:
                            gold_outputs.append(0.)

        # When adding unlabeled arcs, make sure the graph stays connected.
        # Otherwise, enforce connectedness by adding some extra arcs
        # that connect words to the root.
        # NOTE: if --projective, enforcing connectedness is not enough,
        # so we add arcs of the form m-1 -> m to make sure the sentence
        # has a projective parse.
        if not add_relation_parts:
            arcs = parts[num_parts_initial:]
            inserted_arcs = self.enforce_well_formed_graph(instance, arcs)
            for h, m in inserted_arcs:
                part = DependencyPartArc(h, m)
                parts.append(part)
                if make_gold:
                    if instance.get_head(m) == h:
                        gold_outputs.append(1.)
                    else:
                        gold_outputs.append(0.)
            parts.set_offset(DependencyPartArc,
                             num_parts_initial, len(parts) - num_parts_initial)
        else:
            parts.set_offset(DependencyPartLabeledArc,
                             num_parts_initial, len(parts) - num_parts_initial)

        if make_gold:
            gold_outputs = np.array(gold_outputs)

        return gold_outputs

    def enforce_well_formed_graph(self, instance, arcs):
        if self.options.projective:
            return self.enforce_projective_graph(instance, arcs)
        else:
            return self.enforce_connected_graph(instance, arcs)

    def enforce_connected_graph(self, instance, arcs):
        '''Make sure the graph formed by the unlabeled arc parts is connected,
        otherwise there is no feasible solution.
        If necessary, root nodes are added and passed back through the last
        argument.'''
        inserted_arcs = []
        # Create a list of children for each node.
        children = [[] for i in range(len(instance))]
        for r in range(len(arcs)):
            assert type(arcs[r]) == DependencyPartArc
            children[arcs[r].head].append(arcs[r].modifier)

        # Check if the root is connected to every node.
        visited = [False] * len(instance)
        nodes_to_explore = [0]
        while nodes_to_explore:
            h = nodes_to_explore.pop(0)
            visited[h] = True
            for m in children[h]:
                if visited[m]:
                    continue
                nodes_to_explore.append(m)
            # If there are no more nodes to explore, check if all nodes
            # were visited and, if not, add a new edge from the node to
            # the first node that was not visited yet.
            if not nodes_to_explore:
                for m in range(1, len(instance)):
                    if not visited[m]:
                        logging.info('Inserted root node 0 -> %d.' % m)
                        inserted_arcs.append((0, m))
                        nodes_to_explore.append(m)
                        break

        return inserted_arcs

    def enforce_projective_graph(self, instance, arcs):
        raise NotImplementedError

    def make_selected_features(self, instance, parts, selected_parts):
        """
        Create a DependencyFeatures object to store features describing the
        instance.

        :param instance: a DependencyInstance object
        :param parts: a list of dependency arcs
        :param selected_parts: a list of booleans, indicating for which parts
            features should be computed
        :return: a DependencyFeatures object containing a feature list for each
            arc
        """
        features = DependencyFeatures(self, parts)
        pruner = False

        # Even in the case of labeled parsing, build features for unlabeled arcs
        # only. They will later be conjoined with the labels.
        offset, size = parts.get_offset(DependencyPartArc)
        for r in range(offset, offset + size):
            if not selected_parts[r]:
                continue
            arc = parts[r]
            assert arc.head >= 0
            if pruner:
                features.add_arc_features_light(instance, r, arc.head,
                                                arc.modifier)
            else:
                features.add_arc_features(instance, r, arc.head, arc.modifier)

        return features

    def label_instance(self, instance, parts, output):
        heads = [-1 for i in range(len(instance))]
        relations = ['NULL' for i in range(len(instance))]
        instance.output = DependencyInstanceOutput(heads, relations)
        threshold = .5
        if self.options.unlabeled:
            offset, size = parts.get_offset(DependencyPartArc)
            for r in range(offset, offset + size):
                arc = parts[r]
                if output[r] >= threshold:
                    instance.output.heads[arc.modifier] = arc.head
        else:
            offset, size = parts.get_offset(DependencyPartLabeledArc)
            for r in range(offset, offset + size):
                arc = parts[r]
                if output[r] >= threshold:
                    instance.output.heads[arc.modifier] = arc.head
                    instance.output.relations[arc.modifier] = \
                        self.dictionary.get_relation_name(arc.label)
        for m in range(1, len(instance)):
            if instance.get_head(m) < 0:
                logging.info('Word without head.')
                instance.output.heads[m] = 0
                if not self.options.unlabeled:
                    instance.output.relations[m] = \
                        self.dictionary.get_relation_name(0)


def _check_gold_arc(instance, head, modifier):
    """
    Auxiliar function to check whether there is an arc from head to
    modifier in the gold output in instance.

    If instance has no gold output, return False.

    :param instance: a DependencyInstance
    :param head: integer, index of the head
    :param modifier: integer
    :return: boolean
    """
    if instance.output is None:
        return False
    if instance.get_head(modifier) == head:
        return True
    return False


def main():
    '''Main function for the dependency parser.'''
    # Parse arguments.
    import argparse
    parser = argparse. \
        ArgumentParser(prog='Turbo parser.',
                       description='Trains/test a dependency parser.')
    options = DependencyOptions(parser)
    args = vars(parser.parse_args())
    options.parse_args(args)

    if options.train:
        logging.info('Training parser...')
        train_parser(options)
    elif options.test:
        logging.info('Running parser...')
        test_parser(options)

def train_parser(options):
    logging.info('Training the parser...')
    dependency_parser = TurboParser(options)
    dependency_parser.train()
    dependency_parser.save()

def test_parser(options):
    logging.info('Running the parser...')
    dependency_parser = TurboParser(options)
    dependency_parser.load()
    dependency_parser.run()


if __name__ == "__main__":
    main()
