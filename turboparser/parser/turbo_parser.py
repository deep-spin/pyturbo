from ..classifier.structured_classifier import StructuredClassifier
from ..classifier.neural_scorer import NeuralScorer
from ..classifier import utils
from .dependency_reader import DependencyReader
from .dependency_writer import DependencyWriter
from .dependency_decoder import DependencyDecoder
from .dependency_dictionary import DependencyDictionary
from .dependency_instance import DependencyInstanceOutput
from .dependency_instance_numeric import DependencyInstanceNumeric
from .token_dictionary import TokenDictionary
from .dependency_parts import DependencyParts, Arc, LabeledArc, Grandparent, \
    NextSibling, GrandSibling
from .dependency_features import DependencyFeatures
from .dependency_neural_model import DependencyNeuralModel

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
        self._set_options()        

        if self.options.train:
            word_indices, embeddings = self._load_embeddings()
            self.token_dictionary.initialize(
                self.reader, self.options.form_case_sensitive, word_indices)
            embeddings = self._update_embeddings(embeddings)
            self.dictionary.create_relation_dictionary(self.reader)

            if self.options.neural:
                if embeddings is None:
                    embeddings = self._create_random_embeddings()

                model = DependencyNeuralModel(
                    self.token_dictionary, self.dictionary, embeddings,
                    tag_embedding_size=self.options.tag_embedding_size,
                    distance_embedding_size=self.options.
                    distance_embedding_size,
                    rnn_size=self.options.rnn_size,
                    mlp_size=self.options.mlp_size,
                    num_layers=self.options.num_layers,
                    dropout=self.options.dropout)
                self.neural_scorer.initialize(model, self.options.learning_rate)

    def _create_random_embeddings(self):
        """
        Create random embeddings for the vocabulary of the token dict.
        """
        num_words = self.token_dictionary.get_num_embeddings()
        dim = self.options.embedding_size
        embeddings = np.random.normal(0, 0.1, [num_words, dim])
        return embeddings

    def _load_embeddings(self):
        """
        Load word embeddings and dictionary. If they are not used, return both
        as None.

        :return: word dictionary, numpy embedings
        """
        if self.options.embeddings is not None:
            words, embeddings = utils.read_embeddings(self.options.embeddings)
        else:
            words = None
            embeddings = None

        return words, embeddings

    def _update_embeddings(self, embeddings):
        """
        Update the embedding matrix creating new ones if needed by the token
        dictionary.
        """
        if embeddings is not None:
            num_new_words = self.token_dictionary.get_num_embeddings() - \
                            len(embeddings)
            dim = embeddings.shape[1]
            new_vectors = np.random.normal(embeddings.mean(), embeddings.std(),
                                           [num_new_words, dim])
            embeddings = np.concatenate([embeddings, new_vectors])

        return embeddings

    def _set_options(self):
        """
        Set some parameters of the parser determined from its `options`
        attribute.
        """
        self.model_type = self.options.model_type.split('+')
        self.has_pruner = bool(self.options.pruner_path)

        if self.has_pruner:
            self.pruner = self.load_pruner(self.options.pruner_path)
        else:
            self.pruner = None

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
                pickle.dump(self.neural_scorer.model.embedding_vocab_size, f)
                pickle.dump(self.neural_scorer.model.word_embedding_size, f)
                pickle.dump(self.neural_scorer.model.tag_embedding_size, f)
                pickle.dump(self.neural_scorer.model.distance_embedding_size, f)
                pickle.dump(self.neural_scorer.model.rnn_size, f)
                pickle.dump(self.neural_scorer.model.mlp_size, f)
                pickle.dump(self.neural_scorer.model.num_layers, f)
                pickle.dump(self.neural_scorer.model.dropout_rate, f)
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
                embedding_vocab_size = pickle.load(f)
                word_embedding_size = pickle.load(f)
                tag_embedding_size = pickle.load(f)
                distance_embedding_size = pickle.load(f)
                rnn_size = pickle.load(f)
                mlp_size = pickle.load(f)
                num_layers = pickle.load(f)
                dropout = pickle.load(f)
                dummy_embeddings = np.empty([embedding_vocab_size,
                                             word_embedding_size], np.float32)
                neural_model = DependencyNeuralModel(
                    self.token_dictionary,
                    self.dictionary, dummy_embeddings,
                    tag_embedding_size=tag_embedding_size,
                    distance_embedding_size=distance_embedding_size,
                    rnn_size=rnn_size,
                    mlp_size=mlp_size,
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

        # threshold for the basic pruner, if used
        self.options.pruner_posterior_threshold = \
            model_options.pruner_posterior_threshold

        # maximum candidate heads per word in the basic pruner, if used
        self.options.pruner_max_heads = model_options.pruner_max_heads
        self._set_options()

    def should_save(self, validation_loss):
        """
        Return a bool for whether the model should be saved. This function
        should be called after running on validation data.

        It returns True if validation UAS increased in the last epoch, False
        otherwise.
        """
        return self._should_save

    def _reset_best_validation_metric(self):
        """
        Set the best validation UAS score to 0
        """
        self.best_validation_uas = 0
        self._should_save = False

    def _reset_task_metrics(self):
        """
        Reset the accumulated UAS counter
        """
        self.accumulated_uas = 0.
        self.total_tokens = 0
        self.validation_uas = 0.

    def _get_task_train_report(self):
        """
        Return task-specific metrics on training data.

        :return: a string describing naive UAS on training data
        """
        uas = self.accumulated_uas / self.total_tokens
        return 'Naive train UAS: %f' % uas

    def _get_task_valid_report(self):
        """
        Return task-specific metrics on validation data.

        :return: a string describing naive UAS on validation data
        """
        return 'Naive validation UAS: %f' % self.validation_uas

    def _update_task_metrics(self, predicted, gold, instance, parts):
        """
        Update the accumulated UAS count. It sums the UAS for each sentence
        scaled by its number of tokens; when reporting performance, this value
        is divided by the total number of tokens seen in all sentences combined.
        """
        # UAS doesn't consider the root
        length = len(instance) - 1
        uas = get_naive_uas(predicted, gold, parts, length)
        self.accumulated_uas += length * uas
        self.total_tokens += length

    def load_pruner(self, model_path):
        """
        Load and return a pruner model.

        This function takes care of keeping the main parser and the pruner
        configurations separate.
        """
        logging.info('Loading pruner from %s' % model_path)
        with open(model_path, 'rb') as f:
            pruner_options = pickle.load(f)

        pruner_options.train = False
        pruner = TurboParser(pruner_options)
        pruner.load(model_path)
        return pruner

    def format_instance(self, instance):
        return DependencyInstanceNumeric(instance, self.dictionary)

    def prune(self, instance, parts, gold_output):
        """
        Prune out some arcs with the pruner model.

        :param instance: a DependencyInstance object
        :param parts: a DependencyParts object with arcs
        :param gold_output: either None or a 0/1 array indicating gold parts
        :return: a new DependencyParts object contained the kept arcs
        """
        scores = self.pruner.compute_scores(instance, parts)[0]
        new_parts, new_gold = self.decoder.decode_matrix_tree(
            len(instance), parts.arc_index, parts, scores, gold_output,
            self.options.pruner_max_heads,
            self.options.pruner_posterior_threshold)

        # during training, make sure that the gold parts are included
        if not self.options.train:
            for m in range(1, len(instance)):
                h = instance.output.heads[m]
                if new_parts.find_arc_index(h, m) < 0:
                    new_parts.append(Arc(h, m))
                    new_gold.append(1)

            new_parts.set_offset(Arc, 0, len(new_parts))

        return new_parts, new_gold

    def _report_make_parts(self, instances, parts):
        """
        Log some statistics about the calls to make parts in a dataset.

        :type instances: list[DependencyInstance]
        :type parts: list[DependencyParts]
        """
        num_arcs = 0
        num_pruner_mistakes = 0
        num_tokens = 0
        num_possible_arcs = 0
        output_available = instances[0].output is not None

        for instance, inst_parts in zip(instances, parts):
            inst_len = len(instance)
            num_tokens += inst_len - 1  # exclude root
            num_possible_arcs += (inst_len - 1) ** 2  # exclude root and self

            # skip the root symbol
            for m in range(1, inst_len):
                for h in range(inst_len):
                    r = inst_parts.find_arc_index(h, m)
                    if r < 0:
                        # pruned
                        if output_available and instance.output.heads[m] == h:
                            num_pruner_mistakes += 1
                        continue

                    num_arcs += 1

        msg = '%f heads per token after pruning' % (num_arcs / num_tokens)
        logging.info(msg)

        msg = '%d arcs after pruning, out of %d possible (%f)' % \
              (num_arcs, num_possible_arcs, num_arcs / num_possible_arcs)
        logging.info(msg)

        if output_available:
            ratio = (num_tokens - num_pruner_mistakes) / num_tokens
            msg = 'Pruner recall (gold arcs retained after pruning): %f' % ratio
            logging.info(msg)

    def make_parts(self, instance):
        """
        Create the parts (arcs) into which the problem is factored.

        :param instance: a DependencyInstance object
        :return: a tuple (parts, gold_output). If the instances don't have the
            gold label, `gold_output` is None. If it does, it is a numpy array.
        """
        parts = DependencyParts()
        gold_output = None if instance.output is None else []

        self.make_parts_basic(instance, parts, gold_output)
        if self.has_pruner:
            parts, gold_output = self.prune(instance, parts, gold_output)

        assert len(parts) == len(gold_output)

        if not self.options.unlabeled:
            self.make_parts_labeled(instance, parts, gold_output)

        if 'cs' in self.model_type:
            self.make_parts_consecutive_siblings(instance, parts,
                                                 gold_output)
        if 'gp' in self.model_type:
            self.make_parts_grandparent(instance, parts, gold_output)

        if 'gs' in self.model_type:
            self.make_parts_grandsibling(instance, parts, gold_output)

        if instance.output is not None:
            gold_output = np.array(gold_output)
            num_parts = len(parts)
            num_gold = len(gold_output)
            assert num_parts == num_gold, \
                'Number of parts = %d and number of gold outputs = % d' \
                % (num_parts, num_gold)

        return parts, gold_output

    def print_parts(self, part_type, parts, gold):
        """
        Print the parts of a given type and their respective gold labels.

        This function is for debugging purposes.

        :param part_type: a subclass of DependencyPart
        :param parts:
        :param gold:
        :return:
        """
        assert isinstance(parts, DependencyParts)
        print('Iterating over parts of type', part_type.__name__)
        for i, part in parts.iterate_over_type(part_type, True):
            gold_label = gold[i] if gold is not None else None
            print(part, gold_label)

    def make_parts_consecutive_siblings(self, instance, parts, gold_output):
        """
        Create the parts relative to consecutive siblings.

        Each part means that an arc h -> m and h -> s exist at the same time,
        with both h > m and h > s or both h < m and h < s.

        :param instance: DependencyInstance
        :param parts: a DependencyParts object. It must already have been
            pruned.
        :type parts: DependencyParts
        :param gold_output: either None or a list with binary values indicating
            the presence of each part.
        :return: if `instances` have the attribute output, return a numpy array
            with the gold output. If it doesn't, return None.
        """
        make_gold = instance.output is not None

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

                    parts.append(NextSibling(h, m, s))
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

                    parts.append(NextSibling(h, m, s))
                    if make_gold:
                        gold_hs = s == 0 or _check_gold_arc(instance, h, s)

                        if gold_hm and gold_hs and not arc_between:
                            value = 1
                            arc_between = True
                        else:
                            value = 0
                        gold_output.append(value)

        parts.set_offset(NextSibling, initial_index,
                         len(parts) - initial_index)

    def make_parts_grandsibling(self, instance, parts, gold_output):
        """
        Create the parts relative to grandsibling nodes.

        Each part means that arcs g -> h, h -> m, and h ->s exist at the same
        time.

        :param instance: DependencyInstance
        :param parts: DependencyParts, already pruned
        :param gold_output: either None or a list with binary values indicating
            the presence of each part. If a list, it will be modified in-place.
        :return:
        """
        make_gold = instance.output is not None

        initial_index = len(parts)

        for g in range(len(instance)):
            for h in range(1, len(instance)):
                if g == h:
                    continue

                if 0 > parts.find_arc_index(g, h):
                    # pruned
                    continue

                gold_gh = _check_gold_arc(instance, g, h)

                # check modifiers to the right
                for m in range(h, len(instance)):
                    if h != m and 0 > parts.find_arc_index(h, m):
                        # pruned; h == m signals first child
                        continue

                    gold_hm = m == h or _check_gold_arc(instance, h, m)
                    arc_between = False

                    for s in range(m + 1, len(instance) + 1):
                        if s < len(instance) and 0 > parts.find_arc_index(h, s):
                            # pruned; s == len signals last child
                            continue

                        gold_hs = s == len(instance) or \
                            _check_gold_arc(instance, h, s)

                        part = GrandSibling(h, m, g, s)
                        parts.append(part)

                        if make_gold:
                            value = 0
                            if gold_hm and gold_hs and not arc_between:
                                if gold_gh:
                                    value = 1

                                arc_between = True

                            gold_output.append(value)

                # check modifiers to the left
                for m in range(h, 0, -1):
                    if h != m and 0 > parts.find_arc_index(h, m):
                        # pruned; h == m signals last child
                        continue

                    gold_hm = m == h or _check_gold_arc(instance, h, m)
                    arc_between = False

                    for s in range(m - 1, -1, -1):
                        if s != 0 and 0 > parts.find_arc_index(h, s):
                            # pruned out
                            continue

                        gold_hs = s == 0 or _check_gold_arc(instance, h, s)
                        part = GrandSibling(h, m, g, s)
                        parts.append(part)

                        if make_gold:
                            value = 0
                            if gold_hm and gold_hs and not arc_between:
                                if gold_gh:
                                    value = 1

                                arc_between = True

                            gold_output.append(value)

        parts.set_offset(GrandSibling, initial_index,
                         len(parts) - initial_index)

    def make_parts_grandparent(self, instance, parts, gold_output):
        """
        Create the parts relative to grandparents.

        Each part means that an arc h -> m and g -> h exist at the same time.

        :param instance: DependencyInstance
        :param parts: a DependencyParts object. It must already have been
            pruned.
        :type parts: DependencyParts
        :param gold_output: either None or a list with binary values indicating
            the presence of each part. If a list, it will be modified in-place.1
        :return: if `instances` have the attribute output, return a numpy array
            with the gold output. If it doesn't, return None.
        """
        make_gold = instance.output is not None

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
                    if h == m:
                        # g == m is necessary to run the grandparent factor
                        continue

                    if 0 > parts.find_arc_index(h, m):
                        # pruned out
                        continue

                    arc = Grandparent(h, m, g)
                    parts.append(arc)
                    if make_gold:
                        if gold_gh and instance.get_head(m) == h:
                            gold_output.append(1)
                        else:
                            gold_output.append(0)

        parts.set_offset(Grandparent,
                         initial_index, len(parts) - initial_index)

    def make_parts_labeled(self, instance, parts, gold_output):
        """
        Create labeled arcs. This function expects that `make_parts_basic` has
        already been called and populated `parts` with unlabeled arcs.

        :param instance: DependencyInstance
        :param parts: DependencyParts
        :param gold_output: list to be modified in-place
        """
        make_gold = instance.output is not None

        num_parts_initial = len(parts)
        for h in range(len(instance)):
            for m in range(1, len(instance)):
                if h == m:
                    continue

                # If no unlabeled arc is there, just skip it.
                # This happens if that arc was pruned out.
                if 0 > parts.find_arc_index(h, m):
                    continue

                # determine which relations are allowed between h and m
                modifier_tag = instance.get_tag(m)
                head_tag = instance.get_tag(h)
                allowed_relations = self.dictionary.get_existing_relations(
                    modifier_tag, head_tag)

                # If there is no allowed relation for this arc, but the
                # unlabeled arc was added, then it was forced to be present
                # to maintain connectivity of the graph. In that case (which
                # should be pretty rare) consider all the possible
                # relations.
                if not allowed_relations:
                    allowed_relations = range(len(
                        self.dictionary.get_relation_alphabet()))
                for l in allowed_relations:
                    part = LabeledArc(h, m, l)
                    parts.append(part)
                    if make_gold:
                        if instance.get_head(m) == h and \
                           instance.get_relation(m) == l:
                            gold_output.append(1.)
                        else:
                            gold_output.append(0.)

        parts.set_offset(LabeledArc,
                         num_parts_initial, len(parts) - num_parts_initial)

    def make_parts_basic(self, instance, parts, gold_output):
        """
        Create the first-order arcs into which the problem is factored.

        The objects `parts` is modified in-place.

        :param instance: a DependencyInstance object
        :param parts: a DependencyParts object, modified in-place
        :param gold_output: either None or a list with binary values indicating
            the presence of each part.
        :return: if `instances` have the attribute `output`, return a numpy
            array with 1 signaling the presence of an arc in a combination
            (head, modifier) or (head, modifier, label) and 0 otherwise.
            It is a one-dimensional array.

            If `instances` doesn't have the attribute `output` return None.
        """
        make_gold = instance.output is not None

        num_parts_initial = len(parts)
        for h in range(len(instance)):
            for m in range(1, len(instance)):
                if h == m:
                    continue

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
                    if not allowed_relations:
                        continue

                part = Arc(h, m)
                parts.append(part)
                if make_gold:
                    if instance.get_head(m) == h:
                        gold_output.append(1.)
                    else:
                        gold_output.append(0.)

        # When adding unlabeled arcs, make sure the graph stays connected.
        # Otherwise, enforce connectedness by adding some extra arcs
        # that connect words to the root.
        # NOTE: if --projective, enforcing connectedness is not enough,
        # so we add arcs of the form m-1 -> m to make sure the sentence
        # has a projective parse.
        arcs = parts[num_parts_initial:]
        inserted_arcs = self.enforce_well_formed_graph(instance, arcs)
        for h, m in inserted_arcs:
            part = Arc(h, m)
            parts.append(part)
            if make_gold:
                if instance.get_head(m) == h:
                    gold_output.append(1.)
                else:
                    gold_output.append(0.)
        parts.set_offset(Arc,
                         num_parts_initial, len(parts) - num_parts_initial)

    def _get_task_validation_metrics(self, valid_data, valid_pred):
        """
        Compute and store internally validation UAS.

        :param valid_data: InstanceData
        :param valid_pred: Predicted output
        """
        accumulated_uas = 0.
        total_tokens = 0

        for i in range(len(valid_data)):
            instance = valid_data.instances[i]
            parts = valid_data.parts[i]
            gold = valid_data.gold_labels[i]

            # exclude root
            length = len(instance) - 1
            uas = get_naive_uas(valid_pred[i], gold, parts, length)
            accumulated_uas += uas * length
            total_tokens += length

        self.validation_uas = accumulated_uas / total_tokens
        if self.validation_uas > self.best_validation_uas:
            self.best_validation_uas = self.validation_uas
            self._should_save = True
        else:
            self._should_save = False

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
            assert type(arcs[r]) == Arc
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
        offset, size = parts.get_offset(Arc)
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
        root = -1
        root_score = -1
        removed_roots = 0

        offset, size = parts.get_offset(Arc)
        arcs = parts[offset:offset + size]
        scores = output[offset:offset + size]
        score_matrix = _make_score_matrix(len(instance), arcs, scores)
        heads = chu_liu_edmonds(score_matrix)

        for m, h in enumerate(heads):
            instance.output.heads[m] = h
            if h == 0:
                index = parts.find_arc_index(h, m)
                score = output[index]

                if self.options.single_root and root != -1:
                    removed_roots += 1
                    if score > root_score:
                        # this token is better scored for root
                        # attach the previous root candidate to it
                        instance.output.heads[root] = m
                        root = m
                        root_score = score
                    else:
                        # attach it to the other root
                        instance.output.heads[m] = root
                else:
                    root = m
                    root_score = score

            if not self.options.unlabeled:
                index = parts.find_arc_index(h, m) - offset
                label = parts.best_labels[index]
                label_name = self.dictionary.get_relation_name(label)
                instance.output.relations[m] = label_name

        if removed_roots > 0:
            logging.info('%d tokens reassigned to root' % removed_roots)

        # assign words without heads to the root word
        for m in range(1, len(instance)):
            if instance.get_head(m) < 0:
                logging.info('Word without head.')
                instance.output.heads[m] = root
                if not self.options.unlabeled:
                    instance.output.relations[m] = \
                        self.dictionary.get_relation_name(0)


def _make_score_matrix(length, arcs, scores):
    """
    Makes a score matrix from an array of scores ordered in the same way as a
    list of DependencyPartArcs. Positions [h, m] corresponding to non-existing
    arcs have score of -inf.
    """
    score_matrix = np.full([length, length], -np.inf, np.float32)
    for arc, score in zip(arcs, scores):
        h = arc.head
        m = arc.modifier
        score_matrix[h, m] = score

    return score_matrix


def chu_liu_edmonds(score_matrix):
    """
    Run the Chu-Liu-Edmonds' algorithm to find the maximum spanning tree.

    :param score_matrix: a matrix such that cell [h, m] has the score for the
        arc (h, m).
    """
    while True:
        # pick the highest score head for each modifier
        heads = score_matrix.argmax(0)

        # find and solve cycles
        cycle = find_cycle(heads)

        if cycle is None:
            break
        solve_cycle(heads, cycle, score_matrix)

    # set the head of the root pseudo token to -1
    heads[0] = -1

    return heads


def find_cycle(heads):
    """
    Finds and returns the first cycle in the given list of heads, or None if
    there is no cycle.

    :param heads: candidate heads for each word in a sentence; i.e., heads[i]
        contains the head for modifier i. heads[0] is ignored (0 is the root).
    :return:
    """
    # this set stores all vertices with a valid path to the root
    reachable_vertices = {0}

    # vertices known to be unreachable from the root, i.e., in a cycle
    vertices_in_cycles = set()

    # vertices currently being evaluated, not known if they're reachable
    visited = set()

    # the directions of the edges don't matter if we only want to find cycles
    for vertex in range(len(heads)):
        if vertex in reachable_vertices or vertex in vertices_in_cycles:
            continue

        cycle = _find_cycle_recursive(heads, vertex, visited,
                                      reachable_vertices, vertices_in_cycles)
        if cycle is not None:
            return cycle

    return None


def _find_cycle_recursive(heads, token, visited_tokens, reachable_tokens,
                          tokens_in_cycles):
    """
    Return the first cycle it finds starting from the given token.

    :param heads: heads for each token; heads[i] has the head of token i
    :param token: which token is being currently visited
    :param visited_tokens: set of tokens already visited in this round of
        recursive calls
    :param reachable_tokens: set of tokens known to be reachable from the root
    :param tokens_in_cycles: set of tokens known to be unreachable from the
        root
    :return:
    """
    next_token = heads[token]
    visited_tokens.add(token)

    if next_token in reachable_tokens:
        reachable_tokens.update(visited_tokens)
        visited_tokens.clear()
        cycle = None

    elif next_token in visited_tokens:
        # we found a cycle. return the tokens that are part of it.
        visited_tokens.clear()
        cycle = {token}
        while next_token != token:
            cycle.add(next_token)
            next_token = heads[next_token]

        tokens_in_cycles.update(cycle)

    elif next_token in tokens_in_cycles:
        # vertex linked to an existing cycle, but not part of it
        visited_tokens.clear()
        cycle = None

    else:
        # we still don't know if it's reachable or not, continue exploring
        cycle = _find_cycle_recursive(heads, next_token, visited_tokens,
                                      reachable_tokens, tokens_in_cycles)

    return cycle


def solve_cycle(heads, cycle, scores):
    """
    Resolve a cycle in the dependency tree.

    :param heads: heads for each token; heads[i] has the head of token i
    :param cycle: set of tokens which make up a cycle
    :param scores: 2d numpy array with the scores for each [h, m] arc
    :return:
    """
    num_tokens = len(heads)

    # tokens outside this cycle. 0 (root) is always outside
    tokens_outside = np.array([x for x in range(num_tokens) if x not in cycle])

    cycle = np.array(list(cycle))

    # first, pick an arc from the cycle to the outside tokens
    # if len(outside) == 1, all vertices except for the root are in a cycle
    if len(tokens_outside) > 1:
        # make an Nx1 index array
        cycle_inds = np.array([[i] for i in cycle])

        # these are the weights of arcs connecting tokens in the cycle to the
        # ones outside it. (-1 because we can't take the root now)
        outgoing_weights = scores[cycle_inds, tokens_outside[1:]]

        # find one outgoing arc for each token outside the cycle
        max_outgoing_inds = outgoing_weights.argmax(0)
        max_outgoing_weights = outgoing_weights.max(0)

        # set every outgoing weight to -inf and then restore the highest ones
        outgoing_weights[:] = -np.Infinity
        inds_y = np.arange(len(tokens_outside) - 1)
        outgoing_weights[max_outgoing_inds, inds_y] = max_outgoing_weights
        scores[cycle_inds, tokens_outside[1:]] = outgoing_weights

    # now, adjust incoming arcs. Each arc from a token v (outside the cycle)
    # to v' (inside) is reweighted as:
    # s(v, v') = s(v, v') + s(head(v'), v')
    # and then we pick the highest arc for each outside token
    token_inds = np.array([[i] for i in tokens_outside])
    incoming_weights = scores[token_inds, cycle]

    for i, token in enumerate(cycle):
        head_to_t = scores[heads[token], token]
        incoming_weights[:, i] += head_to_t

    max_incoming_inds = incoming_weights.argmax(1)
    max_incoming_weights = incoming_weights.max(1)

    # analogous to the outgoing weights
    incoming_weights[:] = -np.Infinity
    incoming_weights[np.arange(len(tokens_outside)),
                     max_incoming_inds] = max_incoming_weights
    scores[token_inds, cycle] = incoming_weights

    # the token with the maximum weighted incoming edge now changes
    # its head, thus breaking the cycle
    new_head_ind = max_incoming_weights.argmax()
    token_leaving_cycle_ind = max_incoming_inds[new_head_ind]

    token_leaving_cycle = cycle[token_leaving_cycle_ind]
    old_head = heads[token_leaving_cycle]
    scores[old_head, token_leaving_cycle] = -np.Infinity


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


def get_naive_uas(predicted, gold_output, parts, length):
    """
    Compute the UAS (unlabeled accuracy score) naively, without considering tree
    constraints. It just takes the highest scoring head for each token.

    :param predicted: numpy array with scores for each part (can also be the
        decoder output)
    :param gold_output: same as predicted, with gold data. This
    :param parts: DependencyParts
    :param length: length of the sentence, excluding root
    :return:
    """
    # length + 1 to match the modifier indices
    head_per_modifier = np.zeros(length + 1, np.int)
    max_per_modifier = np.zeros(length + 1, np.float)
    hits = np.zeros(length + 1, np.float)

    for i, arc in parts.iterate_over_type(Arc, return_index=True):
        m = arc.modifier
        score = predicted[i]
        if score < max_per_modifier[m]:
            continue

        max_per_modifier[m] = score
        head_per_modifier[m] = arc.head
        if gold_output[i] == 1:
            hits[m] = 1
        else:
            hits[m] = 0

    # exclude root
    uas = hits[1:].mean()
    return uas
