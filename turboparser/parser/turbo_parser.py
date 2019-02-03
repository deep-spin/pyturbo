from ..classifier.structured_classifier import StructuredClassifier
from ..classifier import utils
from ..classifier.instance import InstanceData
from .dependency_reader import DependencyReader
from .dependency_writer import DependencyWriter
from .dependency_decoder import DependencyDecoder, chu_liu_edmonds, \
    make_score_matrix
from .dependency_scorer import DependencyNeuralScorer
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
from sklearn.metrics import f1_score


class ModelType(object):
    """Dummy class to store the types of parts used by a parser"""
    def __init__(self, type_string):
        """
        :param type_string: a string encoding multiple types of parts:
            af: arc factored (always used)
            cs: consecutive siblings
            gp: grandparents
            as: arbitrary siblings
            hb: head bigrams
            gs: grandsiblings
            ts: trisiblings

            More than one type must be concatenated by +, e.g., af+cs+gp
        """
        codes = type_string.lower().split('+')
        self.consecutive_siblings = 'cs' in codes
        self.grandparents = 'gp' in codes
        self.grandsiblings = 'gs' in codes
        self.arbitrary_siblings = 'as' in codes
        self.head_bigrams = 'hb' in codes
        self.trisiblings = 'ts' in codes


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
        self.structured_target = 'dependency'
        self._set_options()

        if options.neural:
            self.neural_scorer = DependencyNeuralScorer()

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
                    self.model_type,
                    self.token_dictionary, self.dictionary, embeddings,
                    char_embedding_size=self.options.char_embedding_size,
                    tag_embedding_size=self.options.tag_embedding_size,
                    distance_embedding_size=self.options.
                    distance_embedding_size,
                    rnn_size=self.options.rnn_size,
                    mlp_size=self.options.mlp_size,
                    label_mlp_size=self.options.label_mlp_size,
                    rnn_layers=self.options.rnn_layers,
                    mlp_layers=self.options.mlp_layers,
                    dropout=self.options.dropout,
                    word_dropout=options.word_dropout,
                    tag_dropout=options.tag_dropout,
                    tag_mlp_size=options.tag_mlp_size,
                    predict_upos=options.predict_upos,
                    predict_xpos=options.predict_xpos,
                    predict_morph=options.predict_morph)

                print('Model summary:')
                print(model)

                self.neural_scorer.initialize(
                    model, self.options.learning_rate, options.decay,
                    options.beta1, options.beta2)

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
        self.model_type = ModelType(self.options.model_type)
        self.has_pruner = bool(self.options.pruner_path)

        if self.has_pruner:
            self.pruner = self.load_pruner(self.options.pruner_path)
        else:
            self.pruner = None

        self.additional_targets = []
        if self.options.predict_morph:
            self.additional_targets.append('morph')
        if self.options.predict_upos:
            self.additional_targets.append('upos')
        if self.options.predict_xpos:
            self.additional_targets.append('xpos')

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
                pickle.dump(self.neural_scorer.model.char_embedding_size, f)
                pickle.dump(self.neural_scorer.model.tag_embedding_size, f)
                pickle.dump(self.neural_scorer.model.distance_embedding_size, f)
                pickle.dump(self.neural_scorer.model.rnn_size, f)
                pickle.dump(self.neural_scorer.model.mlp_size, f)
                pickle.dump(self.neural_scorer.model.tag_mlp_size, f)
                pickle.dump(self.neural_scorer.model.label_mlp_size, f)
                pickle.dump(self.neural_scorer.model.rnn_layers, f)
                pickle.dump(self.neural_scorer.model.mlp_layers, f)
                pickle.dump(self.neural_scorer.model.dropout_rate, f)
                pickle.dump(self.neural_scorer.model.word_dropout_rate, f)
                pickle.dump(self.neural_scorer.model.tag_dropout_rate, f)
                pickle.dump(self.neural_scorer.model.predict_upos, f)
                pickle.dump(self.neural_scorer.model.predict_xpos, f)
                pickle.dump(self.neural_scorer.model.predict_morph, f)
                self.neural_scorer.model.save(f)

    def load(self, model_path=None):
        '''Load the full configuration and model.'''
        if not model_path:
            model_path = self.options.model_path
        with open(model_path, 'rb') as f:
            model_options = pickle.load(f)

            self.options.neural = model_options.neural
            self.options.model_type = model_options.model_type
            self.options.unlabeled = model_options.unlabeled
            self.options.projective = model_options.projective
            self.options.predict_tags = model_options.predict_tags

            # prune arcs with label/head POS/modifier POS unseen in training
            self.options.prune_relations = model_options.prune_relations

            # prune arcs with a distance unseen with the given POS tags
            self.options.prune_distances = model_options.prune_distances

            # threshold for the basic pruner, if used
            self.options.pruner_posterior_threshold = \
                model_options.pruner_posterior_threshold

            # maximum candidate heads per word in the basic pruner, if used
            self.options.pruner_max_heads = model_options.pruner_max_heads
            self._set_options()

            self.token_dictionary.load(f)
            self.dictionary.load(f)
            self.parameters = pickle.load(f)
            if model_options.neural:
                embedding_vocab_size = pickle.load(f)
                word_embedding_size = pickle.load(f)
                char_embedding_size = pickle.load(f)
                tag_embedding_size = pickle.load(f)
                distance_embedding_size = pickle.load(f)
                rnn_size = pickle.load(f)
                mlp_size = pickle.load(f)
                tag_mlp_size = pickle.load(f)
                label_mlp_size = pickle.load(f)
                rnn_layers = pickle.load(f)
                mlp_layers = pickle.load(f)
                dropout = pickle.load(f)
                word_dropout = pickle.load(f)
                tag_dropout = pickle.load(f)
                predict_upos = pickle.load(f)
                predict_xpos = pickle.load(f)
                predict_morph = pickle.load(f)
                dummy_embeddings = np.empty([embedding_vocab_size,
                                             word_embedding_size], np.float32)
                neural_model = DependencyNeuralModel(
                    self.model_type,
                    self.token_dictionary,
                    self.dictionary, dummy_embeddings,
                    char_embedding_size,
                    tag_embedding_size=tag_embedding_size,
                    distance_embedding_size=distance_embedding_size,
                    rnn_size=rnn_size,
                    mlp_size=mlp_size,
                    tag_mlp_size=tag_mlp_size,
                    label_mlp_size=label_mlp_size,
                    rnn_layers=rnn_layers,
                    mlp_layers=mlp_layers,
                    dropout=dropout,
                    word_dropout=word_dropout, tag_dropout=tag_dropout,
                    predict_upos=predict_upos, predict_xpos=predict_xpos,
                    predict_morph=predict_morph)
                neural_model.load(f)
                self.neural_scorer = DependencyNeuralScorer()
                self.neural_scorer.set_model(neural_model)

                print('Model summary:')
                print(neural_model)

        # most of the time, we load a model to run its predictions
        self.eval_mode()

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
        self.best_validation_uas = 0.
        self.best_validation_las = 0.
        self._should_save = False

    def _reset_task_metrics(self):
        """
        Reset the accumulated UAS counter
        """
        self.accumulated_uas = 0.
        self.accumulated_las = 0.
        self.accumulated_upos = 0
        self.accumulated_xpos = 0
        self.accumulated_morph = 0
        self.total_tokens = 0
        self.validation_uas = 0.
        self.validation_las = 0.

    def _get_task_train_report(self):
        """
        Return task-specific metrics on training data.

        :return: a string describing naive UAS on training data
        """
        uas = self.accumulated_uas / self.total_tokens
        msg = 'Naive train UAS: %f' % uas
        if not self.options.unlabeled:
            las = self.accumulated_las / self.total_tokens
            msg += '\tNaive train LAS: %f' % las

        if self.options.predict_upos:
            acc = self.accumulated_upos / self.total_tokens
            msg += '\tUPOS train acc: %f' % acc
        if self.options.predict_xpos:
            acc = self.accumulated_xpos / self.total_tokens
            msg += '\tXPOS train acc: %f' % acc
        if self.options.predict_morph:
            acc =  self.accumulated_morph / self.total_tokens
            msg += '\tUFeats train acc: %f' % acc

        return msg

    def _get_task_valid_report(self):
        """
        Return task-specific metrics on validation data.

        :return: a string describing naive UAS on validation data
        """
        msg = 'Naive validation UAS: %f' % self.validation_uas
        if not self.options.unlabeled:
            msg += '\tNaive validation LAS: %f' % self.validation_las
        if self.options.predict_upos:
            msg += '\tUPOS accuracy: %f' % self.validation_upos
        if self.options.predict_xpos:
            msg += '\tXPOS accuracy: %f' % self.validation_xpos
        if self.options.predict_morph:
            msg += '\tUFeats accuracy: %f' % self.validation_morph

        return msg

    def _get_post_train_report(self):
        """
        Return the best parsing accuracy.
        """
        msg = 'Best validation UAS: %f' % self.best_validation_uas
        if not self.options.unlabeled:
            msg += '\tBest validation LAS: %f' % self.validation_las

        return msg

    def get_gold_labels(self, instance):
        """
        Return a list of dictionary mapping the name of each target to a numpy
        vector with the gold values.

        :param instance: DependencyInstanceNumeric
        :return: dict
        """
        gold_dict = {}
        if self.options.predict_upos:
            gold_dict['upos'] = instance.get_all_upos()
        if self.options.predict_xpos:
            gold_dict['xpos'] = instance.get_all_xpos()
        if self.options.predict_morph:
            gold_dict['morph'] = instance.get_all_morph_singletons()
        return gold_dict

    def decode_unstructured(self, scores):
        """
        Decode tag labes for a list of instances.

        :return: a dictionary mapping the name of each tag to a vector of
            predictions
        """
        predictions = {}

        # iterate over upos, xpos, morph
        for target_name in self.additional_targets:
            # target_scores is (batch, length, num_classes)
            target_scores = scores[target_name]
            predictions[target_name] = target_scores.argmax(-1)

        return predictions

    def _decode_unstructured_train(self, instances, scores, gold_labels):
        """
        Decode tag labes for a list of instances.

        :return: a dictionary mapping the name of each tag to a vector of
            predictions
        """
        return self.decode_unstructured(scores)

    def _update_task_metrics(self, predicted_parts, instance, scores, parts,
                             gold_parts, gold_labels):
        """
        Update the accumulated UAS, LAS and other targets count.

        It sums the metrics for each
        sentence scaled by its number of tokens; when reporting performance,
        this value is divided by the total number of tokens seen in all
        sentences combined.

        :type predicted_parts: list
        :type scores: dict
        """
        # UAS doesn't consider the root
        length = len(instance) - 1
        uas, las = get_naive_metrics(predicted_parts, gold_parts, parts,
                                     length)

        predicted_labels = self.decode_unstructured(scores)
        if self.options.predict_upos:
            gold = gold_labels['upos']
            pred = predicted_labels['upos']
            hits = (gold == pred)[:len(gold)]
            self.accumulated_upos += hits

        if self.options.predict_xpos:
            gold = gold_labels['xpos']
            pred = predicted_labels['xpos']
            hits = (gold == pred)[:len(gold)]
            self.accumulated_xpos += hits

        if self.options.predict_morph:
            gold = gold_labels['morph']
            pred = predicted_labels['morph']
            hits = (gold == pred)[:len(gold)]
            self.accumulated_morph += hits

        self.accumulated_uas += length * uas
        self.accumulated_las += length * las
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
        return DependencyInstanceNumeric(instance, self.token_dictionary,
                                         self.dictionary)

    def prune(self, instance, parts):
        """
        Prune out some arcs with the pruner model.

        :param instance: a DependencyInstance object, not formatted
        :param parts: a DependencyParts object with arcs
        :type parts:DependencyParts
        :return: a new DependencyParts object contained the kept arcs
        """
        instance = self.pruner.format_instance(instance)
        scores = self.pruner.compute_scores(instance, parts)[0]
        new_parts = self.decoder.decode_matrix_tree(
            len(instance), parts.arc_index, parts, scores,
            self.options.pruner_max_heads,
            self.options.pruner_posterior_threshold)

        if self.options.train:
            for m in range(1, len(instance)):
                h = instance.output.heads[m]
                if new_parts.find_arc_index(h, m) < 0:
                    new_parts.append(Arc(h, m), 1)

                    # accumulate pruner mistakes here instead of later, because
                    # it is simpler to keep `parts` with only arcs
                    self.pruner_mistakes += 1

                    # also add all labels if doing labeled parsing
                    if not self.options.unlabeled:
                        for label in range(self.dictionary.get_num_labels()):
                            if instance.output.relations[m] == label:
                                gold = 1
                            else:
                                gold = 0
                            new_parts.append(LabeledArc(h, m, label), gold)

        return new_parts

    def _report_make_parts(self, instances, parts):
        """
        Log some statistics about the calls to make parts in a dataset.

        :type instances: list[DependencyInstance]
        :type parts: list[DependencyParts]
        """
        num_arcs = 0
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
                            self.pruner_mistakes += 1
                        continue

                    num_arcs += 1

        msg = '%f heads per token after pruning' % (num_arcs / num_tokens)
        logging.info(msg)

        msg = '%d arcs after pruning, out of %d possible (%f)' % \
              (num_arcs, num_possible_arcs, num_arcs / num_possible_arcs)
        logging.info(msg)

        if output_available:
            ratio = (num_tokens - self.pruner_mistakes) / num_tokens
            msg = 'Pruner recall (gold arcs retained after pruning): %f' % ratio
            logging.info(msg)

    def get_parts_scores(self, score_dict):
        """
        Return the scores of the structured parts inside the score dictionary.

        These are the scores for the parts of the parse tree.
        """
        return score_dict['dependency']

    def make_gradient_step(self, gold_parts, predicted_parts,
                           gold_additional_labels=None,
                           parts=None, features=None, eta=None, t=None,
                           instances=None):
        """
        Perform a gradient step minimizing both parsing error and all tagging
        errors.

        The inputs are a batch.
        """
        self.neural_scorer.compute_tag_gradients(gold_additional_labels)
        super(TurboParser, self).make_gradient_step(
            gold_parts, predicted_parts, None, parts, features, eta, t,
            instances)

    def create_gold_targets(self, instance):
        """
        Create the gold targets of an instance that do not depend on parts.

        This will create targets for POS tagging and morphological tags, if
        used.

        :param instance: a formated instance
        :type instance: DependencyInstanceNumeric
        :return: numpy array
        """
        targets = {}
        if self.options.predict_upos:
            targets['upos'] = np.array([instance.get_all_upos()])
        if self.options.predict_xpos:
            targets['xpos'] = np.array([instance.get_all_xpos()])
        if self.options.predict_morph:
            # TODO: combine singleton morph tags (containing all morph
            # information) with separate tags
            targets['morph'] = np.array([instance.get_all_morph_singletons()])

        return targets

    def make_parts(self, instance):
        """
        Create the parts (arcs) into which the problem is factored.

        :param instance: a DependencyInstance object, not yet formatted.
        :return: a tuple (instance, parts).
            The returned instance will have been formatted.
        """
        parts = DependencyParts()
        self.pruner_mistakes = 0

        orig_instance = instance
        instance = self.format_instance(instance)

        self.make_parts_basic(instance, parts)
        if not self.options.unlabeled:
            self.make_parts_labeled(instance, parts)

        if self.has_pruner:
            parts = self.prune(orig_instance, parts)

        if self.model_type.consecutive_siblings:
            self.make_parts_consecutive_siblings(instance, parts)
        if self.model_type.grandparents:
            self.make_parts_grandparent(instance, parts)

        if self.model_type.grandsiblings:
            self.make_parts_grandsibling(instance, parts)

        return instance, parts

    def make_parts_consecutive_siblings(self, instance, parts):
        """
        Create the parts relative to consecutive siblings.

        Each part means that an arc h -> m and h -> s exist at the same time,
        with both h > m and h > s or both h < m and h < s.

        :param instance: DependencyInstance
        :param parts: a DependencyParts object. It must already have been
            pruned.
        :type parts: DependencyParts
        """
        make_gold = instance.output is not None

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

                    if make_gold:
                        gold_hs = s == len(instance) or \
                                    _check_gold_arc(instance, h, s)

                        if gold_hm and gold_hs and not arc_between:
                            gold = 1
                            arc_between = True
                        else:
                            gold = 0
                    else:
                        gold = None
                    parts.append(NextSibling(h, m, s), gold)

            # siblings to the left of h
            for m in range(h, -1, -1):
                if h != m and 0 > parts.find_arc_index(h, m):
                    # pruned out
                    continue

                gold_hm = m == h or _check_gold_arc(instance, h, m)
                arc_between = False

                # when s = 0, it signals that m encoded the leftmost child
                for s in range(m - 1, -2, -1):
                    if s != -1 and 0 > parts.find_arc_index(h, s):
                        # pruned out
                        continue

                    if make_gold:
                        gold_hs = s == -1 or _check_gold_arc(instance, h, s)

                        if gold_hm and gold_hs and not arc_between:
                            gold = 1
                            arc_between = True
                        else:
                            gold = 0
                    else:
                        gold = None
                    parts.append(NextSibling(h, m, s), gold)

    def make_parts_grandsibling(self, instance, parts):
        """
        Create the parts relative to grandsibling nodes.

        Each part means that arcs g -> h, h -> m, and h ->s exist at the same
        time.

        :param instance: DependencyInstance
        :param parts: DependencyParts, already pruned
        """
        make_gold = instance.output is not None

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

                        if make_gold:
                            gold = 0
                            if gold_hm and gold_hs and not arc_between:
                                if gold_gh:
                                    gold = 1

                                arc_between = True
                        else:
                            gold = None

                        part = GrandSibling(h, m, g, s)
                        parts.append(part, gold)

                # check modifiers to the left
                for m in range(h, 0, -1):
                    if h != m and 0 > parts.find_arc_index(h, m):
                        # pruned; h == m signals last child
                        continue

                    gold_hm = m == h or _check_gold_arc(instance, h, m)
                    arc_between = False

                    for s in range(m - 1, -2, -1):
                        if s != -1 and 0 > parts.find_arc_index(h, s):
                            # pruned out
                            continue

                        gold_hs = s == -1 or _check_gold_arc(instance, h, s)
                        if make_gold:
                            gold = 0
                            if gold_hm and gold_hs and not arc_between:
                                if gold_gh:
                                    gold = 1

                                arc_between = True
                        else:
                            gold = None

                        part = GrandSibling(h, m, g, s)
                        parts.append(part, gold)

    def make_parts_grandparent(self, instance, parts):
        """
        Create the parts relative to grandparents.

        Each part means that an arc h -> m and g -> h exist at the same time.

        :param instance: DependencyInstance
        :param parts: a DependencyParts object. It must already have been
            pruned.
        :type parts: DependencyParts
        """
        make_gold = instance.output is not None

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
                    if make_gold:
                        if gold_gh and instance.get_head(m) == h:
                            gold = 1
                        else:
                            gold = 0
                    else:
                        gold = None
                    parts.append(arc, gold)

    def make_parts_labeled(self, instance, parts):
        """
        Create labeled arcs. This function expects that `make_parts_basic` has
        already been called and populated `parts` with unlabeled arcs.

        :param instance: DependencyInstance
        :param parts: DependencyParts
        :type parts: DependencyParts
        """
        make_gold = instance.output is not None

        for h in range(len(instance)):
            for m in range(1, len(instance)):
                if h == m:
                    continue

                # If no unlabeled arc is there, just skip it.
                # This happens if that arc was pruned out.
                if 0 > parts.find_arc_index(h, m):
                    continue

                # determine which relations are allowed between h and m
                modifier_tag = instance.get_upos(m)
                head_tag = instance.get_upos(h)
                allowed_relations = self.dictionary.get_existing_relations(
                    modifier_tag, head_tag)

                # If there is no allowed relation for this arc, but the
                # unlabeled arc was added, then it was forced to be present
                # to maintain connectivity of the graph. In that case (which
                # should be pretty rare) consider all the possible
                # relations.
                if not allowed_relations:
                    allowed_relations = range(self.dictionary.get_num_labels())
                for l in allowed_relations:
                    part = LabeledArc(h, m, l)

                    if make_gold:
                        if instance.get_head(m) == h and \
                           instance.get_relation(m) == l:
                            gold = 1
                        else:
                            gold = 0
                    else:
                        gold = None

                    parts.append(part, gold)

    def make_parts_basic(self, instance, parts):
        """
        Create the first-order arcs into which the problem is factored.

        All parts except the ones pruned by distance or POS tag combination
        are created (if such pruning methods are used). In higher order models,
        the resulting parts should be further pruned by another parser.

        The objects `parts` is modified in-place.

        :param instance: a DependencyInstance object
        :param parts: a DependencyParts object, modified in-place
        :type parts: DependencyParts
        :return: if `instances` have the attribute `output`, return a numpy
            array with 1 signaling the presence of an arc in a combination
            (head, modifier) or (head, modifier, label) and 0 otherwise.
            It is a one-dimensional array.

            If `instances` doesn't have the attribute `output`, return None.
        """
        make_gold = instance.output is not None

        for h in range(len(instance)):
            for m in range(1, len(instance)):
                if h == m:
                    continue

                if h and self.options.prune_distances:
                    modifier_tag = instance.get_upos(m)
                    head_tag = instance.get_upos(h)
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
                    modifier_tag = instance.get_upos(m)
                    head_tag = instance.get_upos(h)
                    allowed_relations = self.dictionary.get_existing_relations(
                        modifier_tag, head_tag)
                    if not allowed_relations:
                        continue

                part = Arc(h, m)
                if make_gold:
                    if instance.get_head(m) == h:
                        gold = 1
                    else:
                        gold = 0
                else:
                    gold = None

                parts.append(part, gold)

        # When adding unlabeled arcs, make sure the graph stays connected.
        # Otherwise, enforce connectedness by adding some extra arcs
        # that connect words to the root.
        # NOTE: if --projective, enforcing connectedness is not enough,
        # so we add arcs of the form m-1 -> m to make sure the sentence
        # has a projective parse.
        arcs = parts.get_parts_of_type(Arc)
        inserted_arcs = self.enforce_well_formed_graph(instance, arcs)
        for h, m in inserted_arcs:
            part = Arc(h, m)
            if make_gold:
                if instance.get_head(m) == h:
                    gold = 1
                else:
                    gold = 0
            else:
                gold = None

            parts.append(part, gold)

    def compute_loss(self, gold_parts, predicted_parts, scores, gold_labels):
        """
        Compute the loss for a batch of predicted parts and label scores.
        """
        loss = 0
        for i in enumerate(predicted_parts):
            parts = predicted_parts[i]
            gold = gold_parts[i]
            loss += self.decoder.compute_loss(gold, parts,
                                              scores['dependency'][i])

        for target in self.additional_targets:
            target_scores = scores[target]
            gold_labels_target = [x[target] for x in gold_labels]
            loss += self.neural_scorer.compute_tag_loss(target_scores,
                                                        gold_labels_target)

        return loss

    def _get_task_validation_metrics(self, valid_data, valid_pred):
        """
        Compute and store internally validation metrics. Also call the neural
        scorer to update learning rate.

        At least the UAS is computed. Depending on the options, also LAS and
        POS accuracy.

        :param valid_data: InstanceData
        :type valid_data: InstanceData
        :param valid_pred: list with predicted outputs (decoded) for each item
            in the data. Each item may be a tuple with parser and POS output.
        """
        accumulated_uas = 0.
        accumulated_las = 0.
        accumulated_tag_hits = {target: 0.
                                for target in self.additional_targets}
        total_tokens = 0

        def count_tag_hits(gold, predicted, target_name):
            gold_tags = gold_labels[target_name]
            pred_tags = valid_pred['upos'][i]
            hits = gold_tags == pred_tags
            num_hits = np.sum(hits)

        for i in range(len(valid_data)):
            instance = valid_data.instances[i]
            parts = valid_data.parts[i]
            gold_labels = valid_data.gold_labels[i]

            dep_prediction = valid_pred['dependency'][i]
            offset = parts.get_type_offset(Arc)
            num_arcs = parts.get_num_type(Arc)
            arcs = parts.get_parts_of_type(Arc)
            arc_scores = dep_prediction[offset:offset + num_arcs]
            gold_heads = instance.output.heads[1:]

            score_matrix = make_score_matrix(len(instance), arcs, arc_scores)
            pred_heads = chu_liu_edmonds(score_matrix)[1:]
            real_length = len(instance) - 1

            # scale UAS by sentence length; it is normalized later
            head_hits = gold_heads == pred_heads
            accumulated_uas += np.sum(head_hits)
            total_tokens += real_length

            if not self.options.unlabeled:
                gold_labels = instance.output.relations[1:]
                pred_labels = get_predicted_labels(pred_heads, parts)
                label_hits = gold_labels == pred_labels
                label_head_hits = np.logical_and(head_hits, label_hits)
                accumulated_las += np.sum(label_head_hits)

            for target in self.additional_targets:
                target_gold = gold_labels[target]
                target_pred = valid_pred[target][i][:len(target_gold)]
                hits = target_gold == target_pred
                accumulated_tag_hits[target] += np.sum(hits)

        self.validation_uas = accumulated_uas / total_tokens
        self.validation_las = accumulated_las / total_tokens
        if 'upos' in self.additional_targets:
            self.validation_upos = accumulated_tag_hits['upos'] / total_tokens
        if 'xpos' in self.additional_targets:
            self.validation_xpos = accumulated_tag_hits['xpos'] / total_tokens
        if 'morph' in self.additional_targets:
            self.validation_morph = accumulated_tag_hits['morph'] / total_tokens

        # always update UAS; use it as a criterion for saving if no LAS
        if self.validation_uas > self.best_validation_uas:
            self.best_validation_uas = self.validation_uas
            improved_uas = True
        else:
            improved_uas = False

        if self.options.unlabeled:
            acc = self.validation_uas
            self._should_save = improved_uas
        else:
            acc = self.validation_las
            if self.validation_las > self.best_validation_las:
                self.best_validation_las = self.validation_las
                self._should_save = True
            else:
                self._should_save = False

        self.neural_scorer.lr_scheduler_step(acc)

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

    def begin_evaluation(self):
        super(TurboParser, self).begin_evaluation()

    def end_evaluation(self, num_instances):
        #TODO: evaluation that takes into account parsing and POS tagging
        # super(TurboParser, self).end_evaluation(num_instances)
        pass

    def run(self):
        self.reassigned_roots = 0
        super(TurboParser, self).run()

    def _run_report(self, num_instances):
        if self.options.single_root:
            ratio = self.reassigned_roots / num_instances
            msg = '%d reassgined roots (sentence had more than one), %f per ' \
                  'sentence' % (self.reassigned_roots, ratio)
            logging.info(msg)
            
    def label_instance(self, instance, parts, output):
        """
        :type instance: DependencyInstance
        :type parts: DependencyParts
        :param output: array with predictions (decoder output) or tuples of
            predictions (parse and POS)
        :return:
        """
        heads = [-1 for i in range(len(instance))]
        relations = ['NULL' for i in range(len(instance))]
        tags = ['NULL' for i in range(len(instance))]
        instance.output = DependencyInstanceOutput(heads, relations, tags)
        root = -1
        root_score = -1

        if self.options.predict_tags:
            dep_output, pos_output = output
        else:
            dep_output = output

        offset = parts.get_type_offset(Arc)
        num_arcs = parts.get_num_type(Arc)
        arcs = parts.get_parts_of_type(Arc)
        scores = dep_output[offset:offset + num_arcs]
        score_matrix = make_score_matrix(len(instance), arcs, scores)
        heads = chu_liu_edmonds(score_matrix)

        for m, h in enumerate(heads):
            instance.output.heads[m] = h
            if h == 0:
                index = parts.find_arc_index(h, m)
                score = dep_output[index]

                if self.options.single_root and root != -1:
                    self.reassigned_roots += 1
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

            if self.options.predict_tags and m > 0:
                tag = pos_output[m - 1]
                tag_name = self.token_dictionary.\
                    tag_alphabet.get_label_name(tag)
                instance.output.tags[m] = tag_name

        # assign words without heads to the root word
        for m in range(1, len(instance)):
            if instance.get_head(m) < 0:
                logging.info('Word without head.')
                instance.output.heads[m] = root
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


def get_naive_metrics(predicted, gold_output, parts, length):
    """
    Compute the UAS (unlabeled accuracy score) and LAS (labeled accuracy score)
    naively, without considering tree constraints. It just takes the highest
    scoring head and label for each token.

    If the decoder predicted output is given, tree constraints have already been
    imposed, except for single root (which is not necessary for some treebanks).

    :param predicted: numpy array with scores for each part (can also be the
        decoder output)
    :param gold_output: same as predicted, with gold data. This
    :param parts: DependencyParts
    :type parts: DependencyParts
    :param length: length of the sentence, excluding root
    :return: UAS, LAS
    """
    # length + 1 to match the modifier indices
    head_per_modifier = np.zeros(length + 1, np.int)
    head_score_per_modifier = np.zeros(length + 1, np.float)
    label_per_modifier = np.zeros(length + 1, np.float)
    label_score_per_modifier = np.zeros(length + 1, np.float)
    head_hits = np.zeros(length + 1, np.float)
    label_hits = np.zeros(length + 1, np.float)

    offset_arc = parts.get_type_offset(Arc)
    for i, arc in enumerate(parts.iterate_over_type(Arc), offset_arc):
        m = arc.modifier
        score = predicted[i]
        if score < head_score_per_modifier[m]:
            continue

        head_score_per_modifier[m] = score
        head_per_modifier[m] = arc.head
        if gold_output[i] == 1:
            head_hits[m] = 1
        else:
            head_hits[m] = 0

    offset_labeled = parts.get_type_offset(LabeledArc)
    for i, arc in enumerate(parts.iterate_over_type(LabeledArc),
                            offset_labeled):
        m = arc.modifier
        score = predicted[i]
        if score < label_score_per_modifier[m]:
            continue

        label_score_per_modifier[m] = score
        label_per_modifier[m] = arc.label
        if gold_output[i] == 1:
            label_hits[m] = 1
        else:
            label_hits[m] = 0

    # LAS counts modifiers with correct label AND head
    label_head_hits = np.logical_and(head_hits, label_hits)

    # exclude root
    uas = head_hits[1:].mean()
    las = label_head_hits[1:].mean()

    return uas, las


def get_predicted_labels(predicted_heads, parts):
    """
    Get the labels of the dependency relations.

    :param predicted_heads: list or array with the the head of the each word
        (only for real words, i.e., not the root symbol)
    :param parts: DependencyParts object
    :return: numpy integer array with labels
    """
    labels = np.zeros(len(predicted_heads), dtype=np.int)

    for modifier, head in enumerate(predicted_heads, 1):
        i = parts.find_arc_index(head, modifier)
        labels[modifier - 1] = parts.best_labels[i]

    return labels
