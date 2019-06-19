# -*- coding: utf-8 -*-

from ..classifier import utils
from ..classifier.instance import InstanceData
from .token_dictionary import TokenDictionary
from .constants import Target, target2string
from .dependency_reader import read_instances
from .dependency_writer import DependencyWriter
from .dependency_decoder import DependencyDecoder, chu_liu_edmonds, \
    make_score_matrix
from .dependency_parts import DependencyParts
from .dependency_neural_model import DependencyNeuralModel
from .dependency_scorer import DependencyNeuralScorer
from .dependency_instance_numeric import DependencyInstanceNumeric
from .constants import SPECIAL_SYMBOLS

import sys
from collections import defaultdict
import pickle
import numpy as np
import logging
import time
import datetime


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


class TurboParser(object):
    '''Dependency parser.'''
    def __init__(self, options):
        self.options = options
        self.token_dictionary = TokenDictionary()
        self.writer = DependencyWriter()
        self.decoder = DependencyDecoder()
        self.model = None
        self._set_options()
        self.neural_scorer = DependencyNeuralScorer()

        if self.options.train:
            pretrain_words, pretrain_embeddings = self._load_embeddings()
            self.token_dictionary.initialize(
                self.options.training_path, self.options.case_sensitive,
                pretrain_words, char_cutoff=options.char_cutoff,
                form_cutoff=options.form_cutoff,
                lemma_cutoff=options.lemma_cutoff)

            model = DependencyNeuralModel(
                self.model_type,
                self.token_dictionary, pretrain_embeddings,
                char_hidden_size=self.options.char_hidden_size,
                transform_size=self.options.transform_size,
                trainable_word_embedding_size=self.options.embedding_size,
                char_embedding_size=self.options.char_embedding_size,
                tag_embedding_size=self.options.tag_embedding_size,
                distance_embedding_size=self.options.
                distance_embedding_size,
                rnn_size=self.options.rnn_size,
                arc_mlp_size=self.options.arc_mlp_size,
                label_mlp_size=self.options.label_mlp_size,
                ho_mlp_size=self.options.ho_mlp_size,
                rnn_layers=self.options.rnn_layers,
                mlp_layers=self.options.mlp_layers,
                dropout=self.options.dropout,
                word_dropout=options.word_dropout,
                tag_mlp_size=options.tag_mlp_size,
                predict_upos=options.upos,
                predict_xpos=options.xpos,
                predict_morph=options.morph)

            self.neural_scorer.initialize(
                model, self.options.learning_rate, options.decay,
                options.beta1, options.beta2)

            if self.options.verbose:
                print('Model summary:', file=sys.stderr)
                print(model, file=sys.stderr)

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
            words, embeddings = utils.read_embeddings(self.options.embeddings,
                                                      SPECIAL_SYMBOLS)
        else:
            words = None
            embeddings = None

        return words, embeddings

    def _set_options(self):
        """
        Set some parameters of the parser determined from its `options`
        attribute.
        """
        self.model_type = ModelType(self.options.model_type)
        self.has_pruner = bool(self.options.pruner_path)

        if self.has_pruner:
            self.pruner = load_pruner(self.options.pruner_path)
        else:
            self.pruner = None

        self.additional_targets = []
        if self.options.morph:
            self.additional_targets.append(Target.MORPH)
        if self.options.upos:
            self.additional_targets.append(Target.UPOS)
        if self.options.xpos:
            self.additional_targets.append(Target.XPOS)

    def save(self, model_path=None):
        """Save the full configuration and model."""
        if not model_path:
            model_path = self.options.model_path
        with open(model_path, 'wb') as f:
            pickle.dump(self.options, f)
            self.token_dictionary.save(f)
            self.neural_scorer.model.save(f)

    @classmethod
    def load(cls, options):
        """Load the full configuration and model."""
        with open(options.model_path, 'rb') as f:
            loaded_options = pickle.load(f)

            options.model_type = loaded_options.model_type
            options.unlabeled = loaded_options.unlabeled
            options.morph = loaded_options.morph
            options.xpos = loaded_options.xpos
            options.upos = loaded_options.upos

            # prune arcs with label/head POS/modifier POS unseen in training
            options.prune_relations = loaded_options.prune_relations

            # prune arcs with a distance unseen with the given POS tags
            options.prune_tags = loaded_options.prune_tags

            # threshold for the basic pruner, if used
            options.pruner_posterior_threshold = \
                loaded_options.pruner_posterior_threshold

            # maximum candidate heads per word in the basic pruner, if used
            options.pruner_max_heads = loaded_options.pruner_max_heads

            parser = TurboParser(options)
            parser.token_dictionary.load(f)

            model = DependencyNeuralModel.load(f, parser.token_dictionary)

        parser.neural_scorer.set_model(model)

        # most of the time, we load a model to run its predictions
        parser.neural_scorer.eval_mode()

        return parser

    def _reset_best_validation_metric(self):
        """
        Set the best validation UAS score to 0
        """
        self.best_validation_uas = 0.
        self.best_validation_las = 0.
        self._should_save = False

    def get_gold_labels(self, instance):
        """
        Return a list of dictionary mapping the name of each target to a numpy
        vector with the gold values.

        :param instance: DependencyInstanceNumeric
        :return: dict
        """
        gold_dict = {}

        # [1:] to skip root symbol
        if self.options.upos:
            gold_dict[Target.UPOS] = instance.get_all_upos()[1:]
        if self.options.xpos:
            gold_dict[Target.XPOS] = instance.get_all_xpos()[1:]
        if self.options.morph:
            gold_dict[Target.MORPH] = instance.get_all_morph_singletons()[1:]

        gold_dict[Target.HEADS] = instance.get_all_heads()[1:]
        gold_dict[Target.RELATIONS] = instance.get_all_relations()[1:]

        return gold_dict

    def _update_task_metrics(self, predicted_parts, instance, scores, parts,
                             gold_labels):
        """
        Update the accumulated UAS, LAS and other targets count for one
        sentence.

        It sums the metrics for one
        sentence scaled by its number of tokens; when reporting performance,
        this value is divided by the total number of tokens seen in all
        sentences combined.

        :param predicted_parts: predicted parts of one sentence.
        :param scores: dictionary mapping target names to scores
        :type predicted_parts: list
        :type scores: dict
        :param gold_labels: dictionary mapping targets to the gold output
        """
        # UAS doesn't consider the root
        length = len(instance) - 1
        pred_heads, pred_labels = self.decode_predictions(
            None, parts, scores[Target.HEADS], scores[Target.RELATIONS])

        gold_heads = gold_labels[Target.HEADS]
        gold_deprel = gold_labels[Target.RELATIONS]

        head_hits = pred_heads == gold_heads
        uas = np.mean(head_hits)

        if self.options.unlabeled:
            las = 0
        else:
            label_hits = pred_labels == gold_deprel
            las = np.logical_and(head_hits, label_hits).mean()

        for target in self.additional_targets:
            gold = gold_labels[target]
            predicted = scores[target]

            # remove padding
            predicted = predicted[:len(gold)]
            hits = np.sum(gold == predicted)
            self.accumulated_hits[target] += hits

        self.accumulated_uas += length * uas
        self.accumulated_las += length * las
        self.total_tokens += length

    def run_pruner(self, instance):
        """
        Prune out some arcs with the pruner model.

        To use the current model as a pruner, use `prune` instead.

        :param instance: a DependencyInstance object, not formatted
        :return: a boolean 2d array masking arcs. It has shape (n, n) where
            n is the instance length including root. Position (h, m) has True
            if the arc is valid, False otherwise.
            During training, gold arcs always are True.
        """
        new_mask = self.pruner.prune(instance)

        if self.options.train:
            for m in range(1, len(instance)):
                h = instance.heads[m]
                if not new_mask[h, m]:
                    new_mask[h, m] = True
                    self.pruner_mistakes += 1

        return new_mask

    def prune(self, instance):
        """
        Prune out some possible arcs in the given instance.

        This function uses the current model as the pruner; to run an
        encapsulated pruner, use `run_pruner` instead.

        :param instance: a DependencyInstance object, not formatted
        :return: a boolean 2d array masking arcs. It has shape (n, n) where
            n is the instance length including root. Position (h, m) has True
            if the arc is valid, False otherwise.
        """
        instance, parts = self.preprocess_instance(instance)
        scores = self.neural_scorer.compute_scores(instance, parts)[0]
        new_mask = self.decoder.decode_matrix_tree(
            parts, scores, self.options.pruner_max_heads,
            self.options.pruner_posterior_threshold)

        return new_mask

    def _report_make_parts(self, instances, parts):
        """
        Log some statistics about the calls to make parts in a dataset.

        :type instances: list[DependencyInstance]
        :type parts: list[DependencyParts]
        """
        num_arcs = 0
        num_tokens = 0
        num_possible_arcs = 0

        for instance, inst_parts in zip(instances, parts):
            inst_len = len(instance)
            num_tokens += inst_len - 1  # exclude root
            num_possible_arcs += (inst_len - 1) ** 2  # exclude root and self

            # skip the root symbol
            for h in range(inst_len):
                for m in range(1, inst_len):
                    if not inst_parts.arc_mask[h, m]:
                        # pruned
                        if self.options.train and instance.heads[m] == h:
                            self.pruner_mistakes += 1
                        continue

                    num_arcs += 1

        msg = '%f heads per token after pruning' % (num_arcs / num_tokens)
        logging.info(msg)

        msg = '%d arcs after pruning, out of %d possible (%f)' % \
              (num_arcs, num_possible_arcs, num_arcs / num_possible_arcs)
        logging.info(msg)

        if self.options.train:
            ratio = (num_tokens - self.pruner_mistakes) / num_tokens
            msg = 'Pruner recall (gold arcs retained after pruning): %f' % ratio
            logging.info(msg)

    def preprocess_instance(self, instance):
        """
        Create the parts (arcs) into which the problem is factored.

        :param instance: a DependencyInstance object, not yet formatted.
        :return: a tuple (instance, parts).
            The returned instance will have been formatted.
        """
        self.pruner_mistakes = 0

        if self.has_pruner:
            prune_mask = self.run_pruner(instance)
        else:
            prune_mask = None

        instance = DependencyInstanceNumeric(instance, self.token_dictionary,
                                             self.options.case_sensitive)
        num_relations = self.token_dictionary.get_num_deprels()
        labeled = not self.options.unlabeled
        parts = DependencyParts(instance, self.model_type, prune_mask,
                                labeled, num_relations)

        return instance, parts

    def decode_predictions(self, predictions, parts, head_score_matrix=None,
                           label_matrix=None):
        """
        Decode the predicted heads and labels after having running the decoder.

        This function takes care of the cases when the variable assignments by
        the decoder does not produce a valid tree running the Chu-Liu-Edmonds
        algorithm.

        :param predictions: indicator array of predicted dependency parts (with
            values between 0 and 1)
        :param parts: the dependency parts
        :type parts: DependencyParts
        :return: a tuple (pred_heads, pred_labels)
            The first is an array such that position heads[m] contains the head
            for token m; it starts from the first actual word, not the root.
            If the model is not trained for predicting labels, the second item
            is None.
        """
        if head_score_matrix is None:
            length = len(parts.arc_mask)
            arc_scores = predictions[:parts.num_arcs]
            score_matrix = make_score_matrix(length, parts.arc_mask, arc_scores)
        else:
            # TODO: provide the matrix already (n x n)
            zeros = np.zeros_like(head_score_matrix[0]).reshape([1, -1])
            score_matrix = np.concatenate([zeros, head_score_matrix], 0)

        pred_heads = chu_liu_edmonds(score_matrix)
        if self.options.single_root:
            root = -1
            root_score = -1

            for m, h in enumerate(pred_heads[1:], 1):
                if h == 0:
                    # score_matrix is (m, h), starting from 0
                    score = score_matrix[m - 1, h]

                    if root != -1:
                        # we have already found another root before

                        if score > root_score:
                            # this token is better scored for root
                            # attach the previous root candidate to it
                            pred_heads[root] = m
                            parts.add_dummy_relation(m, root)
                            root = m
                            root_score = score

                        else:
                            # attach it to the other root
                            pred_heads[m] = root
                            parts.add_dummy_relation(root, m)

                        self.reassigned_roots += 1
                    else:
                        root = m
                        root_score = score

        pred_heads = pred_heads[1:]
        if parts.labeled:
            if label_matrix is not None:
                pred_labels = []
                for m, h in enumerate(pred_heads):
                    pred_labels.append(label_matrix[m, h])

                pred_labels = np.array(pred_labels)
            else:
                pred_labels = parts.get_labels(pred_heads)
        else:
            pred_labels = None

        return pred_heads, pred_labels

    def compute_validation_metrics(self, valid_data, valid_pred):
        """
        Compute and store internally validation metrics. Also call the neural
        scorer to update learning rate.

        At least the UAS is computed. Depending on the options, also LAS and
        POS accuracy.

        :param valid_data: InstanceData
        :type valid_data: InstanceData
        :param valid_pred: list with predicted outputs (decoded) for each item
            in the data. Each item is a dictionary mapping target names to the
            prediction vectors.
        """
        accumulated_uas = 0.
        accumulated_las = 0.
        accumulated_tag_hits = {target: 0.
                                for target in self.additional_targets}
        total_tokens = 0

        for i in range(len(valid_data)):
            instance = valid_data.instances[i]
            parts = valid_data.parts[i]
            gold_output = valid_data.gold_labels[i]
            inst_pred = valid_pred[i]

            real_length = len(instance) - 1
            # dep_prediction = inst_pred[Target.DEPENDENCY_PARTS]
            gold_heads = gold_output[Target.HEADS]

            head_scores = inst_pred[Target.HEADS][:real_length, :len(instance)]
            deprel_scores = inst_pred[Target.RELATIONS][:real_length, :len(instance)]
            pred_heads, pred_labels = self.decode_predictions(
                None, parts, head_scores, deprel_scores)

            # scale UAS by sentence length; it is normalized later
            head_hits = gold_heads == pred_heads
            accumulated_uas += np.sum(head_hits)
            total_tokens += real_length

            if not self.options.unlabeled:
                deprel_gold = gold_output[Target.RELATIONS]
                label_hits = deprel_gold == pred_labels
                label_head_hits = np.logical_and(head_hits, label_hits)
                accumulated_las += np.sum(label_head_hits)

            for target in self.additional_targets:
                target_gold = gold_output[target]
                target_pred = inst_pred[target][:len(target_gold)]
                hits = target_gold == target_pred
                accumulated_tag_hits[target] += np.sum(hits)

        self.validation_uas = accumulated_uas / total_tokens
        self.validation_las = accumulated_las / total_tokens
        self.validation_accuracies = {}
        for target in self.additional_targets:
            self.validation_accuracies[target] = accumulated_tag_hits[
                                       target] / total_tokens

        # always update UAS; use it as a criterion for saving if no LAS
        if self.validation_uas >= self.best_validation_uas:
            self.best_validation_uas = self.validation_uas
            improved_uas = True
        else:
            improved_uas = False

        if self.options.unlabeled:
            self._should_save = improved_uas
        else:
            if self.validation_las >= self.best_validation_las:
                self.best_validation_las = self.validation_las
                self._should_save = True
            else:
                self._should_save = False

    def enforce_well_formed_graph(self, instance, arcs):
        if self.options.projective:
            raise NotImplementedError
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

    def run(self):
        self.reassigned_roots = 0
        tic = time.time()

        instances = read_instances(self.options.test_path)
        logging.info('Number of instances: %d' % len(instances))
        data = self.preprocess_instances(instances)
        data.prepare_batches(self.options.batch_size, sort=False)
        predictions = []

        for batch in data.batches:
            batch_predictions = self.run_batch(batch)
            predictions.extend(batch_predictions)

        self.write_predictions(instances, data.parts, predictions)
        toc = time.time()
        logging.info('Time: %f' % (toc - tic))

        self._run_report(len(instances))

    def write_predictions(self, instances, parts, predictions):
        """
        Write predictions to a file.

        :param instances: the instances in the original format (i.e., not the
            "formatted" one, but retaining the original contents)
        :param parts: list with the parts per instance
        :param predictions: list with predictions per instance
        """
        self.writer.open(self.options.output_path)
        for instance, inst_parts, inst_prediction in zip(instances,
                                                         parts, predictions):
            self.label_instance(instance, inst_parts, inst_prediction)
            self.writer.write(instance)

        self.writer.close()

    def _run_report(self, num_instances):
        if self.options.single_root:
            ratio = self.reassigned_roots / num_instances
            msg = '%d reassgined roots (sentence had more than one), %f per ' \
                  'sentence' % (self.reassigned_roots, ratio)
            logging.info(msg)

    def read_train_instances(self):
        '''Create batch of training and validation instances.'''
        import time
        tic = time.time()
        logging.info('Creating instances...')

        train_instances = read_instances(self.options.training_path)
        valid_instances = read_instances(self.options.valid_path)
        logging.info('Number of train instances: %d' % len(train_instances))
        logging.info('Number of validation instances: %d'
                     % len(valid_instances))
        toc = time.time()
        logging.info('Time: %f' % (toc - tic))
        return train_instances, valid_instances

    def preprocess_instances(self, instances):
        """
        Create parts for all instances in the batch.

        :param instances: list of non-formatted Instance objects
        :return: an InstanceData object.
            It contains formatted instances.
            In neural models, features is a list of None.
        """
        all_parts = []
        all_gold_labels = []
        formatted_instances = []

        for instance in instances:
            f_instance, parts = self.preprocess_instance(instance)
            gold_labels = self.get_gold_labels(f_instance)

            formatted_instances.append(f_instance)
            all_parts.append(parts)
            all_gold_labels.append(gold_labels)

        self._report_make_parts(instances, all_parts)
        data = InstanceData(formatted_instances, all_parts, all_gold_labels)
        return data

    def reset_performance_metrics(self):
        """
        Reset some variables used to keep track of training performance.
        """
        self.num_train_instances = 0
        self.time_scores = 0
        self.time_decoding = 0
        self.time_gradient = 0
        self.train_losses = {target: 0. for target in self.additional_targets}
        self.train_losses[Target.DEPENDENCY_PARTS] = 0.
        self.accumulated_hits = {}
        for target in self.additional_targets:
            self.accumulated_hits[target] = 0

        self.accumulated_uas = 0.
        self.accumulated_las = 0.
        self.total_tokens = 0
        self.validation_uas = 0.
        self.validation_las = 0.
        self.reassigned_roots = 0

    def train(self):
        '''Train with a general online algorithm.'''
        train_instances, valid_instances = self.read_train_instances()
        train_data = self.preprocess_instances(train_instances)
        valid_data = self.preprocess_instances(valid_instances)
        train_data.prepare_batches(self.options.batch_size, sort=True)
        valid_data.prepare_batches(self.options.batch_size, sort=True)
        logging.info('Training data spread across %d batches\n'
                     % len(train_data.batches))

        self._reset_best_validation_metric()
        self.reset_performance_metrics()
        using_amsgrad = False
        num_bad_evals = 0

        for global_step in range(1, self.options.max_steps + 1):
            batch = train_data.get_next_batch()
            self.train_batch(batch)

            if global_step % self.options.log_interval == 0:
                msg = '%s Step %d' % (datetime.datetime.now(), global_step)
                logging.info(msg)
                self.train_report(self.num_train_instances)
                self.reset_performance_metrics()

            if global_step % self.options.eval_interval == 0:
                self.run_on_validation(valid_data)
                if self._should_save:
                    self.save()
                    num_bad_evals = 0
                else:
                    num_bad_evals += 1
                    self.neural_scorer.decrease_learning_rate()

                if num_bad_evals == self.options.patience:
                    if not using_amsgrad:
                        logging.info('Switching to AMSGrad')
                        using_amsgrad = True
                        self.neural_scorer.switch_to_amsgrad(
                            self.options.learning_rate, self.options.beta1,
                            self.options.beta2)
                        num_bad_evals = 0
                    else:
                        break

            train_data.shuffle_batches()

        msg = 'Best validation UAS: %f' % self.best_validation_uas
        if not self.options.unlabeled:
            msg += '\tBest validation LAS: %f' % self.best_validation_las
        logging.info(msg)

    def run_on_validation(self, valid_data):
        """
        Run the model on validation data
        """
        valid_start = time.time()
        self.neural_scorer.eval_mode()

        predictions = []
        losses = defaultdict(float)
        for batch in valid_data.batches:
            result = self.run_batch(batch, return_loss=True)
            batch_predictions = result[0]
            batch_losses = result[1]
            batch_size = len(batch)
            predictions.extend(batch_predictions)

            for target in batch_losses:
                # store non-normalized losses
                losses[target] += batch_size * batch_losses[target].item()

        self.compute_validation_metrics(valid_data, predictions)

        valid_end = time.time()
        time_validation = valid_end - valid_start

        logging.info('Time to run on validation: %.2f' % time_validation)
        msgs = ['Validation losses:'] + make_loss_msgs(losses,
                                                       len(valid_data))
        logging.info('\t'.join(msgs))

        msgs = ['Validation accuracies:\tUAS: %.4f' % self.validation_uas]
        if not self.options.unlabeled:
            msgs.append('LAS: %.4f' % self.validation_las)
        for target in self.additional_targets:
            target_name = target2string[target]
            acc = self.validation_accuracies[target]
            msgs.append('%s: %.4f' % (target_name, acc))
        logging.info('\t'.join(msgs))

        if self._should_save:
            logging.info('Saved model')

        logging.info('\n')

    def train_report(self, num_instances):
        """
        Log a short report of the training loss.
        """
        msgs = ['Train losses:'] + make_loss_msgs(self.train_losses,
                                                  num_instances)
        logging.info('\t'.join(msgs))

        time_msg = 'Time to score: %.2f\tDecode: %.2f\tGradient step: %.2f'
        time_msg %= (self.time_scores, self.time_decoding, self.time_gradient)
        logging.info(time_msg)

        # uas = self.accumulated_uas / self.total_tokens
        # msgs = ['Train accuracies:\tUAS: %.6f' % uas]
        # if not self.options.unlabeled:
        #     las = self.accumulated_las / self.total_tokens
        #     msgs.append('LAS: %.6f' % las)
        #
        # for target in self.additional_targets:
        #     target_name = target2string[target]
        #     acc = self.accumulated_hits[target] / self.total_tokens
        #     msgs.append('%s: %.6f' % (target_name, acc))
        # logging.info('\t'.join(msgs))

    def run_batch(self, instance_data, return_loss=False):
        """
        Predict the output for the given instances.

        :type instance_data: InstanceData
        :param return_loss: if True, also return the loss (only use if
            instance_data has the gold outputs) as a list of values
        :return: a list of arrays with the predicted outputs if return_loss is
            False. If it's True, a tuple with predictions and losses.
            Each prediction is a dictionary mapping a target name to the
            prediction vector.
        """
        self.neural_scorer.eval_mode()
        scores = self.neural_scorer.compute_scores(instance_data.instances,
                                                   instance_data.parts)
        #
        # predictions = []
        # all_predicted_parts = []
        # for i in range(len(instance_data)):
        #     instance = instance_data.instances[i]
        #     parts = instance_data.parts[i]
        #     inst_scores = scores[i]
        #
        #     predicted_parts = self.decoder.decode(instance, parts, inst_scores)
        #     inst_prediction = {Target.DEPENDENCY_PARTS: predicted_parts}
        #     if return_loss:
        #         all_predicted_parts.append(predicted_parts)
        #     for target in self.additional_targets:
        #         model_answer = inst_scores[target].argmax(-1)
        #         inst_prediction[target] = model_answer
        #
        #     predictions.append(inst_prediction)

        if return_loss:
            losses = self.neural_scorer.compute_loss(instance_data,
                                                     None)
            return scores, losses

        return scores

    def train_batch(self, instance_data):
        '''
        Run one batch of a learning algorithm. If it is an online one, just
        run through each instance.

        :param instance_data: InstanceData object containing the instances of
            the batch
        '''
        self.neural_scorer.train_mode()

        start_time = time.time()
        # scores is a list of dictionaries [target] -> score array
        scores = self.neural_scorer.compute_scores(instance_data.instances,
                                                   instance_data.parts)
        end_time = time.time()
        self.time_scores += end_time - start_time

        all_predicted_parts = []
        for i in range(len(instance_data)):
            instance = instance_data.instances[i]
            parts = instance_data.parts[i]
            gold_labels = instance_data.gold_labels[i]
            inst_scores = scores[i]

            # predicted_parts = self.decode_train(instance, parts, inst_scores)
            # all_predicted_parts.append(predicted_parts)

            self._update_task_metrics(
                None, instance, inst_scores, parts, gold_labels)

        # run the gradient step for the whole batch
        start_time = time.time()
        losses = self.neural_scorer.compute_loss(instance_data,
                                                 all_predicted_parts)
        self.neural_scorer.make_gradient_step(losses)
        batch_size = len(instance_data)
        self.num_train_instances += batch_size
        for target in losses:
            # store non-normalized losses
            self.train_losses[target] += batch_size * losses[target].item()

        end_time = time.time()
        self.time_gradient += end_time - start_time

    def decode_train(self, instance, parts, scores):
        """
        Decode the scores for parsing at training time.

        Return the predicted output (for each part)

        :param instance: a DependencyInstanceNumeric
        :param parts: DependencyParts
        :type parts: DependencyParts
        :param scores: a dictionary mapping target names to scores produced by
            the network
        :return: prediction array
        """
        # Do the decoding.
        start_decoding = time.time()
        predicted_output = self.decoder.decode_cost_augmented(
            instance, parts, scores)

        end_decoding = time.time()
        self.time_decoding += end_decoding - start_decoding

        return predicted_output

    def label_instance(self, instance, parts, output):
        """
        :type instance: DependencyInstance
        :type parts: DependencyParts
        :param output: dictionary mapping target names to predictions
        :return:
        """
        # dep_output = output[Target.DEPENDENCY_PARTS]
        head_scores = output[Target.HEADS][:len(instance) - 1, :len(instance)]
        deprel_scores = output[Target.RELATIONS][:len(instance) - 1,
                                                 :len(instance)]
        heads, relations = self.decode_predictions(
            None, parts, head_scores, deprel_scores)

        for m, h in enumerate(heads, 1):
            instance.heads[m] = h

            if parts.labeled:
                relation = relations[m - 1]
                relation_name = self.token_dictionary.deprel_alphabet.\
                    get_label_name(relation)
                instance.relations[m] = relation_name

            if self.options.upos:
                # -1 because there's no tag for the root
                tag = output[Target.UPOS][m - 1]
                tag_name = self.token_dictionary. \
                    upos_alphabet.get_label_name(tag)
                instance.upos[m] = tag_name
            if self.options.xpos:
                tag = output[Target.XPOS][m - 1]
                tag_name = self.token_dictionary. \
                    xpos_alphabet.get_label_name(tag)
                instance.xpos[m] = tag_name
            if self.options.morph:
                tag = output[Target.MORPH][m - 1]
                tag_name = self.token_dictionary. \
                    morph_singleton_alphabet.get_label_name(tag)
                instance.morph_singletons[m] = tag_name


def load_pruner(model_path):
    """
    Load and return a pruner model.

    This function takes care of keeping the main parser and the pruner
    configurations separate.
    """
    logging.info('Loading pruner from %s' % model_path)
    with open(model_path, 'rb') as f:
        pruner_options = pickle.load(f)

    pruner_options.train = False
    pruner = TurboParser.load(pruner_options)

    return pruner


def make_loss_msgs(losses, dataset_size):
    """
    Return a list of strings in the shape

    NAME: LOSS_VALUE

    :param losses: dictionary mapping targets to loss values
    :param dataset_size: value used to normalize (divide) each loss value
    :return: list of strings
    """
    msgs = []
    for target in losses:
        target_name = target2string[target]
        normalized_loss = losses[target] / dataset_size
        msg = '%s: %.4f' % (target_name, normalized_loss)
        msgs.append(msg)
    return msgs
