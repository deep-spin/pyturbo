# -*- coding: utf-8 -*-

from ..classifier import utils
from ..classifier.instance import InstanceData
from .token_dictionary import TokenDictionary
from .constants import Target, target2string, ParsingObjective as Objective
from .dependency_reader import read_instances
from .dependency_writer import DependencyWriter
from . import decoding
from .dependency_parts import DependencyParts
from .dependency_neural_model import DependencyNeuralModel
from .dependency_scorer import DependencyNeuralScorer
from .dependency_instance_numeric import DependencyInstanceNumeric
from .constants import SPECIAL_SYMBOLS, EOS, EMPTY

from collections import defaultdict
import pickle
import numpy as np
import time
from transformers import BertTokenizer


logger = utils.get_logger()


class TurboParser(object):
    '''Dependency parser.'''
    target_priority = [Target.RELATIONS, Target.HEADS, Target.LEMMA,
                       Target.XPOS, Target.UPOS, Target.MORPH]

    def __init__(self, options):
        self.options = options
        self.token_dictionary = TokenDictionary()
        self.writer = DependencyWriter()
        self.model = None
        self._set_options()
        self.neural_scorer = DependencyNeuralScorer()

        if self.options.train:
            pretrain_words, pretrain_embeddings = self._load_embeddings()
            self.token_dictionary.initialize(
                self.options.training_path, self.options.case_sensitive,
                pretrain_words, char_cutoff=options.char_cutoff,
                morph_cutoff=options.morph_tag_cutoff,
                form_cutoff=options.form_cutoff,
                lemma_cutoff=options.lemma_cutoff)

            model = DependencyNeuralModel(
                self.options.model_type,
                self.token_dictionary, pretrain_embeddings,
                lemma_embedding_size=self.options.lemma_embedding_size,
                char_hidden_size=self.options.char_hidden_size,
                transform_size=self.options.transform_size,
                trainable_word_embedding_size=self.options.embedding_size,
                char_embedding_size=self.options.char_embedding_size,
                tag_embedding_size=self.options.tag_embedding_size,
                rnn_size=self.options.rnn_size,
                arc_mlp_size=self.options.arc_mlp_size,
                label_mlp_size=self.options.label_mlp_size,
                ho_mlp_size=self.options.ho_mlp_size,
                shared_rnn_layers=self.options.rnn_layers,
                dropout=self.options.dropout,
                word_dropout=options.word_dropout,
                tag_mlp_size=options.tag_mlp_size,
                predict_upos=options.upos,
                predict_xpos=options.xpos,
                predict_morph=options.morph,
                predict_lemma=options.lemma,
                predict_tree=options.parse,
                pretrained_name_or_config=options.bert_model)

            self.neural_scorer.initialize(
                model, self.options.parsing_loss,
                self.options.learning_rate, options.decay, options.beta1,
                options.beta2, options.l2)

            if self.options.verbose:
                logger.debug('Model summary:')
                logger.debug(str(model))

    def set_as_pruner(self):
        """Set this model to be used as a pruner"""
        self.options.train = False
        self.options.parsing_loss = Objective.GLOBAL_PROBABILITY
        self.neural_scorer.parsing_loss = Objective.GLOBAL_PROBABILITY

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
        logger.info('Loading embeddings')
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
        self.has_pruner = bool(self.options.pruner_path)

        if self.options.model_type.first_order:
            if self.has_pruner:
                msg = 'Pruner set for arc-factored model. This is probably ' \
                      'not necessary and inefficient'
                logger.warning(msg)
        else:
            if self.options.parsing_loss != Objective.GLOBAL_MARGIN:
                msg = 'Only global margin objective implemented for ' \
                      'higher order models'
                logger.error(msg)
                exit(1)

            if not self.has_pruner:
                msg = 'Running higher-order model without pruner! ' \
                      'Parser may be very slow!'
                logger.warning(msg)

        if self.has_pruner:
            self.pruner = load_pruner(self.options.pruner_path)
            if self.options.pruner_batch_size > 0:
                self.pruner.options.batch_size = self.options.pruner_batch_size
        else:
            self.pruner = None

        self.additional_targets = []
        if self.options.morph:
            self.additional_targets.append(Target.MORPH)
        if self.options.upos:
            self.additional_targets.append(Target.UPOS)
        if self.options.xpos:
            self.additional_targets.append(Target.XPOS)
        if self.options.lemma:
            self.additional_targets.append(Target.LEMMA)

    def save(self, model_path=None):
        """Save the full configuration and model."""
        if not model_path:
            model_path = self.options.model_path

        data = {'options': self.options,
                'dictionary': self.token_dictionary,
                'metadata': self.neural_scorer.model.create_metadata()}

        with open(model_path, 'wb') as f:
            pickle.dump(data, f)
            self.neural_scorer.model.save(f)

    @classmethod
    def load(cls, options=None, path=None):
        """Load the full configuration and model."""
        if path is None:
            path = options.model_path

        with open(path, 'rb') as f:
            data = pickle.load(f)
            loaded_options = data['options']
            token_dictionary = data['dictionary']
            model_metadata = data['metadata']
            model = DependencyNeuralModel.load(
                f, loaded_options, token_dictionary, model_metadata)

        if options is None:
            options = loaded_options
        else:
            options.model_type = loaded_options.model_type
            options.unlabeled = loaded_options.unlabeled
            options.morph = loaded_options.morph
            options.xpos = loaded_options.xpos
            options.upos = loaded_options.upos
            options.lemma = loaded_options.lemma
            options.parse = loaded_options.parse
            options.parsing_loss = loaded_options.parsing_loss
            options.bert_model = loaded_options.bert_model

            # threshold for the basic pruner, if used
            options.pruner_posterior_threshold = \
                loaded_options.pruner_posterior_threshold

            # maximum candidate heads per word in the basic pruner, if used
            options.pruner_max_heads = loaded_options.pruner_max_heads

        options.train = False
        parser = TurboParser(options)
        parser.neural_scorer.set_model(model)
        parser.token_dictionary = token_dictionary
        parser.neural_scorer.parsing_loss = options.parsing_loss

        # most of the time, we load a model to run its predictions
        parser.neural_scorer.eval_mode()

        return parser

    def _reset_best_validation_metric(self):
        """
        Unset the best validation scores
        """
        self.best_metric_value = defaultdict(float)
        self._should_save = False

    #TODO: remove this function and access gold data directly in the instance
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
        if self.options.lemma:
            gold_dict[Target.LEMMA] = instance.lemma_characters[1:]

        if self.options.parse:
            gold_dict[Target.HEADS] = instance.get_all_heads()[1:]
            gold_dict[Target.RELATIONS] = instance.get_all_relations()[1:]

        return gold_dict

    def run_pruner(self, instances):
        """
        Prune out some arcs with the pruner model.

        :param instances: a list of DependencyInstance objects, not formatted
        :return: a list of boolean 2d arrays masking arcs, one for each
            instance. It has shape (n, n) where n is the instance length
            including root. Position (h, m) has True if the arc is valid, False
            otherwise. During training, gold arcs always are True.
        """
        pruner = self.pruner
        instance_data = pruner.preprocess_instances(instances, report=False)
        instance_data.prepare_batches(pruner.options.batch_size, sort=False)
        masks = []
        entropies = []

        for batch in instance_data.batches:
            batch_masks, batch_entropies = self.prune_batch(batch)
            masks.extend(batch_masks)
            entropies.extend(batch_entropies)

        entropies = np.array(entropies)
        logger.info('Pruner mean entropy: %f' % entropies.mean())

        return masks

    def prune_batch(self, instance_data: InstanceData):
        """
        Prune out some possible arcs in the given instances.

        This function runs the encapsulated pruner.

        :param instance_data: a InstanceData object
        :return: a tuple (masks, entropies)
            masks: a list of  boolean 2d array masking arcs. It has shape (n, n)
            where n is the instance length including root. Position (h, m) has
            True if the arc is valid, False otherwise.

            entropies: a list of the tree entropies found by the matrix tree
            theorem
        """
        pruner = self.pruner
        scores = pruner.neural_scorer.predict(instance_data, decode_tree=False)
        masks = []

        # scores is a dictionary mapping [target] -> (batch, scores)
        for i, instance_scores in enumerate(scores):
            marginals = instance_scores[Target.HEADS].T
            new_mask = decoding.generate_arc_mask(
                marginals, self.options.pruner_max_heads,
                self.options.pruner_posterior_threshold)

            if self.options.train:
                # if training, put back any gold arc pruned out
                instance = instance_data.instances[i]
                for m in range(1, len(instance)):
                    h = instance.heads[m]
                    if not new_mask[h, m]:
                        new_mask[h, m] = True
                        self.pruner_mistakes += 1

            masks.append(new_mask)

        entropies = pruner.neural_scorer.entropies
        return masks, entropies

    def _report_make_parts(self, data):
        """
        Log some statistics about the calls to make parts in a dataset.

        :type data: InstanceData
        """
        num_arcs = 0
        num_tokens = 0
        num_possible_arcs = 0
        num_higher_order = defaultdict(int)

        for instance, inst_parts in zip(data.instances, data.parts):
            inst_len = len(instance)
            num_inst_tokens = inst_len - 1  # exclude root
            num_tokens += num_inst_tokens
            num_possible_arcs += num_inst_tokens ** 2

            mask = inst_parts.arc_mask
            num_arcs += mask.sum()

            for part_type in inst_parts.part_lists:
                num_parts = len(inst_parts.part_lists[part_type])
                num_higher_order[part_type] += num_parts

        logger.info('%d tokens in the data' % num_tokens)
        msg = '%d arcs' % num_arcs
        if self.has_pruner:
            msg += ', out of %d possible (%f)' % \
                   (num_possible_arcs, num_arcs / num_possible_arcs)
        logger.info(msg)

        head_to_token = num_arcs / num_tokens
        msg = '%f heads per token' % head_to_token
        if self.has_pruner:
            possible_heads = num_possible_arcs / num_tokens
            msg += ', out of %f possible' % possible_heads
        logger.info(msg)

        for part_type in num_higher_order:
            num = num_higher_order[part_type]
            name = target2string[part_type]
            msg = '%d %s parts' % (num, name)
            logger.info(msg)

        if self.options.train and self.has_pruner:
            ratio = (num_tokens - self.pruner_mistakes) / num_tokens
            msg = 'Pruner recall (gold arcs retained after pruning): %f' % ratio
            msg += '\n%d arcs incorrectly pruned' % self.pruner_mistakes
            logger.info(msg)

    def compute_validation_metrics(self, valid_data, valid_pred):
        """
        Compute and store internally validation metrics. Also call the neural
        scorer to update learning rate.

        At least the UAS is computed. Depending on the options, also LAS and
        tagging accuracy.

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
            gold_output = valid_data.gold_labels[i]
            inst_pred = valid_pred[i]

            real_length = len(instance) - 1
            total_tokens += real_length
            if self.options.parse:
                gold_heads = gold_output[Target.HEADS]
                pred_heads = inst_pred[Target.HEADS]

                # scale UAS by sentence length; it is normalized later
                head_hits = gold_heads == pred_heads
                accumulated_uas += np.sum(head_hits)

                pred_labels = inst_pred[Target.RELATIONS]
                gold_labels = gold_output[Target.RELATIONS]
                label_hits = gold_labels == pred_labels
                label_head_hits = np.logical_and(head_hits, label_hits)
                accumulated_las += np.sum(label_head_hits)

            for target in self.additional_targets:
                if target == Target.LEMMA:
                    # lemma has to match the whole sequence
                    # inst_pred[LEMMA] has a nested list of arrays
                    gold_lemmas = gold_output[Target.LEMMA]
                    pred_lemmas = inst_pred[Target.LEMMA]
                    for gold, pred in zip(gold_lemmas, pred_lemmas):
                        if len(gold) == len(pred) and np.all(gold == pred):
                            accumulated_tag_hits[Target.LEMMA] += 1
                else:
                    target_gold = gold_output[target]
                    target_pred = inst_pred[target]
                    hits = target_gold == target_pred
                    accumulated_tag_hits[target] += np.sum(hits)

        accuracies = {}
        if self.options.parse:
            accuracies[Target.HEADS] = accumulated_uas / total_tokens
            accuracies[Target.RELATIONS] = accumulated_las / total_tokens
        for target in self.additional_targets:
            accuracies[target] = accumulated_tag_hits[target] / total_tokens

        # check if the prioritized target improved accuracy
        for target in self.target_priority:
            if target not in accuracies:
                continue

            current_value = accuracies[target]
            best = self.best_metric_value[target]
            if current_value == best:
                # consider the next top priority
                continue

            if current_value > best:
                # since we will save now, overwrite all values
                self.best_metric_value.update(accuracies)
                self._should_save = True
            else:
                self._should_save = False

            break

        self.validation_accuracies = accuracies

    def run(self):
        tic = time.time()
        self.neural_scorer.reset_metrics()

        instances = read_instances(self.options.test_path)
        logger.info('Number of instances: %d' % len(instances))
        data = self.preprocess_instances(instances)
        data.prepare_batches(self.options.batch_size, sort=False)
        predictions = []

        for batch in data.batches:
            batch_predictions = self.run_batch(batch)
            predictions.extend(batch_predictions)

        self.write_predictions(instances, predictions)
        toc = time.time()
        logger.debug('Scoring time: %f' % self.neural_scorer.time_scoring)
        logger.debug('Decoding time: %f' % self.neural_scorer.time_decoding)
        logger.info('Total running time: %f' % (toc - tic))

    def write_predictions(self, instances, predictions):
        """
        Write predictions to a file.

        :param instances: the instances in the original format (i.e., not the
            "formatted" one, but retaining the original contents)
        :param predictions: list with predictions per instance
        """
        self.writer.open(self.options.output_path)
        for instance, inst_prediction in zip(instances, predictions):
            self.label_instance(instance, inst_prediction)
            self.writer.write(instance)

        self.writer.close()

    def read_train_instances(self):
        '''Create batch of training and validation instances.'''
        import time
        tic = time.time()
        logger.info('Creating instances...')

        train_instances = read_instances(self.options.training_path)
        valid_instances = read_instances(self.options.valid_path)
        logger.info('Number of train instances: %d' % len(train_instances))
        logger.info('Number of validation instances: %d'
                     % len(valid_instances))
        toc = time.time()
        logger.info('Time: %f' % (toc - tic))
        return train_instances, valid_instances

    def preprocess_instances(self, instances, report=True):
        """
        Create parts for all instances in the batch.

        :param instances: list of non-formatted Instance objects
        :param report: log the number of created parts and pruner errors. It
            should be False in a pruner model.
        :return: an InstanceData object.
            It contains formatted instances.
            In neural models, features is a list of None.
        """
        start = time.time()
        all_parts = []
        all_gold_labels = []
        formatted_instances = []
        self.pruner_mistakes = 0
        num_relations = self.token_dictionary.get_num_deprels()
        labeled = not self.options.unlabeled
        if self.options.bert_model is None:
            bert_tokenizer = None
        else:
            bert_tokenizer = BertTokenizer.from_pretrained(
                self.options.bert_model)

        if self.options.parse and self.has_pruner:
            prune_masks = self.run_pruner(instances)
        else:
            prune_masks = None

        for i, instance in enumerate(instances):
            mask = None if prune_masks is None else prune_masks[i]
            numeric_instance = DependencyInstanceNumeric(
                instance, self.token_dictionary, self.options.case_sensitive,
                bert_tokenizer)
            parts = DependencyParts(numeric_instance, self.options.model_type,
                                    mask, labeled, num_relations)
            gold_labels = self.get_gold_labels(numeric_instance)

            formatted_instances.append(numeric_instance)
            all_parts.append(parts)
            all_gold_labels.append(gold_labels)

        data = InstanceData(formatted_instances, all_parts, all_gold_labels)
        if report:
            self._report_make_parts(data)
        preprocess_time = time.time() - start
        logger.debug('Time to preprocess: %f' % preprocess_time)
        return data

    def reset_performance_metrics(self):
        """
        Reset some variables used to keep track of training performance.
        """
        self.num_train_instances = 0
        self.neural_scorer.reset_metrics()
        self.accumulated_hits = {}
        for target in self.additional_targets:
            self.accumulated_hits[target] = 0

        self.accumulated_uas = 0.
        self.accumulated_las = 0.
        self.total_tokens = 0

    def train(self):
        """Train the parser and/or tagger and/or lemmatizer"""
        train_instances, valid_instances = self.read_train_instances()
        logger.info('Preprocessing training data')
        train_data = self.preprocess_instances(train_instances)
        logger.info('\nPreprocessing validation data')
        valid_data = self.preprocess_instances(valid_instances)
        train_data.prepare_batches(self.options.batch_size, sort=True)
        valid_data.prepare_batches(self.options.batch_size, sort=True)
        logger.info('Training data spread across %d batches'
                     % len(train_data.batches))
        logger.info('Validation data spread across %d batches\n'
                     % len(valid_data.batches))

        self._reset_best_validation_metric()
        self.reset_performance_metrics()
        frozen_encoder = True
        num_bad_evals = 0

        for global_step in range(1, self.options.max_steps + 1):
            batch = train_data.get_next_batch()
            self.neural_scorer.train_batch(batch)
            self.num_train_instances += len(batch)

            if global_step % self.options.log_interval == 0:
                msg = 'Step %d' % global_step
                logger.info(msg)
                self.neural_scorer.train_report(self.num_train_instances)
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
                    break

                # unfreeze encoder weights after first dev set run
                if frozen_encoder:
                    encoder_lr = self.options.bert_learning_rate
                    self.neural_scorer.unfreeze_encoder(encoder_lr,
                                                        self.options.max_steps)
                    frozen_encoder = False

        msg = 'Saved model with the following validation accuracies:\n'
        for target in self.best_metric_value:
            name = target2string[target]
            value = self.best_metric_value[target]
            msg += '%s: %f\n' % (name, value)

        logger.info(msg)

    def run_on_validation(self, valid_data):
        """
        Run the model on validation data
        """
        valid_start = time.time()
        self.neural_scorer.eval_mode()

        predictions = []
        for batch in valid_data.batches:
            batch_predictions = self.run_batch(batch)
            predictions.extend(batch_predictions)

        self.compute_validation_metrics(valid_data, predictions)

        valid_end = time.time()
        time_validation = valid_end - valid_start

        logger.info('Time to run on validation: %.2f' % time_validation)

        msgs = ['Validation accuracies:\t']
        for target in self.validation_accuracies:
            target_name = target2string[target]
            acc = self.validation_accuracies[target]
            msgs.append('%s: %.4f' % (target_name, acc))
        logger.info('\t'.join(msgs))

        if self._should_save:
            logger.info('Saved model')

        logger.info('\n')

    def run_batch(self, instance_data: InstanceData):
        """
        Predict the output for the given instances.

        :return: a list of arrays with the predicted outputs if return_loss is
            False. If it's True, a tuple with predictions and losses.
            Each prediction is a dictionary mapping a target name to the
            prediction vector.
        """
        self.neural_scorer.eval_mode()
        predictions = self.neural_scorer.predict(
            instance_data, single_root=self.options.single_root)

        if self.options.lemma:
            # if this model includes a lemmatizer, cut sequences at EOS
            eos = self.token_dictionary.get_character_id(EOS)
            empty = self.token_dictionary.get_character_id(EMPTY)

            for i in range(len(instance_data)):
                lemma_prediction = predictions[i][Target.LEMMA]
                lemmas = cut_sequences_at_eos(lemma_prediction, eos, empty)
                predictions[i][Target.LEMMA] = lemmas

        return predictions

    def label_instance(self, instance, output):
        """
        :type instance: DependencyInstance
        :param output: dictionary mapping target names to predictions
        :return:
        """
        for m in range(1, len(instance)):
            if self.options.parse:
                instance.heads[m] = output[Target.HEADS][m - 1]
                relation = output[Target.RELATIONS][m - 1]
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
            if self.options.lemma:
                predictions = output[Target.LEMMA][m - 1]
                lemma = ''.join(self.token_dictionary.
                                character_alphabet.get_label_name(c)
                                for c in predictions)
                instance.lemmas[m] = lemma


def load_pruner(model_path):
    """
    Load and return a pruner model.

    This function takes care of keeping the main parser and the pruner
    configurations separate.
    """
    logger.info('Loading pruner from %s' % model_path)
    pruner = TurboParser.load(path=model_path)
    pruner.set_as_pruner()

    return pruner


def cut_sequences_at_eos(predictions, eos_index, empty_index):
    """
    Convert a tensor of lemmas to lists ending when EOS character is used.

    :param predictions: an array (num_tokens, max_num_chars)
    :return: a list with num_tokens arrays. Each array contains the char ids.
    """
    lemmas = []

    # store positions for faster access
    eos_mask = predictions == eos_index
    # mask is a boolean vector; argmax returns the first occurence of True
    eos_positions = eos_mask.argmax(-1)

    for i, token_prediction in enumerate(predictions):
        eos_position = eos_positions[i]
        if eos_position == 0:
            # either no EOS (and we use the whole row) or EOS at the very
            # first position
            if token_prediction[0] == eos_index:
                lemma = [empty_index]
            else:
                lemma = token_prediction
        else:
            lemma = token_prediction[:eos_position]
        lemmas.append(lemma)

    return lemmas
