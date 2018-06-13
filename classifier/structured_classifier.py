'''A generic implementation of an abstract structured classifier.'''

import numpy as np
#import sparse_vector as sv
#import linear_model as lm
import classifier.utils
from classifier.parameters import Parameters
#import sys
import logging
import time

logging.basicConfig(level=logging.DEBUG)

class StructuredClassifier(object):
    '''An abstract structured classifier.'''
    def __init__(self, options):
        self.options = options
        self.dictionary = None
        self.reader = None
        self.writer = None
        self.decoder = None
        self.parameters = None

        # To delete:
        #self.model = lm.LinearModel()
        #self.decoder = StructuredDecoder()

    def save(self, model_path):
        '''Save the full configuration and model.'''
        raise NotImplementedError

    def load(self, model_path):
        '''Load the full configuration and model.'''
        raise NotImplementedError

    def get_formatted_instance(self, instance):
        '''Obtain a "formatted" instance. Override this function for
        task-specific formatted instances, which may be different from instance
        since they may have extra information, data in numeric format for faster
        processing, etc.'''
        return instance

    def label_instance(self, instance, parts, predicted_output):
        '''Return a labeled instance by adding the predicted output
        information.
        Given a vector of parts of a desired output, builds the output
        information in the instance that corresponds to that output.
        Note: this function is task-specific and needs to be implemented by the
        deriving class.'''
        raise NotImplementedError

    def preprocess_data(self):
        '''Preprocess the data before training begins. Override this function
        for task-specific instance preprocessing.'''
        return

    def write_prediction(self, instance, parts, predicted_output):
        '''Write prediction to a file.'''
        raise NotImplementedError

    def make_parts(self, instance):
        '''Compute the task-specific parts for this instance.
        Construct the vector of parts for a particular instance.
        Eventually, obtain the binary vector of gold outputs (one entry per
        part) if this information is available.
        Note: this function is task-specific and needs to be implemented by the
        deriving class.'''
        raise NotImplementedError

    def make_features(self, instance, parts):
        '''Construct the vector of features for a particular instance and given
        the parts. The vector will be of the same size as the vector of parts.
        '''
        return self.make_selected_features(instance, parts,
                                           [True for part in parts])

    def make_selected_features(self, instance, parts, selected_parts):
        '''Construct the vector of features for a particular instance and given
        a selected set of parts (parts which are selected as marked as true).
        The vector will be of the same size as the vector of parts.
        Note: this function is task-specific and needs to be implemented by the
        deriving class.'''
        raise NotImplementedError

    def compute_scores(self, instance, parts, features):
        '''Compute a score for every part in the instance using the current
        model and the part-specific features.
        Given an instance, parts, and features, compute the scores. This will
        look at the current parameters. Each part will receive a score, so the
        vector of scores will be of the same size as the vector of parts.
        NOTE: Override this method for task-specific score computation (e.g.
        to handle labeled features, etc.).
        TODO: handle labeled features here instead of having to override.'''
        num_parts = len(parts)
        scores = np.zeros(num_parts)
        for r in range(num_parts):
            scores[r] = self.parameters.compute_score(features[r])
        return scores

    def make_gradient_step(self, parts, features, eta, t, gold_output,
                           predicted_output):
        '''Perform a gradient step updating the current model.
        Perform a gradient step with stepsize eta. The iteration number is
        provided as input since it may be necessary to keep track of the
        averaged weights. The gold output and the predicted output are also
        provided. The meaning of "predicted_output" depends on the training
        algorithm. In perceptron, it is the most likely output predicted by the
        model. In cost-augmented MIRA and structured SVMs, it is the
        cost-augmented prediction.
        In CRFs, it is the vector of posterior marginals for the parts.
        TODO: use "FeatureVector *difference" as input (see function
        MakeFeatureDifference(...) instead of computing on the fly).'''
        for r in range(len(parts)):
            if predicted_output[r] == gold_output[r]:
                continue
            part_features = features[r]
            self.parameters.make_gradient_step(
                part_features, eta, t, predicted_output[r]-gold_output[r])

    def make_feature_difference(self, parts, features, gold_output,
                                predicted_output):
        '''Compute the difference between predicted and gold feature vector.
        The meaning of "predicted_output" depends on the training algorithm.
        In perceptron, it is the most likely output predicted by the model.
        In cost-augmented MIRA and structured SVMs, it is the cost-augmented
        prediction.
        In CRFs, it is the vector of posterior marginals for the parts.'''
        difference = SparseVector()
        for r in range(len(parts)):
            if predicted_output[r] == gold_output[r]:
                continue
            part_features = features[r]
            difference.add(part_features,
                           predicted_output[r] - gold_output[r])
        return difference

    def remove_unsupported_features(self, instance, parts, features):
        '''Given an instance, a vector of parts, and features for those parts,
        remove all the features which are not supported, i.e., that were not
        previously created in the parameter vector. This is used for training
        with supported features (flag --only_supported_features).'''
        self.remove_unsupported_features(instance, parts,
                                         [True for part in parts], features)

    def remove_unsupported_features(self, instance, parts, selected_parts,
                                    features):
        '''Given an instance, a vector of selected parts, and features for those
        parts, remove all the features which are not supported. See description
        above.'''
        for r in range(len(parts)):
            if not selected_parts[r]:
                continue
            part_features = features[r]
            features[r] = [f for f in part_features
                           if self.parameters.exists(f)]

    def touch_parameters(self, parts, features, selected_parts):
        '''Perform an empty gradient step.
        Given a vector of parts, and features for those parts, "touch" all the
        parameters corresponding to those features. This will be a no-op for the
        parameters that exist already, and will create a parameter with a zero
        weight otherwise. This is used in a preprocessing stage for training
        with supported features (flag --only_supported_features).'''
        for r in range(len(parts)):
            if not selected_parts[r]:
                continue
            part_features = features[r]
            self.parameters.make_gradient_step(part_features, 0., 0, 0.)

    def transform_gold(self, instance, parts, scores, gold_output):
        '''This is a no-op by default. But it's convenient to have it here to
        build latent-variable structured classifiers (e.g. for coreference
        resolution).'''
        loss_inner = 0.
        return loss_inner

    def train(self):
        '''Train with a general online algorithm.'''
        self.preprocess_data()
        instances = self.create_instances()
        self.parameters = Parameters(use_average=self.options.use_averaging)
        if self.options.only_supported_features:
            self.make_supported_parameters()
        for epoch in range(self.options.training_epochs):
            self.train_epoch(epoch, instances)
        self.parameters.finalize(options.training_epochs * len(instances))

    def create_instances(self):
        '''Create batch of training instances.'''
        import time
        tic = time.time()
        logging.info('Creating instances...')
        self.reader.open(self.options.training_path)
        instances = []
        instance = self.reader.next()
        while instance:
            formatted_instance = self.get_formatted_instance(instance)
            instances.append(formatted_instance)
            instance = self.reader.next()
        self.reader.close()
        logging.info('Number of instances: %d' % len(instances))
        toc = time.time()
        logging.info('Time: %f' % (toc - tic))
        return instances

    def make_supported_parameters(self):
        '''Create parameters using only those supported in the gold outputs.
        Build and lock a parameter vector with only supported parameters, by
        looking at the gold outputs in the training data. This is a
        preprocessing stage for training with supported features (flag
        --only_supported_features).'''
        logging.info('Building supported feature set...')
        self.dictionary.stop_growth()
        self.parameters.allow_growth()
        for instance in instances:
            parts, gold_outputs = self.make_parts(instance)
            selected_parts = [True if gold_outputs[r] > 0.5 else False
                              for r in range(len(parts))]
            features = self.make_selected_features(instance, parts,
                                                   selected_parts)
            self.touch_parameters(parts, features, selected_parts)
        self.parameters.stop_growth()
        logging.info('Number of features: %d', len(self.parameters))

    def train_epoch(self, epoch, instances):
        '''Run one epoch of an online algorithm.'''
        import time
        start = time.time()
        time_decoding = 0
        time_scores = 0

        total_cost = 0.
        total_loss = 0.
        if self.options.training_algorithm in ['perceptron']:
            num_mistakes = 0
            num_total = 0
        elif self.options.training_algorithm in ['mira', 'svm_mira']:
            truncated = 0

        lambda_coeff = 1.0 / (self.options.regularization_constant *
                              float(len(instances)))
        if epoch == 0:
            logging.info('\t'.join(
                ['Lambda: %f' % lambda_coeff,
                 'Regularization constant: %f' %
                 self.options.regularization_constant,
                 'Number of instances: %d' % len(instances)]))
        logging.info(' Iteration #%d' % (epoch + 1))

        self.dictionary.stop_growth()

        t = len(instances) * epoch
        for instance in instances:
            # Compute parts, features, and scores.
            parts, gold_output = self.make_parts(instance)
            features = self.make_features(instance, parts)

            # If using only supported features, must remove the unsupported
            # ones. This is necessary not to mess up the computation of the
            # squared norm of the feature difference vector in MIRA.
            if self.options.only_supported_features:
                self.remove_unsupported_features(instance, parts, features)

            start_scores = time.time()
            scores = self.compute_scores(instance, parts, features)
            end_scores = time.time()
            time_scores += end_scores - start_scores

            # This is a no-op by default. But it's convenient to have it here to
            # build latent-variable structured classifiers (e.g. for coreference
            # resolution).
            inner_loss = self.transform_gold(instance, parts, scores,
                                             gold_outputs)

            # Do the decoding.
            start_decoding = time.time()
            algorithm = self.options.training_algorithm
            if algorithm in ['perceptron']:
                predicted_output = self.decoder.decode(instance, parts,
                                                       scores)
                for r in range(len(parts)):
                    num_total += 1
                    if not utils.nearly_eq_tol(gold_output[r],
                                               predicted_output[r], 1e-6):
                        num_mistakes += 1

            elif algorithm in ['mira']:
                predicted_output, cost, loss = self.decoder.decode_mira(
                    instance,
                    parts,
                    scores,
                    gold_output,
                    old_mira=True)

            elif algorithm in ['svm_mira', 'svm_sgd']:
                predicted_output, cost, loss = \
                    self.decoder.decode_cost_augmented(instance,
                                                       parts,
                                                       scores,
                                                       gold_output)
                total_cost += cost

            elif algorithm in ['crf_mira', 'crf_sgd']:
                predicted_output, entropy, loss = \
                    self.decoder.decode_marginals(instance,
                                                  parts,
                                                  scores,
                                                  gold_output)
                assert entropy >= 0
            else:
                raise NotImplementedError
            end_decoding = time.time()
            time_decoding += end_decoding - start_decoding

            # Update the total loss and cost.
            if algorithm in ['mira', 'svm_mira', 'svm_sgd', 'crf_mira',
                             'crf_sgd']:
                loss -= inner_loss
                if loss < 0.0:
                    if loss < -1e-12:
                        log.warning('Negative loss set to zero: %f' % loss)
                    loss = 0.0

                total_loss += loss

            # Compute the stepsize.
            if algorithm in ['perceptron']:
                eta = 1.0
            elif algorithm in ['mira', 'svm_mira', 'crf_mira']:
                difference = self.make_feature_difference(parts, features,
                                                          gold_output,
                                                          predicted_output)
                squared_norm = difference.squared_norm()
                threshold = 1e-9
                if loss < threshold or squared_norm < threshold:
                    eta = 0.0
                else:
                    eta = loss / squared_norm
                if eta > self.options.regularization_constant:
                    eta = self.options.regularization_constant
                    truncated += 1
            elif algorithm in ['svm_sgd']:
                if self.options.learning_rate_schedule == 'fixed':
                    eta = self.options.initial_learning_rate
                elif self.options.learning_rate_schedule == 'invsqrt':
                    eta = self.options.initial_learning_rate / \
                          np.sqrt(float(t + 1))
                elif self.options.learning_rate_schedule == 'inv':
                    eta = self.options.initial_learning_rate / (float(t + 1))
                else:
                    raise NotImplementedError

                # Scale the weight vector.
                decay = 1.0 - eta * lambda_coeff
                assert decay >= 0.
                self.parameters.scale(decay)

            # Make gradient step.
            self.make_gradient_step(parts, features, eta, t, gold_output,
                                    predicted_output)

            # Increment the round.
            t += 1

        end = time.time()
        logging.info('Time: %f' % end - start)
        logging.info('Time to score: %f' % time_scores)
        logging.info('Time to decode: %f' % time_decoding)
        logging.info('Number of features: %f' % len(parameters))

        if algorithm in ['perceptron']:
            logging.info('Number of mistakes: %d/%d (%f)' % num_mistakes,
                         num_total, float(num_mistakes) / float(num_total))
        else:
            sq_norm = self.parameters.squared_norm()
            regularization_value = 0.5 * lambda_coeff * \
                float(len(instances)) * sq_norm
            logging.info('\t'.join(['Total Loss: %f' % total_loss,
                                    'Total Reg: %f' % regularization_value,
                                    'Total Loss+Reg: %f' % total_loss +
                                    regularization_value,
                                    'Squared norm: %f' % sq_norm]))


    def run(self):
        '''Run the structured classifier on test data.'''
        import time
        tic = time.time()

        if self.options.evaluate:
            self.begin_evaluation()

        self.reader.open(options.test_file)
        self.writer.open(options.output_file)

        instances = []
        instance = self.reader.next()
        while instance:
            output_instance = self.classify_instance(instance)
            self.writer.write(output_instance)
            instances.append(instance)
            instance = self.reader.next()

        self.writer.close()
        self.reader.close()

        logging.info('Number of instances: %d' % len(instances))
        toc = time.time()
        logging.info('Time: %f' % (toc - tic))

        if self.options.evaluate:
            self.end_evaluation()

    def classify_instance(self, instance):
        '''Run the structured classifier on a single instance.'''
        formatted_instance = self.get_formatted_instance(instance)
        parts, gold_output = self.make_parts(formatted_instance)
        features = self.make_features(formatted_instance, parts)
        scores = self.compute_scores(instance, parts, features)
        predicted_output = self.decoder.decode(formatted_instance, parts,
                                               scores)
        output_instance = instance.copy()
        self.label_instance(parts, predicted_output, output_instance)
        if self.options.evaluate:
            self.evaluate_instance(instance, output_instance, parts,
                                   gold_outputs, predicted_outputs)
        return output_instance

    def begin_evaluation(self):
        '''Start all the evaluation counters for evaluating the classifier,
        evaluate each instance, and plot evaluation information at the end.
        This is done at test time when the flag --evaluate is activated.
        The version implemented here plots accuracy based on Hamming distance
        for the predicted and gold parts. Override this function for
        task-specific evaluation.'''
        self.num_mistakes = 0
        self.num_total_parts = 0

    def evaluate_instance(self, instance, output_instance, parts, gold_output,
                          predicted_output):
        for r in range(len(parts)):
            if not utils.nearly_eq_tol(gold_output[r],
                                       predicted_output[r], 1e-6):
                self.num_mistakes += 1
            self.num_total_parts += 1

    def end_evaluation(self):
        logging.info('Accuracy (parts): %f' %
                     float(self.num_total_parts - self.num_mistakes) /
                     float(self.num_total_parts))

