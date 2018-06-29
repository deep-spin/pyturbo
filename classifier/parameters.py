'''A generic implementation of a parameter vector.'''
from classifier.sparse_parameter_vectors import SparseParameterVector, \
    SparseLabeledParameterVector
import logging

class FeatureVector(object):
    '''This class implements a feature vector, which is convenient to sum over
    binary features, weight them, etc. It just uses the classes
    SparseParameterVector and SparseLabeledParameterVector, which allow fast
    insertions and lookups.'''
    def __init__(self):
        self.weights = SparseParameterVector()
        self.labeled_weights = SparseLabeledParameterVector()

    def get_squared_norm(self):
        '''Get the squared norm of the parameter vector.'''
        return self.weights.get_squared_norm() + \
            self.labeled_weights.get_squared_norm()


class Parameters(object):
    '''A class for holding and updating parameters in a linear model.'''
    def __init__(self, use_average=True):
        self.use_average = use_average
        self.weights = SparseParameterVector()
        self.labeled_weights = SparseLabeledParameterVector()
        if self.use_average:
            self.averaged_weights = SparseParameterVector()
            self.averaged_labeled_weights = SparseLabeledParameterVector()

    def save(self, file):
        '''Save the parameters.'''
        raise NotImplementedError

    def load(self, file):
        '''Load the parameters.'''
        raise NotImplementedError

    def stop_growth(self):
        '''Lock the parameter vector. A locked vector means that no
        features can be added.'''
        self.weights.stop_growth();
        self.averaged_weights.stop_growth();
        self.labeled_weights.stop_growth();
        self.averaged_labeled_weights.stop_growth();

    def allow_growth(self):
        '''Unlock the parameter vector. A locked vector means that no
        features can be added.'''
        self.weights.allow_growth();
        self.averaged_weights.allow_growth();
        self.labeled_weights.allow_growth();
        self.averaged_labeled_weights.allow_growth();

    def __len__(self):
        '''Get the number of parameters.
        NOTE: this counts the parameters of the features that are conjoined with
        output labels as a single parameter.'''
        return len(self.weights) + len(self.labeled_weights)

    def exists(self, key):
        '''Checks if a feature exists.'''
        return self.weights.exists(key)

    def exists_labeled(self, key):
        '''Checks if a labeled feature exists.'''
        return self.labeled_weights.exists(key)

    def get(self, key):
        '''Get the weight of a "simple" feature.'''
        return self.weights.get(key)

    def get_labeled(self, key, labels):
        '''Get the weights of features conjoined with output labels.
        The vector "labels" contains the labels that we want to conjoin with;
        label_scores will contain (as output) the weight for each label.
        If the feature does not exist, the label_scores will be an empty
        vector.'''
        return self.labeled_weights.get(key, labels)

    def get_squared_norm(self):
        '''Get the squared norm of the parameter vector.'''
        return self.weights.get_squared_norm() + \
            self.labeled_weights.get_squared_norm()

    def compute_score(self, features):
        '''Compute the score corresponding to a set of "simple" features.
        "features" is a list of keys (e.g. 64-bit integers).'''
        return sum([self.get(key) for key in features])

    def compute_labeled_scores(self, features, labels):
        '''Compute the scores corresponding to a set of features, conjoined with
        output labels. The vector scores, provided as output, contains the score
        for each label.'''
        scores = [0.] * len(labels)
        for key in features:
            label_scores = self.get_labeled(key, labels)
            if len(label_scores):
                for k in range(labels):
                    scores[k] += label_scores[k]
        return scores

    def scale(self, scale_factor):
        '''Scale the parameter vector by a scale factor.'''
        self.weights.scale(scale_factor)
        self.labeled_weights.scale(scale_factor)

    def make_gradient_step(self, features, eta, iteration, gradient_value):
        '''Make a gradient step with a stepsize of eta, with respect to a vector
        of "simple" features.
        The iteration number is provided as input since it is necessary to
        update the wanna-be "averaged parameters" in an efficient manner.'''
        for key in features:
            self.weights.add(key, -eta*gradient_value)
            if self.use_average:
                # perceptron/mira:
                # T*u1 + (T-1)*u2 + ... u_T
                # = T*(u1 + u2 + ...) - u2 - 2*u3 - (T-1)*u_T
                # = T*w_T - u2 - 2*u3 - (T-1)*u_T.
                self.averaged_weights.add(
                    key, float(iteration) * eta * gradient_value)

    def make_label_gradient_step(self, features, eta, iteration, label,
                                 gradient_value):
        '''Make a gradient step with a stepsize of eta, with respect to a vector
        of features conjoined with a label.
        The iteration number is provided as input since it is necessary to
        update the wanna-be "averaged parameters" in an efficient manner.'''
        for key in features:
            self.labeled_weights.add(key, label, -eta*gradient_value)
            if self.use_average:
                self.averaged_labeled_weights.add(
                    key, label, float(iteration) * eta * gradient_value)

    def finalize(self, num_iterations):
        '''Finalize training, after a total of num_iterations. This is a no-op
        unless we are averaging the parameter vector, in which case the averaged
        parameters are finally computed and replace the original parameters.'''
        if self.use_average:
            logging.info('Averaging the weights...')
            self.averaged_weights.scale(1./float(num_iterations))
            self.weights.add_vector(self.averaged_weights)
            self.averaged_labeled_weights.scale(1./float(num_iterations))
            self.labeled_weights.add_vector(self.averaged_labeled_weights)

