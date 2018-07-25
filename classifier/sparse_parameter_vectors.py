'''Implementations of sparse parameter vectors to use in linear models.'''

import logging


class SparseParameterVector(object):
    '''A class for defining a sparse parameter vector in a linear model.'''
    def __init__(self):
        # Weight vectors, up to a scale.
        self.values = dict()
        # The scale factor, such that w = values * scale.
        self.scale_factor = 1.
        # The squared norm of the parameter vector.
        self.squared_norm = 0.
        # True if parameters are locked.
        self.locked = False
        # Threshold for renormalizing the parameter vector.
        self.scale_factor_threshold = 1e-9

    def stop_growth(self):
        '''Lock the parameter vector. If the vector is locked, no new
        features can be inserted.'''
        self.locked = True

    def allow_growth(self):
        '''Unlock the parameter vector. If the vector is locked, no new
        features can be inserted.'''
        self.locked = False

    def save(self, file):
        '''Save the parameters.'''
        raise NotImplementedError

    def load(self, file):
        '''Load the parameters.'''
        raise NotImplementedError

    def __len__(self):
        '''Get the number of instantiated features.'''
        return len(self.values)

    def exists(self, key):
        '''True if this feature key is already instantiated.'''
        return key in self.values

    def get(self, key):
        '''Get the weight of this feature key.'''
        if key in self.values:
            return self._get_value(key)
        else:
            return 0.

    def get_squared_norm(self):
        '''Get the squared norm of the parameter vector.'''
        return self.squared_norm

    def scale(self, scale_factor):
        '''Scale the parameter vector by a factor.
        w_k' = w_k * c_k'''
        self.scale_factor *= scale_factor
        self.squared_norm *= scale_factor*scale_factor
        self._renormalize_if_necessary()

    def set(self, key, value):
        '''Set the weight of this feature key to "value". Return false if the
        feature is not instantiated and cannot be inserted.
        w'[id] = val.'''
        if self._find_or_insert(key):
            self._set_value(key, value)
            return True
        else:
            return False

    def add(self, key, value):
        '''Increment the weight of this feature key by an amount of "value".
        Return false if the feature is not instantiated and cannot be inserted.
        w'[id] = w[id] + val.'''
        if self._find_or_insert(key):
            self._set_value(key, self._get_value(key) + value)
            return True
        else:
            return False

    def add_multiple(self, keys, values):
        '''Increments the weights of several features.
        NOTE: Silently bypasses the ones that could not be inserted, if any.
        w'[id] = w[id] + val.'''
        for key, value in zip(keys, values):
            self.add(self, key, value)

    def add_vector(self, parameters):
        '''Adds two parameter vectors. This has the effect of incrementing the
        weights of several features.
        NOTE: Silently bypasses the ones that could not be inserted, if any.
        w'[id] = w[id] + val.'''
        for key in parameters.values:
            self.add(key, parameters._get_value(key))

    def _set_value(self, key, value):
        '''Set the parameter value of a feature pointed by an iterator.'''
        current_value = self._get_value(key)
        self.squared_norm += value*value - current_value*current_value
        # Might lose precision here.
        self.values[key] = value / self.scale_factor
        # This prevents numerical issues:
        if self.squared_norm < 0.:
            self.squared_norm = 0.

    def _get_value(self, key):
        '''Get the parameter value of a feature pointed by a iterator.'''
        return self.values[key] * self.scale_factor

    def _find_or_insert(self, key):
        '''Obtain the iterator pointing to a feature key. If the key does not
        exist and the parameters are not locked, inserts the key and returns the
        corresponding iterator.'''
        if key in self.values:
            return True
        elif self.locked:
            return False
        else:
            self.values[key] = 0.
            return True

    def _renormalize_if_necessary(self):
        '''If the scale factor is too small, renormalize the entire parameter
        map.'''
        if self.scale_factor > -self.scale_factor_threshold and \
           self.scale_factor < self.scale_factor_threshold:
            self._renormalize()

    def _renormalize(self):
        '''Renormalize the entire parameter map (an expensive operation).'''
        logging.info('Renormalizing the parameter map...')
        for key in self.values:
            self.values[key] *= self.scale_factor
        self.scale_factor = 1.


class SparseLabelWeights(object):
    '''Sparse implementation of LabelWeights.'''
    def __init__(self):
        self.label_weights = []

    def is_sparse(self):
        return True

    def __len__(self):
        return len(self.label_weights)

    def get_weight(self, label):
        for k in range(len(self.label_weights)):
            if label == self.label_weights[k][0]:
                return self.label_weights[k][1]
        return 0.

    def set_weight(self, label, weight):
        for k in range(len(self.label_weights)):
            if label == self.label_weights[k][0]:
                self.label_weights[k][1] = weight
                return
        self.label_weights.append((label, weight))

    def add_weight(self, label, weight):
        for k in range(len(self.label_weights)):
            if label == self.label_weights[k][0]:
                self.label_weights[k][1] += weight
                return
        self.label_weights.append((label, weight))

    def get_label_weight_by_position(self, position):
        return self.label_weights[position]

    def set_weight_by_position(self, position, weight):
        self.label_weights[position][1] = weight


class DenseLabelWeights(object):
    '''Dense implementation of LabelWeights.'''
    def __init__(self):
        self.weights = []

    def is_sparse(self):
        return False

    def __len__(self):
        return len(self.weights)

    def get_weight(self, label):
        if label >= len(self.weights):
            return 0.
        else:
            return self.weights[label]

    def set_weight(self, label, weight):
        if label >= len(self.weights):
            self.weights.extend([0.] * (1 + label - len(self.weights)))
        self.weights[label] = weight

    def add_weight(self, label, weight):
        if label >= len(self.weights):
            self.weights.extend([0.] * (1 + label - len(self.weights)))
        self.weights[label] += weight

    def get_label_weight_by_position(self, position):
        return position, self.weight[position]

    def set_weight_by_position(self, position, weight):
        self.weight[position] = weight


class SparseLabeledParameterVector(object):
    '''This class implements a sparse parameter vector, which contains weights
    for the labels conjoined with each feature key. For fast lookup, this is
    implemented using an hash table.
    We represent a weight vector as a triple
    (values_, scale_factor_ , squared_norm_), where values_ contains the
    weights up to a scale, scale_factor_ is a factor such that
    weights[k] = scale_factor_ * values_[k], and the squared norm is cached.
    This way we can scale the weight vector in constant time (this operation is
    necessary in some training algorithms such as SGD), and manipulating a few
    elements is still fast. Plus, we can obtain the norm in constant time.'''
    def __init__(self):
        # Weight vectors, up to a scale.
        self.values = dict()
        # The scale factor, such that w = values * scale.
        self.scale_factor = 1.
        # The squared norm of the parameter vector.
        self.squared_norm = 0.
        # True if parameters are locked.
        self.locked = False
        # Threshold for renormalizing the parameter vector.
        self.scale_factor_threshold = 1e-9
        # After more than these number of labels, use a dense representation.
        self.max_sparse_labels = 5;

    def stop_growth(self):
        '''Lock the parameter vector. If the vector is locked, no new
        features can be inserted.'''
        self.locked = True

    def allow_growth(self):
        '''Unlock the parameter vector. If the vector is locked, no new
        features can be inserted.'''
        self.locked = False

    def save(self, file):
        '''Save the parameters.'''
        raise NotImplementedError

    def load(self, file):
        '''Load the parameters.'''
        raise NotImplementedError

    def __len__(self):
        '''Get the number of instantiated features.'''
        return len(self.values)

    def exists(self, key):
        '''True if this feature key is already instantiated.'''
        return key in self.values

    def get(self, key):
        '''Get the weight of this feature key.'''
        if key in self.values:
            return self._get_values(key)
        else:
            return 0.

    def get_squared_norm(self):
        '''Get the squared norm of the parameter vector.'''
        return self.squared_norm

    def scale(self, scale_factor):
        '''Scale the parameter vector by a factor.
        w_k' = w_k * c_k'''
        self.scale_factor *= scale_factor
        self.squared_norm *= scale_factor*scale_factor
        self._renormalize_if_necessary()

    def set(self, key, label, value):
        '''Set the weight of this feature key to "value". Return false if the
        feature is not instantiated and cannot be inserted.
        w'[id] = val.'''
        if self._find_or_insert(key):
            self._set_value(key, label, value)
            return True
        else:
            return False

    def add(self, key, label, value):
        '''Increment the weight of this feature key by an amount of "value".
        Return false if the feature is not instantiated and cannot be inserted.
        w'[id] = w[id] + val.'''
        if self._find_or_insert(key):
            self._add_value(key, label, value)
            return True
        else:
            return False

    def add_multiple(self, keys, labels, values):
        '''Increments the weights of several features.
        NOTE: Silently bypasses the ones that could not be inserted, if any.
        w'[id] = w[id] + val.'''
        for key, label, value in zip(keys, labels, values):
            self.add(self, key, label, value)

    def add_vector(self, parameters):
        '''Adds two parameter vectors. This has the effect of incrementing the
        weights of several features.
        NOTE: Silently bypasses the ones that could not be inserted, if any.
        w'[id] = w[id] + val.'''
        for key, label_weights in parameters.values.items():
            for k in range(len(label_weights)):
                label, value = label_weights.get_label_weight_by_position(k)
                value *= parameters.scale_factor
                self.add(key, label, value)

    def _get_values(self, key, labels):
        '''Get the weights for the specified labels.'''
        values = [0.] * len(labels)
        label_weights = self.values[key]
        for i in range(len(labels)):
            values[i] = label_weights.get_weight(labels[i]) * scale_factor_
        return values

    def _get_value(self, key, label):
        '''Get the weight for the specified label.'''
        label_weights = self.values[key]
        return label_weights.get_weight(label) * self.scale_factor

    def _set_value(self, key, label, value):
        '''Set the weight for the specified label.'''
        current_value = self._get_value(key, label)
        self.squared_norm += value*value - current_value*current_value
        label_weights = self.values[key]
        label_weights.set_weight(label, value / self.scale_factor)

        # If the number of labels is growing large, make this into dense
        # label weights.
        if len(label_weights) > self.max_sparse_labels and \
           label_weights.is_sparse():
            dense_label_weights = DenseLabelWeights(label_weights)
            self.values[key] = dense_label_weights

        # This prevents numerical issues:
        if self.squared_norm < 0.:
            self.squared_norm = 0.

    def _add_value(self, key, label, value):
        '''Set the weight for the specified label.'''
        current_value = self._get_value(key, label)
        value += current_value
        self.squared_norm += value*value - current_value*current_value
        label_weights = self.values[key]
        if label_weights is None:
            label_weights = SparseLabelWeights()
        label_weights.set_weight(label, value / self.scale_factor)

        # If the number of labels is growing large, make this into dense
        # label weights.
        if len(label_weights) > self.max_sparse_labels and \
           label_weights.is_sparse():
            dense_label_weights = DenseLabelWeights(label_weights)
            self.values[key] = dense_label_weights

        # This prevents numerical issues:
        if self.squared_norm < 0.:
            self.squared_norm = 0.

    def _find_or_insert(self, key):
        '''Obtain the iterator pointing to a feature key. If the key does not
        exist and the parameters are not locked, inserts the key and returns the
        corresponding iterator.'''
        if key in self.values:
            return True
        elif self.growth_stopped():
            return False
        else:
            self.values[key] = SparseLabelWeights()
            return True

    def _renormalize_if_necessary(self):
        '''If the scale factor is too small, renormalize the entire parameter
        map.'''
        if self.scale_factor > -self.scale_factor_threshold and \
           self.scale_factor < self.scale_factor_threshold:
            self._renormalize()

    def _renormalize(self):
        '''Renormalize the entire parameter map (an expensive operation).'''
        logging.info('Renormalizing the parameter map...')
        for key in self.values:
            label_weights = self.values[key]
            for k in range(len(label_weights)):
                label, value = label_weights.get_label_weight_by_position(k)
                label_weights.set_label_weight_by_position(
                    k, value * self.scale_factor)
        self.scale_factor = 1.
