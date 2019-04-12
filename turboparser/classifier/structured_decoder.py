import numpy as np


class StructuredDecoder(object):
    '''An abstract decoder for structured prediction.'''
    def __init__(self):
        pass

    def decode(self, instance, parts, scores):
        '''Decode, computing the highest-scores output.
        Must return a vector of 0/1 predicted_outputs of the same size
        as parts.'''
        raise NotImplementedError

    def _get_margin(self, parts):
        """
        Compute and return a margin vector to be used in the loss and a
        normalization term to be added to it.

        This function is supposed to be overridden.
        """
        return np.zeros_like(len(parts), dtype=np.float), 0

    def _add_cost_vector(self, parts, scores):
        """
        Add the cost margin to the scores.

        This is used before actually decoding.
        """
        return

    def decode_mira(self, instance, parts, scores, old_mira=False):
        '''Perform cost-augmented decoding or classical MIRA.'''
        if not old_mira:
            self._add_cost_vector(parts, scores)

        predicted_outputs = self.decode(instance, parts, scores)

        return predicted_outputs

    def decode_cost_augmented(self, instance, parts, scores):
        '''Perform cost-augmented decoding.'''
        return self.decode_mira(instance, parts, scores, old_mira=False)

    def compute_loss(self, gold_output, predicted_output, scores):
        '''
        Compute the cost-augmented loss for the given prediction

        :return:
        '''
        # there might be spurious scores for padding
        scores = scores[:len(gold_output)]
        p = 0.5 - gold_output
        q = 0.5 * np.sum(gold_output)
        cost = p.dot(predicted_output) + q
        loss = cost + scores.dot(predicted_output - gold_output)

        return loss
