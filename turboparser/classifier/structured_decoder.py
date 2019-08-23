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

    def _add_margin_vector(self, parts, scores):
        """
        Add the margin to the scores.

        This is used before actually decoding.
        """
        return

    def decode_mira(self, instance, parts, scores, old_mira=False):
        '''Perform cost-augmented decoding or classical MIRA.'''
        if not old_mira:
            self._add_margin_vector(parts, scores)

        predicted_outputs = self.decode(instance, parts, scores)

        return predicted_outputs

    def decode_cost_augmented(self, instance, parts, scores):
        '''Perform cost-augmented decoding.'''
        return self.decode_mira(instance, parts, scores, old_mira=False)
