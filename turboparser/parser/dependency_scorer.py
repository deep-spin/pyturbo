from ..classifier.neural_scorer import NeuralScorer

import torch
from torch.nn import functional as F


class DependencyNeuralScorer(NeuralScorer):
    """
    Subclass of neural scorer that can compute loss on both dependency parsing
    and POS tagging.
    """
    def compute_pos_gradients(self, gold_output):
        """
        :param gold_output: list with the gold outputs for each sentence
        """
        # pos_logits is (batch, num_tokens, num_tags)
        # ignore logits after each sentence end
        if not isinstance(gold_output, list):
            gold_output = [gold_output]

        batch_size = len(gold_output)
        max_length = max(len(g) for g in gold_output)
        shape = (batch_size, max_length)
        targets = torch.full(shape, -1, dtype=torch.long)
        for i in range(batch_size):
            targets[i, :len(gold_output[i])] = torch.tensor(gold_output[i])

        # cross_entropy expects (batch, n_classes, ...)
        logits = self.model.pos_logits.transpose(1, 2)
        cross_entropy = F.cross_entropy(logits, targets, ignore_index=-1)
        cross_entropy.backward(retain_graph=True)

    def compute_pos_loss(self, scores, gold_output):
        return F.cross_entropy(scores, gold_output)

    def compute_scores(self, instances, parts):
        dependency_scores = super(DependencyNeuralScorer, self).\
            compute_scores(instances, parts)

        if self.model.predict_tags:
            pos_scores = self.model.pos_logits.detach().numpy()
            return list(zip(dependency_scores, pos_scores))

        return dependency_scores
