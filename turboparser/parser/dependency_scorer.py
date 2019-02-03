from ..classifier.neural_scorer import NeuralScorer

import torch
from torch.nn import functional as F


class DependencyNeuralScorer(NeuralScorer):
    """
    Subclass of neural scorer that can compute loss on both dependency parsing
    and POS tagging.
    """
    def compute_tag_gradients(self, gold_labels):
        """
        Compute the gradients for the tagging tasks.

        :param gold_labels: list of dictionaries mapping each target name to a
            list with the gold targets for the sentences in the batch.
            Each array is as long as its sentence.
        """
        if len(gold_labels[0]) == 0:
            # this model does not output tags
            return

        def _compute_loss(target_gold, logits):
            targets = self.pad_labels(target_gold)

            # cross_entropy expects (batch, n_classes, ...)
            logits = logits.transpose(1, 2)
            cross_entropy = F.cross_entropy(logits, targets, ignore_index=-1)

            return cross_entropy

        loss = torch.tensor(0.)
        if torch.cuda.is_available():
            loss = loss.cuda()

        for target in ['upos', 'xpos', 'morph']:
            if target not in gold_labels[0]:
                continue

            target_gold = [item[target] for item in gold_labels]
            logits = self.model.scores[target]
            loss += _compute_loss(target_gold, logits)

        loss.backward(retain_graph=True)

    def pad_labels(self, labels):
        """Pad labels with -1 so that all of them have the same length"""
        batch_size = len(labels)
        max_length = max(len(a) for a in labels)
        shape = [batch_size, max_length]
        padded = torch.full(shape, -1, dtype=torch.long)
        for i in range(batch_size):
            padded[i, :len(labels[i])] = torch.tensor(labels[i])

        return padded

    def compute_tag_loss(self, scores, gold_output):
        """Compute the loss for any tagging subtask"""
        gold_output = self.pad_labels(gold_output)
        return F.cross_entropy(scores, gold_output)

    def compute_scores(self, instances, parts):
        """
        Compute the scores for all the targets this scorer

        :return: a dictionary mapping each target name to a numpy array with
            the scores.
        """
        if not isinstance(instances, list):
            instances = [instances]
            parts = [parts]

        self.scores = self.model(instances, parts)
        scores = {}
        for target in self.scores:
            scores[target] = self.scores[target].detach().numpy()

        return scores
