from ..classifier.neural_scorer import NeuralScorer

import torch
from torch.nn import functional as F


class DependencyNeuralScorer(NeuralScorer):
    """
    Subclass of neural scorer that can compute loss on both dependency parsing
    and POS tagging.
    """
    def compute_gradients(self, gold_parts, predicted_parts, gold_labels):
        """
        Compute the error gradients for parsing and tagging.

        :param gold_parts: either a numpy 1d array for a single item or a list
            of 1d arrays for a batch.
        :param predicted_parts: same as gold_output
        :param gold_labels: list of dictionaries mapping each target name to a
            list with the gold targets for the sentences in the batch.
            Each array is as long as its sentence.
        """
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

        if loss > 0:
            # TODO: a better way to skip backprop when there's no additional target
            loss.backward(retain_graph=True)
        super(DependencyNeuralScorer, self).compute_gradients(
            gold_parts, predicted_parts, None)

    def pad_labels(self, labels):
        """
        Pad labels with -1 so that all of them have the same length

        :param labels: a list (batch) of lists of labels
        """
        batch_size = len(labels)
        max_length = max(len(a) for a in labels)
        shape = [batch_size, max_length]
        padded = torch.full(shape, -1, dtype=torch.long)
        for i in range(batch_size):
            padded[i, :len(labels[i])] = torch.tensor(labels[i])

        if torch.cuda.is_available():
            padded = padded.cuda()

        return padded

    def compute_tag_loss(self, scores, gold_output,
                         reduction='elementwise_mean'):
        """Compute the loss for any tagging subtask

        scores and gold_output may be either a batch or a single instance.
        """
        if isinstance(gold_output[0], list):
            gold_output = self.pad_labels(gold_output)
            loss = F.cross_entropy(scores, gold_output, ignore_index=-1,
                                   reduction=reduction)
        else:
            gold_output = torch.tensor(gold_output, dtype=torch.long)
            scores = torch.tensor(scores[:len(gold_output)])
            loss = F.cross_entropy(scores, gold_output, reduction=reduction)

        return loss

    def compute_scores(self, instances, parts):
        """
        Compute the scores for all the targets this scorer

        :return: a list of dictionaries mapping each target name to its scores
        """
        if not isinstance(instances, list):
            instances = [instances]
            parts = [parts]

        model_scores = self.model(instances, parts)
        self.part_scores = model_scores['dependency']
        numpy_scores = {target: model_scores[target].detach().cpu().numpy()
                        for target in model_scores}

        score_list = []
        for i in range(len(self.part_scores)):
            instance_scores = {target: numpy_scores[target][i]
                               for target in numpy_scores}
            score_list.append(instance_scores)

        return score_list
