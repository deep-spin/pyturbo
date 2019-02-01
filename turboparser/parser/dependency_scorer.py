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
        loss = torch.tensor(0.)
        if torch.cuda.is_available():
            loss = loss.cuda()
        for i in range(batch_size):
            # targets[i, :len(gold_output[i])] = torch.tensor(gold_output[i])
            logits = self.model.pos_logits[i]
            gold = torch.tensor(gold_output[i], dtype=torch.long)
            if torch.cuda.is_available():
                gold = gold.cuda()
            loss += F.cross_entropy(logits, gold)

        loss.backward(retain_graph=True)
        # # cross_entropy expects (batch, n_classes, ...)
        # logits = self.model.pos_logits.transpose(1, 2)
        # cross_entropy = F.cross_entropy(logits, targets, ignore_index=-1)
        # cross_entropy.backward(retain_graph=True)

    def compute_pos_loss(self, scores, gold_output):
        """Compute the loss for any tagging subtask"""
        return F.cross_entropy(scores, gold_output)

    def compute_scores(self, instances, parts):
        """
        Compute the scores for all the targets this scorer
        :param instances:
        :param parts:
        :return:
        """
        scores = {'dependency': super(DependencyNeuralScorer, self).
            compute_scores(instances, parts)}

        if self.model.predict_tags:
            upos_scores = [sent_scores.detach().numpy()
                           for sent_scores in +
                           self.model.upos_logits]
            scores['upos'] = upos_scores
            xpos_scores = [sent_scores.detach().numpy()
                           for sent_scores in self.model.xpos_logits]
            scores['xpos'] = xpos_scores

        return scores
