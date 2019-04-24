from .constants import Target
import torch
from torch.nn import functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import logging


class DependencyNeuralScorer(object):
    """
    Neural scorer for mediating the training of a Parser/Tagger neural model.
    """
    def __init__(self):
        self.part_scores = None
        self.model = None

    def compute_gradients(self, instance_data, predicted_parts):
        """
        Compute the error gradients for parsing and tagging.

        :param instance_data: InstanceData object
        :param predicted_parts: same as gold_output
        """
        def _compute_loss(target_gold, logits):
            targets = pad_labels(target_gold)

            # cross_entropy expects (batch, n_classes, ...)
            logits = logits.transpose(1, 2)
            cross_entropy = F.cross_entropy(logits, targets, ignore_index=-1)

            return cross_entropy

        loss = torch.tensor(0.)
        gold_labels = instance_data.gold_labels
        if torch.cuda.is_available():
            loss = loss.cuda()

        for target in [Target.UPOS, Target.XPOS, Target.MORPH]:
            if target not in gold_labels[0]:
                continue

            target_gold = [item[target] for item in gold_labels]
            logits = self.model.scores[target]
            loss += _compute_loss(target_gold, logits)

        batch_size = len(instance_data)
        for i in range(batch_size):
            inst_parts = instance_data.parts[i]
            gold_parts = inst_parts.gold_parts
            pred_item = predicted_parts[i]
            part_score_list = [self.model.scores[type_][i]
                               for type_ in inst_parts.type_order]
            part_scores = torch.cat(part_score_list)
            diff = torch.tensor(pred_item - gold_parts, dtype=part_scores.dtype,
                                device=part_scores.device)
            parts_loss_i = torch.dot(part_scores, diff) / gold_parts.sum()

            if parts_loss_i > 0:
                loss += parts_loss_i
            else:
                if parts_loss_i < -10e-6:
                    logging.warning('Ignoring negative loss', parts_loss_i)

        # Backpropagate to accumulate gradients.
        if loss > 0:
            loss.backward()

        return loss.item()

    def compute_tag_loss(self, scores, gold_output, reduction='mean'):
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
        numpy_scores = {}
        for target in model_scores:
            # dependency part scores are stored as lists of tensors; tags
            # are singleton tensors
            if target in [Target.UPOS, Target.XPOS, Target.MORPH]:
                numpy_scores[target] = model_scores[target].detach()\
                    .cpu().numpy()
            else:
                numpy_scores[target] = [tensor.detach().cpu().numpy()
                                        for tensor in model_scores[target]]

        # now convert a dictionary of arrays into a list of dictionaries
        score_list = []
        for i in range(len(instances)):
            instance_scores = {target: numpy_scores[target][i]
                               for target in numpy_scores}
            score_list.append(instance_scores)

        return score_list

    def initialize(self, model, learning_rate=0.001, decay=1,
                   beta1=0.9, beta2=0.999):
        self.set_model(model)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            params, lr=learning_rate, betas=(beta1, beta2))
        self.scheduler = scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=decay, patience=0, verbose=True)

    def set_model(self, model):
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()

    def train_mode(self):
        """
        Set the neural model to training mode
        """
        self.model.train()

    def eval_mode(self):
        """
        Set the neural model to eval mode
        """
        self.model.eval()

    def lr_scheduler_step(self, accuracy):
        """
        Perform a step of the learning rate scheduler, based on accuracy.
        """
        self.scheduler.step(accuracy)

    def make_gradient_step(self):
        self.optimizer.step()
        # Clear out the gradients before the next batch.
        self.model.zero_grad()


def pad_labels(labels):
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
