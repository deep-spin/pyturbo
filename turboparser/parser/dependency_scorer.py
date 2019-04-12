from .constants import Target
from .dependency_parts import type2target
import torch
import numpy as np
from torch.nn import functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


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
        # max_length = max(len(parts) for parts in instance_data.parts)
        # shape = [batch_size, max_length]
        # diff = torch.zeros(shape, dtype=torch.float)
        parts_loss = torch.tensor(0)
        for i in range(batch_size):
            inst_parts = instance_data.parts[i]
            gold_parts = inst_parts.gold_parts
            pred_item = predicted_parts[i]
            inst_scores = self.model.scores
            part_scores = extract_parts_scores(inst_parts, inst_scores, i)
            diff = torch.tensor(pred_item - gold_parts)
            parts_loss += torch.dot(part_scores, diff)
            # diff[i, :len(gold_parts)] = torch.tensor(pred_item - gold_parts)

        parts_loss /= batch_size
        loss += parts_loss.to(loss.device)

        # Backpropagate to accumulate gradients.
        loss.backward()

        return parts_loss

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
        numpy_scores = {target: model_scores[target].detach().cpu().numpy()
                        for target in model_scores}

        # now convert a dictionary of arrays into a list of dictionaries
        score_list = []
        for i in range(len(instances)):
            instance_scores = {target: numpy_scores[target][i]
                               for target in numpy_scores}
            #TODO: make this cleaner
            shape = parts[i].arc_mask.shape
            arc_scores = instance_scores[Target.HEADS]
            instance_scores[Target.HEADS] = arc_scores[:shape[0], :shape[1]]

            if Target.RELATIONS in instance_scores:
                label_scores = instance_scores[Target.RELATIONS]
                instance_scores[Target.RELATIONS] = label_scores[:shape[0],
                                                                 :shape[1]]
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


def extract_parts_scores(parts, scores, i):
    """
    Given a list of dictionary of target scores, produce a single array with
    scores for all parts in instance i.

    :param scores: dicitonary mapping targets such as heads, siblings, etc
        to arrays of scores
    :return: numpy 1d array
    """
    arc_scores = scores[Target.HEADS][i]

    # in case there was padding, take only the valid scores
    shape = parts.arc_mask.shape
    arc_scores = arc_scores[:shape[0], :shape[1]]
    arc_scores = arc_scores[parts.arc_mask]

    part_scores = [arc_scores]
    if parts.labeled:
        labeled_scores = scores[Target.RELATIONS][i]
        labeled_scores = labeled_scores[:shape[0], :shape[1]]
        labeled_scores = labeled_scores[parts.arc_mask]
        part_scores.append(labeled_scores.reshape(-1))

    for type_ in parts.part_lists:
        target = type2target[type_]
        part_scores.append(scores[target][i])

    return np.concatenate(part_scores)
