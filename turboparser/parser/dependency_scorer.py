from .constants import Target
import torch
from torch.nn import functional as F
import torch.optim as optim
import logging


def get_gold_tensors(instance_data):
    """
    Create a 2d tensor with gold heads for all instances in a batch and another
    with the labels for each one.

    In the heads tensor, each entry [i, j] has the index of the head of the
    j-th word in the i-th instance, or -1 if i has less than j words.

    The relations tensor is similar but indicates the dependency relation for
    j (as a modifier)

    :param instance_data: InstanceData object
    :return: two tensors (batch_size, max_instance_length)
    """
    batch_size = len(instance_data)
    max_length = max(len(inst) for inst in instance_data.instances)
    heads = torch.full([batch_size, max_length], -1, dtype=torch.long)
    relations = torch.full([batch_size, max_length], -1, dtype=torch.long)

    for i, inst in enumerate(instance_data.instances):
        # skip root
        inst_heads = inst.heads[1:]
        inst_relations = inst.relations[1:]
        heads[i, :len(inst_heads)] = inst_heads
        relations[i, :len(inst_relations)] = inst_relations

    return heads


class DependencyNeuralScorer(object):
    """
    Neural scorer for mediating the training of a Parser/Tagger neural model.
    """
    def __init__(self):
        self.part_scores = None
        self.model = None

    def compute_loss(self, instance_data, predicted_parts):
        """
        Compute the losses for parsing and tagging.

        :param instance_data: InstanceData object
        :param predicted_parts: list of numpy arrays with predicted parts for
            each instance
        :return: dictionary mapping each target to a loss scalar, as a torch
            variable
        """
        batch_size = len(instance_data)
        losses = {}
        gold_labels = instance_data.gold_labels

        for target in [Target.UPOS, Target.XPOS, Target.MORPH]:
            if target not in gold_labels[0]:
                continue

            target_gold = [item[target] for item in gold_labels]
            padded_gold = pad_labels(target_gold)
            logits = self.model.scores[target]

            # cross_entropy expects (batch, n_classes, ...)
            logits = logits.transpose(1, 2)
            losses[target] = F.cross_entropy(logits, padded_gold,
                                             ignore_index=-1)

        # dependency parts loss
        parts_loss = torch.tensor(0.)
        if torch.cuda.is_available():
            parts_loss = parts_loss.cuda()

        head_scores = self.model.scores[Target.HEADS]
        label_scores = self.model.scores[Target.RELATIONS]
        sign_scores = self.model.scores['sign']
        distance_kld = self.model.scores['dist_kld']
        gold_heads, gold_relations = get_gold_tensors(instance_data)

        # head loss
        # stack the head predictions for all words from all sentences
        scores2d = head_scores.contiguous().view(-1, head_scores.size(2))
        loss = F.cross_entropy(scores2d, gold_heads.view(-1))

        # label loss
        # make 4d indices to gather what were the predicted deprels for the
        # gold arcs. deprel_scores is (batch, num_words, num_words, num_rel)
        indices = gold_heads.unsqueeze(2).unsqueeze(3)
        num_labels = self.model.label_scorer.out_features
        expanded_indices = indices.expand(-1, -1, -1, num_labels)

        label_scores = torch.gather(label_scores, 2, expanded_indices)
        label_scores = label_scores.view(-1, num_labels)
        loss += F.cross_entropy(label_scores.contiguous(),
                                gold_relations.view(-1))

        # linearization (left/right attachment) loss
        arange = torch.arange(gold_heads.size(1), device=head_scores.device)
        position1 = arange.view(1, 1, -1).expand(batch_size, -1, -1)
        position2 = arange.view(1, -1, 1).expand(batch_size, -1, -1)
        head_offset = position1 - position2

        indices = gold_heads.unsqueeze(2)
        sign_scores = torch.gather(
            sign_scores[:, 1:], 2, indices).view(-1)
        sign_scores = torch.cat([-sign_scores.unsqueeze(1) / 2,
                                 sign_scores.unsqueeze(1) / 2], 1)
        sign_target = torch.gather((head_offset[:, 1:] > 0).long(),
                                   2, indices)
        loss += F.cross_entropy(sign_scores.contiguous(), sign_target.view(-1))

        # distance loss
        distance_kld = torch.gather(distance_kld[:, 1:], 2, indices)
        loss -= distance_kld.sum()

        loss[Target.DEPENDENCY_PARTS] = loss
        # for i in range(batch_size):
        #     inst_parts = instance_data.parts[i]
        #     gold_parts = inst_parts.gold_parts
        #
            # inst_pred = predicted_parts[i]
            # part_score_list = [self.model.scores[type_][i]
            #                    for type_ in inst_parts.type_order]
            # part_scores = torch.cat(part_score_list)
            # diff = torch.tensor(inst_pred - gold_parts, dtype=part_scores.dtype,
            #                     device=part_scores.device)
            # error = torch.dot(part_scores, diff)
            # margin, normalizer = inst_parts.get_margin()
            # inst_parts_loss = margin.dot(inst_pred) + normalizer + error

            # if inst_parts_loss > 0:
            #     parts_loss += inst_parts_loss
            # else:
            #     if inst_parts_loss < -10e-6:
            #         logging.warning(
            #             'Ignoring negative loss: %.6f' % inst_parts_loss.item())

        # losses[Target.DEPENDENCY_PARTS] = parts_loss / batch_size

        return losses

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
            # # dependency part scores are stored as lists of tensors; tags
            # # are singleton tensors
            # if target in [Target.UPOS, Target.XPOS, Target.MORPH]:
            numpy_scores[target] = model_scores[target].detach()\
                .cpu().numpy()
            # else:
            #     numpy_scores[target] = [tensor.detach().cpu().numpy()
            #                             for tensor in model_scores[target]]

        # now convert a dictionary of arrays into a list of dictionaries
        score_list = []
        for i in range(len(instances)):
            instance_scores = {target: numpy_scores[target][i]
                               for target in numpy_scores}
            score_list.append(instance_scores)

        return score_list

    def initialize(self, model, learning_rate=0.001, decay=1,
                   beta1=0.9, beta2=0.95):
        self.set_model(model)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            params, lr=learning_rate, betas=(beta1, beta2))
        self.decay = decay

    def set_model(self, model):
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()

    def switch_to_amsgrad(self, learning_rate=0.001, beta1=0.9, beta2=0.95):
        """
        Switch the optimizer to AMSGrad.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            params, amsgrad=True, lr=learning_rate, betas=(beta1, beta2))

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

    def decrease_learning_rate(self):
        """
        Decrease the optimizer's learning rate by multiplying it to the decay
        factor.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.decay
            logging.info('Setting learning rate to %f' % param_group['lr'])

    def make_gradient_step(self, losses):
        """
        :param losses: dictionary mapping targets to losses
        """
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

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
