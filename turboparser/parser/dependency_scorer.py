import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import time

from .constants import Target, dependency_targets, higher_order_parts, \
    structured_objectives
from .constants import ParsingObjective as Objective
from . import decoding
from ..classifier.utils import get_logger
from ..classifier.instance import InstanceData


logger = get_logger()


def get_gold_tensors(instance_data):
    """
    Create a 2d tensor with gold heads for all instances in a batch and another
    with the labels for each one.

    In the heads tensor, each entry [i, j] has the index of the head of the
    j-th word in the i-th instance, or -1 if i has less than j words.

    The relations tensor is similar but indicates the dependency relation for
    j (as a modifier)

    :param instance_data: InstanceData object
    :return: two tensors (batch_size, max_num_actual_words)
    """
    batch_size = len(instance_data)
    max_length = max(len(inst) for inst in instance_data.instances)

    # -1 to skip root
    heads = torch.full([batch_size, max_length - 1], -1, dtype=torch.long)
    relations = torch.full([batch_size, max_length - 1], -1, dtype=torch.long)

    for i, inst in enumerate(instance_data.instances):
        # skip root
        inst_heads = inst.heads[1:]
        inst_relations = inst.relations[1:]
        heads[i, :len(inst_heads)] = torch.tensor(inst_heads, dtype=torch.long)
        relations[i, :len(inst_relations)] = torch.tensor(inst_relations,
                                                          dtype=torch.long)

    if torch.cuda.is_available():
        heads = heads.cuda()
        relations = relations.cuda()

    return heads, relations


class DependencyNeuralScorer(object):
    """
    Neural scorer for mediating the training of a Parser/Tagger neural model.
    """
    def __init__(self):
        self.part_scores = None
        self.model = None
        self.time_decoding = 0.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def compute_loss_global_margin(self, instance_data, all_predicted_parts):
        """
        Compute the losses for parsing and tagging.

        :param instance_data: InstanceData object
        :param all_predicted_parts: list of numpy arrays with predicted parts
            for each instance
        :return: dictionary mapping each target to a loss scalar, as a torch
            variable
        """
        losses = {}

        # dependency parts loss
        parts_loss = torch.tensor(0.)
        if torch.cuda.is_available():
            parts_loss = parts_loss.cuda()

        batch_size = len(instance_data)
        for i in range(batch_size):
            inst_parts = instance_data.parts[i]
            part_score_list = [self.model.scores[type_][i]
                               for type_ in inst_parts.type_order]
            part_scores = torch.cat(part_score_list)

            gold_parts = inst_parts.gold_parts
            pred_parts = all_predicted_parts[i]
            diff_predictions = torch.tensor(pred_parts - gold_parts,
                                            dtype=torch.float,
                                            device=parts_loss.device)
            inst_parts_loss = diff_predictions.dot(part_scores)

            if inst_parts_loss > 0:
                parts_loss += inst_parts_loss
            else:
                if inst_parts_loss < -10e-6:
                    logger.warning(
                        'Ignoring negative loss: %.6f' % inst_parts_loss.item())

        losses[Target.DEPENDENCY_PARTS] = parts_loss / batch_size

        return losses

    def compute_loss_global_probability(self, instance_data: InstanceData,
                                        predicted_parts: list):
        """
        Compute the parsing loss as the cross-entropy of the correct parse tree
        with respect to all possible parse trees, using the Matrix-Tree Theorem.
        """
        # entropy is (batch_size,)
        entropy = torch.tensor(self.entropies)

        # this is a list of lists of scores for each part
        score_list = [self.model.scores[type_]
                      for type_ in instance_data.parts[0].type_order]

        # turn it into a list of single score tensors for each instance
        score_list = [torch.cat(instance_scores)
                      for instance_scores in zip(*score_list)]

        # pad it to be (batch_size, num_parts)
        part_scores = pad_sequence(score_list, batch_first=True)
        gold_part_list = [torch.tensor(parts.gold_parts, dtype=torch.float,
                                       device=part_scores.device)
                          for parts in instance_data.parts]
        gold_parts = pad_sequence(gold_part_list, batch_first=True)
        predicted_parts = [torch.tensor(parts, dtype=torch.float,
                                        device=part_scores.device)
                           for parts in predicted_parts]
        predicted_parts = pad_sequence(predicted_parts, batch_first=True)

        # diff is (batch_size, num_parts)
        diff = predicted_parts - gold_parts

        # loss = entropy + sum_i(scores_i * (predicted_i - gold_i))
        # compute the loss for each instance so we can check for < 0
        losses = entropy.view(-1, 1) + (part_scores * diff).sum(1)

        inds_subzero = losses < 0
        if inds_subzero.sum().item():
            values = losses[inds_subzero]
            for value in values:
                value = value.item()
                if value < -1e-6:
                    logger.warning('Ignoring negative loss: %.6f' % value)
            losses[inds_subzero] = 0

        losses = {Target.DEPENDENCY_PARTS: losses.mean()}

        return losses

    def compute_loss(self, instance_data, predicted_parts=None):
        """
        Compute the losses for parsing and tagging. The appropriate function
        will be called depending on local or global normalization.

        :param instance_data: InstanceData object
        :param predicted_parts: list of numpy arrays with predicted parts for
            each instance
        :return: dictionary mapping each target to a loss scalar, as a torch
            variable
        """
        # compute loss for POS tagging, morphology and lemmas
        losses = self.compute_tagging_loss(instance_data)

        if self.model.predict_tree:
            # loss for dependency parsing
            gold_heads, gold_relations = get_gold_tensors(instance_data)
            if self.parsing_loss == Objective.GLOBAL_MARGIN:
                dep_losses = self.compute_loss_global_margin(instance_data,
                                                             predicted_parts)
            elif self.parsing_loss == Objective.LOCAL:
                dep_losses = self.compute_loss_local(gold_heads, gold_relations)

            elif self.parsing_loss == Objective.GLOBAL_PROBABILITY:
                dep_losses = self.compute_loss_global_probability(
                    instance_data, predicted_parts)
            else:
                msg = 'Unknown parsing loss: %s' % self.parsing_loss
                raise ValueError(msg)

            losses.update(dep_losses)

            # loss for the (head, modifier) distances used in parsing
            positional_losses = self.compute_loss_position(gold_heads)
            losses.update(positional_losses)

        return losses

    def compute_tagging_loss(self, instance_data):
        """
        Compute losses of tagging tasks. No structure is considered, only
        the cross-entropy.

        :param instance_data: InstanceData object
        :return: dictionary mapping each target to a loss scalar, as a torch
            variable
        """
        gold_labels = instance_data.gold_labels
        losses = {}

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

        if Target.LEMMA in gold_labels[0]:
            gold = self.model.lemmatizer.cached_gold_chars
            token_inds = self.model.lemmatizer.cached_real_token_inds
            logits = self.model.scores[Target.LEMMA]
            batch_size, num_words, num_chars, vocab_size = logits.shape
            gold = gold.view(-1)
            logits = logits.view(batch_size * num_words, num_chars, vocab_size)
            logits = logits[token_inds].view(-1, vocab_size)
            losses[Target.LEMMA] = F.cross_entropy(logits, gold, ignore_index=0)

        return losses

    def compute_loss_local(self, gold_heads, gold_relations):
        """
        Compute the losses for parsing, treating each word as an independent
        instance.

        :param gold_heads: tensor (batch, num_words) with gold heads
        :return: dictionary mapping each target to a loss scalar, as a torch
            variable
        """
        losses = {}

        head_scores = self.model.scores[Target.HEADS]
        label_scores = self.model.scores[Target.RELATIONS]

        # head loss
        # stack the head predictions for all words from all sentences
        scores2d = head_scores.contiguous().view(-1, head_scores.size(2))
        loss = self.loss_fn(scores2d, gold_heads.view(-1))

        # label loss
        # avoid -1 in gather
        heads3d = gold_heads.unsqueeze(2)
        negative_inds = heads3d == -1
        heads3d = heads3d.masked_fill(negative_inds, 0)
        heads4d = heads3d.unsqueeze(3)

        # make 4d indices to gather what were the predicted deprels for the
        # gold arcs. deprel_scores is (batch, num_words, num_words, num_rel)
        num_labels = self.model.label_scorer.output_size
        expanded_indices = heads4d.expand(-1, -1, -1, num_labels)

        label_scores = torch.gather(label_scores, 2, expanded_indices)
        label_scores = label_scores.view(-1, num_labels)
        label_loss = self.loss_fn(label_scores.contiguous(),
                                  gold_relations.view(-1))
        loss += label_loss
        losses[Target.DEPENDENCY_PARTS] = loss

        return losses

    def compute_loss_position(self, gold_heads):
        """
        Compute the loss with respect to the relative position of heads and
        modifiers. This is only used for first order parts.

        :param gold_heads: a tensor (batch, num_actual_words) such that position
            (i, j) has the head of word j in the i-th sentence in the batch.
        """
        heads3d = gold_heads.unsqueeze(2)
        padding_inds = heads3d == -1
        heads3d = heads3d.masked_fill(padding_inds, 0)

        sign_scores = self.model.scores[Target.SIGN]
        distance_kld = self.model.scores[Target.DISTANCE]
        batch_size = len(sign_scores)
        heads3d = heads3d.to(sign_scores.device)

        # linearization (left/right attachment) loss
        arange = torch.arange(sign_scores.size(2), device=sign_scores.device)
        position1 = arange.view(1, 1, -1).expand(batch_size, -1, -1)
        position2 = arange.view(1, -1, 1).expand(batch_size, -1, -1)
        head_offset = position1 - position2
        head_offset = head_offset[:, 1:]  # exclude root

        # get the head scores for the gold heads
        head_sign_scores = torch.gather(sign_scores, 2, heads3d).view(-1)
        head_sign_scores = head_sign_scores.unsqueeze(1) / 2
        head_sign_scores = torch.cat([-head_sign_scores, head_sign_scores], 1)

        sign_target = torch.gather((head_offset > 0).long(), 2, heads3d)
        sign_target[padding_inds] = -1  # -1 to padding
        sign_loss = self.loss_fn(head_sign_scores.contiguous(),
                                 sign_target.view(-1))

        # distance loss
        distance_kld = torch.gather(distance_kld, 2, heads3d)
        distance_kld[padding_inds] = 0
        kld_sum = distance_kld.mean()

        losses = {Target.DISTANCE: -kld_sum, Target.SIGN: sign_loss}

        return losses

    def compute_scores(self, instance_data):
        """
        Compute the scores for all the targets this scorer

        :param instance_data: InstanceData
        :return: a list of dictionaries mapping each target name to its scores
        """
        model_scores = self.model(instance_data.instances, instance_data.parts,
                                  self.parsing_loss)

        detached_scores = {}
        for target in model_scores:
            # distance and higher order scores are not necessary outside
            # the graph
            if target == Target.DISTANCE or target == Target.SIGN:
                continue

            value = model_scores[target]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            elif isinstance(value, list):
                # structured prediction stores part lists of different sizes
                value = [item.detach().cpu() for item in value]

            detached_scores[target] = value

        return detached_scores

    def predict(self, instance_data: InstanceData, single_root: bool = True,
                num_jobs: int = 2):
        """
        Predict the outputs for the given data, running decoding when necessary.

        :param instance_data: data to predict outputs for
        :param single_root: whether to enforce a single sentence root. Only used
            if parsing
        :param num_jobs: in case of higher order dependency decoding, number of
            parallel jobs to launch
        :return: a tuple (final predictions, predicted parts)
            final predictions is a list of dictionaries that map each target
            to the predictions
        """
        # scores is a dict[Target] -> batched arrays
        scores = self.compute_scores(instance_data)
        tagging_predictions = {}
        output = []
        parse = Target.HEADS in scores
        head_scores = None
        label_scores = None
        predicted_parts = []
        num_instances = len(instance_data)
        if self.parsing_loss == Objective.GLOBAL_PROBABILITY:
            self.entropies = []

        for target in scores:
            target_scores = scores[target]

            if target not in dependency_targets:
                # tagging and lemmatization

                # at training time:
                # tags: (batch, num_words, label_logits)
                # lemmas: (batch, num_words, num_chars, char_vocab_logits)
                # at inference time, lemmas is (batch, num_words, num_chars)
                # because we use search algorithms over the output space
                if target != Target.LEMMA or self.model.training:
                    tagging_predictions[target] = target_scores.argmax(-1).\
                        numpy()
                else:
                    tagging_predictions[target] = target_scores

            elif self.parsing_loss == Objective.LOCAL:
                # take the log probability of the heads to run MST later
                # and the most likely label for each (h, m)
                if target == Target.HEADS:
                    head_scores = F.log_softmax(target_scores, -1)
                elif target == Target.RELATIONS:
                    label_scores = target_scores.argmax(-1)

        # it may seem counter intuitive to loop through the scores to create a
        # dictionary for each instance with part scores and later loop again
        # to create a dictionary with parse trees, but this allows us to take
        # advantage of batched parallel decoding

        if parse and self.parsing_loss in structured_objectives:
            # decode parts for all sentences
            start_decoding = time.time()
            part_types = higher_order_parts.union({Target.HEADS,
                                                   Target.RELATIONS})
            part_scores = [{target: scores[target][i].numpy()
                            for target in scores
                            if target in part_types}
                           for i in range(num_instances)]

            if self.parsing_loss == Objective.GLOBAL_MARGIN:
                predicted_parts = decoding.batch_decode(
                    instance_data, part_scores, num_jobs)
            elif self.parsing_loss == Objective.GLOBAL_PROBABILITY:
                for i in range(num_instances):
                    instance_parts = instance_data.parts[i]
                    instance_scores = part_scores[i]
                    result = decoding.decode_marginals(
                        instance_parts, instance_scores)
                    instance_prediction, _, entropy = result
                    predicted_parts.append(instance_prediction)
                    self.entropies.append(entropy)

            end_decoding = time.time()
            self.time_decoding += end_decoding - start_decoding

        # now create a list with a dictionary for each instance
        for i in range(num_instances):
            length = len(instance_data.instances[i])
            instance_output = {}

            for target in tagging_predictions:
                if target not in dependency_targets:
                    target_prediction = tagging_predictions[target]
                    instance_output[target] = target_prediction[i][:length]

            # no need to decode the final parse during training
            if parse and not self.model.training:
                # decode the parse tree here
                if predicted_parts is not None:
                    # global objective, either with hinge loss or probability
                    instance_predicted_parts = predicted_parts[i]
                    heads, labels = decoding.decode_predictions(
                        instance_predicted_parts, instance_data.parts[i],
                        single_root=single_root)
                    instance_output[Target.DEPENDENCY_PARTS] = \
                        instance_predicted_parts
                else:
                    instance_head_scores = head_scores[i, :length - 1, :length]
                    instance_label_scores = label_scores[
                                            i, :length - 1, :length]
                    heads, labels = decoding.decode_predictions(
                        head_score_matrix=instance_head_scores,
                        label_matrix=instance_label_scores,
                        single_root=single_root)
                    instance_output[Target.DEPENDENCY_PARTS] = None

                instance_output[Target.HEADS] = heads
                instance_output[Target.RELATIONS] = labels

            elif self.parsing_loss in structured_objectives:
                instance_output[Target.DEPENDENCY_PARTS] = predicted_parts[i]

            output.append(instance_output)

        return output

    def initialize(self, model, parsing_loss=Objective.GLOBAL_MARGIN,
                   learning_rate=0.001, decay=1, beta1=0.9, beta2=0.95):
        self.set_model(model)
        self.parsing_loss = parsing_loss
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            params, lr=learning_rate, betas=(beta1, beta2), eps=1e-6)
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
        # self.optimizer = optim.SGD(params, lr=learning_rate)

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
            logger.info('Setting learning rate to %f' % param_group['lr'])

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
