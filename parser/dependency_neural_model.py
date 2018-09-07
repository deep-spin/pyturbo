import torch
import torch.nn as nn
from parser.dependency_parts import DependencyPartArc, DependencyParts, \
    DependencyPartNextSibling, DependencyPartGrandparent, \
    DependencyPartGrandSibling
import numpy as np
import pickle

#TODO: maybe this should be elsewhere?
# special pseudo-tokens to index embeddings
# root is not one of these since it is a token in the sentences
NULL_SIBLING = '_NULL_SIBLING_'
special_tokens = [NULL_SIBLING]


class DependencyNeuralModel(nn.Module):
    def __init__(self,
                 token_dictionary,
                 dependency_dictionary,
                 word_embedding_size,
                 tag_embedding_size,
                 distance_embedding_size,
                 hidden_size,
                 num_layers,
                 dropout):
        super(DependencyNeuralModel, self).__init__()
        self.word_embedding_size = word_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.distance_embedding_size = distance_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.on_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(token_dictionary.get_num_forms(),
                                            word_embedding_size)
        self.tag_embeddings = nn.Embedding(token_dictionary.get_num_tags(),
                                           tag_embedding_size)
        if self.distance_embedding_size:
            self.distance_bins = np.array(
                list(range(10)) + list(range(10, 40, 5)) + [40])
            self.distance_embeddings = nn.Embedding(len(self.distance_bins) * 2,
                                                    distance_embedding_size)
        else:
            self.distance_bins = None
            self.distance_embeddings = None

        input_size = word_embedding_size + tag_embedding_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True)
        self.tanh = nn.Tanh()

        # first order
        self.head_projection = self._create_projection()
        self.modifier_projection = self._create_projection()

        #TODO: only create parameters as necessary for the model

        # second order -- grandparent
        self.gp_grandparent_projection = self._create_projection()
        self.gp_head_projection = self._create_projection()
        self.gp_modifier_projection = self._create_projection()

        # second order -- consecutive siblings
        self.sib_head_projection = self._create_projection()
        self.sib_modifier_projection = self._create_projection()
        self.sib_sibling_projection = self._create_projection()
        self.null_sibling_tensor = self._create_special_tensor()

        # third order -- grandsiblings (consecutive)
        self.gsib_head_projection = self._create_projection()
        self.gsib_modifier_projection = self._create_projection()
        self.gsib_sibling_projection = self._create_projection()
        self.gsib_grandparent_projection = self._create_projection()

        if self.distance_embedding_size:
            self.distance_projection = nn.Linear(
                distance_embedding_size,
                hidden_size,
                bias=True)
        else:
            self.distance_projection = None

        self.arc_scorer = self._create_scorer()
        self.sibling_scorer = self._create_scorer()
        self.grandparent_scorer = self._create_scorer()
        self.grandsibling_scorer = self._create_scorer()

        # Clear out the gradients before the next batch.
        self.zero_grad()

    def _create_special_tensor(self, shape=None):
        """
        Create a tensor for representing some special token.

        If shape is None, it will have shape equal to hidden_size.
        """
        if shape is None:
            shape = self.hidden_size

        tensor = torch.randn(shape, requires_grad=True)
        if self.on_gpu:
            tensor = tensor.cuda()
        return 

    def _create_scorer(self, input_size=None):
        """
        Create the weights for scoring a given tensor representation to a
        single number.

        :param input_size: expected input size. If None, self.hidden_units
            is used.
        :return: an nn.Linear object
        """
        if input_size is None:
            input_size = self.hidden_size
        scorer = nn.Linear(input_size, 1, bias=False)

        return scorer

    def _create_projection(self):
        """
        Create the weights for projecting an input from the BiLSTM to a
        feedforward layer.

        :return: an nn.Linear object, mapping an input with 2*hidden_units
            to hidden_units.
        """
        projection = nn.Linear(
            self.hidden_size * 2,
            self.hidden_size,
            bias=False
        )
        return projection

    def save(self, file):
        pickle.dump(self.state_dict(), file)

    def load(self, file):
        state_dict = pickle.load(file)
        self.load_state_dict(state_dict)

    def _compute_first_order_scores(self, states, parts, scores):
        """
        Compute the first order scores and store them in the appropriate
        position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        :param scores: tensor for storing the scores for each part. The
            positions relative to first order features are indexed by the parts
            object. It is modified in-place.
        """
        head_tensors = self.head_projection(states)
        modifier_tensors = self.modifier_projection(states)
        offset, num_arcs = parts.get_offset(DependencyPartArc)

        head_indices = []
        modifier_indices = []
        distance_indices = []

        for part in parts.iterate_over_type(DependencyPartArc):
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            if self.distance_embedding_size:
                if part.modifier > part.head:
                    dist = part.modifier - part.head
                    dist = np.nonzero(dist >= self.distance_bins)[0][-1]
                else:
                    dist = part.head - part.modifier
                    dist = np.nonzero(dist >= self.distance_bins)[0][-1]
                    dist += len(self.distance_bins)

                distance_indices.append(dist)

        # now index all of them to process at once
        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        if self.distance_embedding_size:
            distance_indices = torch.tensor(distance_indices, dtype=torch.long)
            if self.on_gpu:
                distance_indices = distance_indices.cuda()
            
            distances = self.distance_embeddings(distance_indices)
            distance_projections = self.distance_projection(distances)
            distance_projections = distance_projections.view(
                -1, 1, self.hidden_size)

        else:
            distance_projections = 0

        arc_states = self.tanh(heads + modifiers + distance_projections)
        arc_scores = self.arc_scorer(arc_states)
        scores[offset:offset + num_arcs] = arc_scores.view(-1)

    def _compute_grandparent_scores(self, states, parts, scores):
        """
        Compute the grandparent scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        :param scores: tensor for storing the scores for each part. The
            positions relative to the features are indexed by the parts
            object. It is modified in-place.
        """
        head_tensors = self.gp_head_projection(states)
        grandparent_tensors = self.gp_grandparent_projection(states)
        modifier_tensors = self.gp_modifier_projection(states)

        head_indices = []
        modifier_indices = []
        grandparent_indices = []

        for part in parts.iterate_over_type(DependencyPartGrandparent):
            # list all indices, then feed the corresponding tensors to the net
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            grandparent_indices.append(part.grandparent)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        grandparents = grandparent_tensors[grandparent_indices]
        part_states = self.tanh(heads + modifiers + grandparents)
        part_scores = self.grandparent_scorer(part_states)

        offset, size = parts.get_offset(DependencyPartGrandparent)
        scores[offset:offset + size] = part_scores.view(-1)

    def _compute_consecutive_sibling_scores(self, states, parts, scores):
        """
        Compute the consecutive sibling scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        :param scores: tensor for storing the scores for each part. The
            positions relative to the features are indexed by the parts
            object. It is modified in-place.
        """
        head_tensors = self.sib_head_projection(states)
        modifier_tensors = self.sib_modifier_projection(states)
        word_sibling_tensors = self.sib_sibling_projection(states)

        # include the vector for null sibling
        # word_sibling_tensors is (num_words, batch=1, hidden_units)
        sibling_tensors = torch.cat([word_sibling_tensors,
                                     self.null_sibling_tensor.view(1, 1, -1)])

        head_indices = []
        modifier_indices = []
        sibling_indices = []

        for part in parts.iterate_over_type(DependencyPartNextSibling):
            # list all indices to the candidate head/modifier/siblings, then
            # process them all at once for faster execution.
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            if part.sibling == 0:
                # this indicates there's no sibling to the left
                # (to the right, sibling == len(states))
                sibling_indices.append(len(states))
            else:
                sibling_indices.append(part.sibling)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        siblings = sibling_tensors[sibling_indices]
        sibling_states = self.tanh(heads + modifiers + siblings)
        sibling_scores = self.sibling_scorer(sibling_states)

        offset, size = parts.get_offset(DependencyPartNextSibling)
        scores[offset:offset + size] = sibling_scores.view(-1)

    def _compute_grandsibling_scores(self, states, parts, scores):
        """
        Compute the consecutive grandsibling scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        :param scores: tensor for storing the scores for each part. The
            positions relative to the features are indexed by the parts
            object. It is modified in-place.
        """
        head_tensors = self.gsib_head_projection(states)
        modifier_tensors = self.gsib_modifier_projection(states)
        word_sibling_tensors = self.gsib_sibling_projection(states)
        grandparent_tensors = self.gsib_grandparent_projection(states)

        # include the vector for null sibling
        # word_sibling_tensors is (num_words, batch=1, hidden_units)
        sibling_tensors = torch.cat([word_sibling_tensors,
                                    self.null_sibling_tensor.view(1, 1, -1)])

        head_indices = []
        modifier_indices = []
        sibling_indices = []
        grandparent_indices = []

        for part in parts.iterate_over_type(DependencyPartGrandSibling):
            # list all indices to the candidate head/mod/sib/grandparent
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            grandparent_indices.append(part.grandparent)
            if part.sibling == 0:
                # this indicates there's no sibling to the left
                # (to the right, sibling == len(states))
                sibling_indices.append(len(states))
            else:
                sibling_indices.append(part.sibling)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        siblings = sibling_tensors[sibling_indices]
        grandparents = grandparent_tensors[grandparent_indices]
        gsib_states = self.tanh(heads + modifiers + siblings + grandparents)
        gsib_scores = self.grandsibling_scorer(gsib_states)

        offset, size = parts.get_offset(DependencyPartGrandSibling)
        scores[offset:offset + size] = gsib_scores.view(-1)

    def forward(self, instance, parts):
        scores = torch.zeros(len(parts))

        word_indices = [instance.get_form(i) for i in range(len(instance))]
        tag_indices = [instance.get_tag(i) for i in range(len(instance))]
        words = torch.tensor(word_indices).cuda()
        tags = torch.tensor(tag_indices).cuda()
        embeds = torch.cat([self.word_embeddings(words),
                            self.tag_embeddings(tags)],
                           dim=1)

        # new shape is (num_tokens, batch=1, hidden_size)
        states, _ = self.rnn(embeds.view(len(instance), 1, -1))

        self._compute_first_order_scores(states, parts, scores)
        if parts.has_type(DependencyPartNextSibling):
            self._compute_consecutive_sibling_scores(states, parts, scores)

        if parts.has_type(DependencyPartGrandparent):
            self._compute_grandparent_scores(states, parts, scores)

        if parts.has_type(DependencyPartGrandSibling):
            self._compute_grandsibling_scores(states, parts, scores)

        return scores
