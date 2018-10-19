import torch
import torch.nn as nn
from .dependency_parts import Arc, DependencyParts, NextSibling, Grandparent, \
    GrandSibling, LabeledArc
import numpy as np

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
                 rnn_size,
                 mlp_size,
                 num_layers,
                 dropout):
        super(DependencyNeuralModel, self).__init__()
        self.word_embedding_size = word_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.distance_embedding_size = distance_embedding_size
        self.rnn_size = rnn_size
        self.mlp_size = mlp_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_labels = len(dependency_dictionary.relation_alphabet)
        self.padding = token_dictionary.token_padding
        self.on_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(token_dictionary.get_num_forms(),
                                            word_embedding_size)

        if self.tag_embedding_size:
            self.tag_embeddings = nn.Embedding(token_dictionary.get_num_tags(),
                                               tag_embedding_size)
        else:
            self.tag_embeddings = None

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
            hidden_size=rnn_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True)
        self.rnn_hidden_size = 2 * rnn_size
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

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
                mlp_size,
                bias=True)
        else:
            self.distance_projection = None

        self.arc_scorer = self._create_scorer()
        self.label_scorer = self._create_scorer(output_size=self.num_labels)
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
            shape = self.rnn_hidden_size

        tensor = torch.randn(shape, requires_grad=True)
        if self.on_gpu:
            tensor = tensor.cuda()

        return tensor

    def _create_scorer(self, input_size=None, output_size=1):
        """
        Create the weights for scoring a given tensor representation to a
        single number.

        :param input_size: expected input size. If None, self.hidden_units
            is used.
        :return: an nn.Linear object
        """
        if input_size is None:
            input_size = self.mlp_size
        linear = nn.Linear(input_size, output_size, bias=False)
        scorer = nn.Sequential(self.dropout, linear)

        return scorer

    def _create_projection(self):
        """
        Create the weights for projecting an input from the BiLSTM to a
        feedforward layer.

        :return: an nn.Linear object, mapping an input with 2*hidden_units
            to hidden_units.
        """
        linear = nn.Linear(self.rnn_size * 2, self.mlp_size, bias=False)
        projection = nn.Sequential(self.dropout, linear)

        return projection

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        if self.on_gpu:
            state_dict = torch.load(file)
        else:
            state_dict = torch.load(file, map_location='cpu')

        self.load_state_dict(state_dict)

    def _compute_arc_scores(self, states, parts, scores):
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
        offset, num_arcs = parts.get_offset(Arc)

        head_indices = []
        modifier_indices = []
        distance_indices = []

        for part in parts.iterate_over_type(Arc):
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
                -1, self.mlp_size)

        else:
            distance_projections = 0

        arc_states = self.tanh(heads + modifiers + distance_projections)
        arc_scores = self.arc_scorer(arc_states)
        scores[offset:offset + num_arcs] = arc_scores.view(-1)

        if parts.has_type(LabeledArc):
            # compute label scores and place them in the correct position
            label_scores = self.label_scorer(arc_states)
            


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

        for part in parts.iterate_over_type(Grandparent):
            # list all indices, then feed the corresponding tensors to the net
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            grandparent_indices.append(part.grandparent)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        grandparents = grandparent_tensors[grandparent_indices]
        part_states = self.tanh(heads + modifiers + grandparents)
        part_scores = self.grandparent_scorer(part_states)

        offset, size = parts.get_offset(Grandparent)
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
        # include the vector for null sibling
        # word_sibling_tensors is (num_words=1, hidden_units)
        states_and_sibling = torch.cat([states,
                                        self.null_sibling_tensor.view(1, -1)])

        head_tensors = self.sib_head_projection(states)
        modifier_tensors = self.sib_modifier_projection(states)
        sibling_tensors = self.sib_sibling_projection(states_and_sibling)

        head_indices = []
        modifier_indices = []
        sibling_indices = []

        for part in parts.iterate_over_type(NextSibling):
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

        offset, size = parts.get_offset(NextSibling)
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
        # include the vector for null sibling
        # word_sibling_tensors is (num_words=1, hidden_units)
        states_and_sibling = torch.cat([states,
                                        self.null_sibling_tensor.view(1, -1)])

        head_tensors = self.gsib_head_projection(states)
        modifier_tensors = self.gsib_modifier_projection(states)
        sibling_tensors = self.gsib_sibling_projection(states_and_sibling)
        grandparent_tensors = self.gsib_grandparent_projection(states)

        head_indices = []
        modifier_indices = []
        sibling_indices = []
        grandparent_indices = []

        for part in parts.iterate_over_type(GrandSibling):
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

        offset, size = parts.get_offset(GrandSibling)
        scores[offset:offset + size] = gsib_scores.view(-1)

    def _get_embeddings(self, instances, max_length, word_or_tag):
        """
        Get the word or tag embeddings for all tokens in the instances.

        This function takes care of padding.

        :param word_or_tag: either 'word' or 'tag'
        """
        index_matrix = torch.full((len(instances), max_length), self.padding,
                                  dtype=torch.long)
        for i, instance in enumerate(instances):
            if word_or_tag == 'word':
                getter = instance.get_form
            else:
                getter = instance.get_tag

            indices = [getter(j) for j in range(len(instance))]
            index_matrix[i, :len(instance)] = torch.tensor(indices)

        if self.on_gpu:
            index_matrix = index_matrix.cuda()

        if word_or_tag == 'word':
            embedding_matrix = self.word_embeddings
        else:
            embedding_matrix = self.tag_embeddings

        return embedding_matrix(index_matrix)

    def forward(self, instances, parts):
        """
        :param instances: a list of DependencyInstance objects
        :param parts: a list of DependencyParts objects
        :return: a score matrix with shape (num_instances, longest_length)
        """
        batch_size = len(instances)
        lengths = torch.tensor([len(instance) for instance in instances],
                               dtype=torch.long)
        # packed sequences must be sorted by decreasing length
        lengths, inds = lengths.sort(descending=True)
        if self.on_gpu:
            lengths = lengths.cuda()

        # instances = [instances[i] for i in inds]
        max_length = lengths[0].item()
        max_num_parts = max(len(p) for p in parts)
        batch_scores = torch.zeros(batch_size, max_num_parts)

        embeddings = self._get_embeddings(instances, max_length, 'word')
        if self.tag_embeddings is not None:
            tag_embeddings = self._get_embeddings(instances, max_length, 'tag')

            # each embedding tensor is (batch, num_tokens, embedding_size)
            embeddings = torch.cat([embeddings, tag_embeddings], dim=2)

        sorted_embeddings = embeddings[inds]

        # pack to account for variable lengths
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            sorted_embeddings, lengths, batch_first=True)

        # batch_states is (batch, num_tokens, hidden_size)
        batch_packed_states, _ = self.rnn(packed_embeddings)
        batch_states, _ = nn.utils.rnn.pad_packed_sequence(
            batch_packed_states, batch_first=True)

        # now go through each batch item
        for i in range(batch_size):
            # i points to positions in the original instances and parts
            # inds[i] points to the corresponding position in the RNN output
            rnn_ind = inds[i]

            # we will set the scores corresponding to the i-th item in the
            # batch, or the rnn_ind-th item in the instances

            length = lengths[i].item()
            states = batch_states[i, :length]
            scores = batch_scores[rnn_ind]
            sent_parts = parts[rnn_ind]

            self._compute_arc_scores(states, sent_parts, scores)

            if sent_parts.has_type(NextSibling):
                self._compute_consecutive_sibling_scores(states, sent_parts,
                                                         scores)

            if sent_parts.has_type(Grandparent):
                self._compute_grandparent_scores(states, sent_parts, scores)

            if sent_parts.has_type(GrandSibling):
                self._compute_grandsibling_scores(states, sent_parts, scores)

        return batch_scores
