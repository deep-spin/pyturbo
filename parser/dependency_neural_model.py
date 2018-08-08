import torch
import torch.nn as nn
from parser.dependency_parts import DependencyPartArc, DependencyParts, \
    DependencyPartConsecutiveSibling, DependencyPartGrandparent
from parser.turbo_parser import special_tokens
import numpy as np
import pickle


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

        num_embeddings = token_dictionary.get_num_forms() + len(special_tokens)
        self.word_embeddings = nn.Embedding(num_embeddings, word_embedding_size)

        num_embeddings = token_dictionary.get_num_tags() + len(special_tokens)
        self.tag_embeddings = nn.Embedding(num_embeddings, tag_embedding_size)
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

        # second order -- grandparent
        self.gp_grandparent_projection = self._create_projection()
        self.gp_parent_projection = self._create_projection()
        self.gp_grandchild_projection = self._create_projection()

        # second order -- consecutive siblings
        self.sib_head_projection = self._create_projection()
        self.sib_modifier_projection = self._create_projection()
        self.sib_sibling_projection = self._create_projection()

        if self.distance_embedding_size:
            self.distance_projection = nn.Linear(
                distance_embedding_size,
                hidden_size,
                bias=True)
        else:
            self.distance_projection = None

        self.arc_scorer = self._create_scorer()
        self.sibling_scorer = self._create_scorer()

        # Clear out the gradients before the next batch.
        self.zero_grad()

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
        :param scores: tensor for storing the scores for each part. The
            positions relative to first order features are indexed by the parts
            object. It is modified in-place.
        """
        heads = self.head_projection(states)
        modifiers = self.modifier_projection(states)
        offset, size = parts.get_offset(DependencyPartArc)
        for r in range(offset, offset + size):
            arc = parts[r]
            if self.distance_embedding_size:
                if arc.modifier > arc.head:
                    dist = arc.modifier - arc.head
                    dist = np.nonzero(dist >= self.distance_bins)[0][-1]
                else:
                    dist = arc.head - arc.modifier
                    dist = np.nonzero(dist >= self.distance_bins)[0][-1]
                    dist += len(self.distance_bins)
                dist = torch.tensor(dist, dtype=torch.long)
                dist_embed = self.distance_embeddings(dist).view(1, -1)
                arc_state = self.tanh(heads[arc.head] + \
                                      modifiers[arc.modifier] + \
                                      self.distance_projection(dist_embed))
            else:
                arc_state = self.tanh(heads[arc.head] + \
                                      modifiers[arc.modifier])
            scores[r] = self.arc_scorer(arc_state)

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
        sibling_tensors = self.sib_sibling_projection(states)

        head_indices = []
        modifier_indices = []
        sibling_indices = []

        for part in parts.iterate_over_type(DependencyPartConsecutiveSibling):
            # list all indices to the candidate head/modifier/siblings, then
            # process them all at once for faster execution.
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            sibling_indices.append(part.sibling)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        siblings = sibling_tensors[sibling_indices]
        sibling_states = self.tanh(heads + modifiers + siblings)
        sibling_scores = self.sibling_scorer(sibling_states)

        offset, size = parts.get_offset(DependencyPartConsecutiveSibling)
        scores[offset:offset + size] = sibling_scores.view(-1)

    def forward(self, instance, parts):
        scores = torch.zeros(len(parts))

        word_indices = [instance.get_form(i) for i in range(len(instance))]
        tag_indices = [instance.get_tag(i) for i in range(len(instance))]
        words = torch.tensor(word_indices, dtype=torch.long)
        tags = torch.tensor(tag_indices, dtype=torch.long)
        embeds = torch.cat([self.word_embeddings(words),
                            self.tag_embeddings(tags)],
                           dim=1)
        states, _ = self.rnn(embeds.view(len(instance), 1, -1))

        self._compute_first_order_scores(states, parts, scores)
        self._compute_consecutive_sibling_scores(states, parts, scores)
        return scores
