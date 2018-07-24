import torch
import torch.nn as nn
from parser.dependency_parts import DependencyPartArc
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
        self.head_projection = nn.Sequential(nn.Linear(
            hidden_size * 2,
            hidden_size,
            bias=False))
        self.modifier_projection = nn.Sequential(nn.Linear(
            hidden_size * 2,
            hidden_size,
            bias=False))
        if self.distance_embedding_size:
            self.distance_projection = nn.Linear(
                distance_embedding_size,
                hidden_size,
                bias=True)
        else:
            self.distance_projection = None
        self.arc_scorer = nn.Linear(hidden_size, 1, bias=False)
        # Clear out the gradients before the next batch.
        self.zero_grad()

    def save(self, file):
        pickle.dump(self.state_dict(), file)

    def load(self, file):
        state_dict = pickle.load(file)
        self.load_state_dict(state_dict)

    def forward(self, instance, parts):
        word_indices = [instance.get_form(i) for i in range(len(instance))]
        tag_indices = [instance.get_tag(i) for i in range(len(instance))]
        #print(len(word_indices))
        #print(word_indices)
        words = torch.tensor(word_indices, dtype=torch.long)
        tags = torch.tensor(tag_indices, dtype=torch.long)
        embeds = torch.cat([self.word_embeddings(words),
                            self.tag_embeddings(tags)],
                           dim=1)
        states, _ = self.rnn(embeds.view(len(instance), 1, -1))
        heads = self.head_projection(states)
        modifiers = self.modifier_projection(states)
        scores = torch.zeros(len(parts))
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
        return scores
