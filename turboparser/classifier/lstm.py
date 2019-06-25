import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.LSTM):
    """
    A wrapper of the torch LSTM with a few built-in functionalities.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0,
                 bidirectional=True):
        super(LSTM, self).__init__(
            input_size=input_size, hidden_size=hidden_size,
            bidirectional=bidirectional, batch_first=True,
            num_layers=num_layers, dropout=dropout)

        num_directions = 2 if bidirectional else 1
        # shape is (num_layers * num_directions, batch, num_units)
        shape = (num_layers * num_directions, 1, hidden_size)
        self.initial_h = nn.Parameter(torch.zeros(shape))
        self.initial_c = nn.Parameter(torch.zeros(shape))

    def forward(self, x, hx=None):
        """
        :param x: a packed padded sequence
        :param hx: unused
        """
        # the packed padded sequence is (actual data, lengths)
        # lengths is sorted descending
        batch_size = x[1][0].item()

        # initial states must be (num_directions * layers, batch, num_units)
        batch_initial_h = self.initial_h.expand(-1, batch_size, -1).contiguous()
        batch_initial_c = self.initial_c.expand(-1, batch_size, -1).contiguous()

        return super(LSTM, self).forward(x, (batch_initial_h, batch_initial_c))


class CharLSTM(nn.Module):
    """
    A wrapper of the torch LSTM with character-based functionalities.
    """
    def __init__(self, char_vocab_size, embedding_size, hidden_size,
                 num_layers=1, dropout=0, rec_dropout=0, attention=True,
                 bidirectional=True):
        super(CharLSTM, self).__init__()
        self.attention = attention
        self.embeddings = nn.Embedding(char_vocab_size, embedding_size)
        self.num_directions = 2 if bidirectional else 1
        if attention:
            # 2 times because bidirectional
            combined_hidden = self.num_directions * hidden_size
            self.attention_layer = nn.Linear(combined_hidden, 1, False)
            nn.init.zeros_(self.attention_layer.weight)

        self.lstm = LSTM(embedding_size, hidden_size, num_layers, rec_dropout,
                         bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_indices, token_lengths):
        """
        :param char_indices: tensor (batch, max_sequence_length,
            max_token_length)
        :param token_lengths: tensor (batch, max_sequence_length)
        :return: a tensor with shape (batch, max_sequence_length, 2*hidden_size)
        """
        batch_size, max_sentence_length, max_token_length = char_indices.shape

        # now we have a 3d matrix with char indices. let's reshape it to 2d,
        # stacking all tokens with no sentence separation
        new_shape = [batch_size * max_sentence_length, max_token_length]
        char_indices = char_indices.view(new_shape)
        lengths1d = token_lengths.view(-1)

        # now order by descending length and keep track of the originals
        sorted_lengths, sorted_inds = lengths1d.sort(descending=True)

        # we can't pass 0-length tensors to the LSTM (they're the padding)
        nonzero = sorted_lengths > 0
        sorted_lengths = sorted_lengths[nonzero]
        sorted_inds = sorted_inds[nonzero]

        sorted_token_inds = char_indices[sorted_inds]
        if torch.cuda.is_available():
            sorted_token_inds = sorted_token_inds.cuda()

        # embedded is [batch * max_sentence_len, max_token_len, char_embedding]
        embedded = self.dropout(self.embeddings(sorted_token_inds))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths,
                                                   batch_first=True)
        outputs, (last_output, cell) = self.lstm(packed)

        if self.attention:
            # first, pad the packed sequence
            padded_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, True)

            # apply attention on the outputs of all time steps
            # attention is (batch, max_token_len, 1)
            raw_attention = self.attention_layer(self.dropout(padded_outputs))
            attention = torch.sigmoid(raw_attention)

            # TODO: use actual attention instead of just sigmoid
            attended = padded_outputs * attention
            last_output_bi = attended.sum(1)
        else:
            # concatenate the last outputs of both directions
            last_output_bi = torch.cat([last_output[0], last_output[1]], dim=-1)

        num_words = batch_size * max_sentence_length
        shape = [num_words, self.num_directions * self.lstm.hidden_size]
        char_representation = torch.zeros(shape)
        char_representation = char_representation.to(last_output_bi.device)
        char_representation[sorted_inds] = last_output_bi

        return char_representation.view([batch_size, max_sentence_length, -1])


class HighwayLSTM(nn.Module):
    """
    LSTM wrapper using highway connections
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0,
                 bidirectional=True):
        super(HighwayLSTM, self).__init__()

        self.lstms = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.highways = nn.ModuleList()
        self.dropout = nn.Dropout(dropout, inplace=True)
        num_directions = 2 if bidirectional else 1
        actual_hidden_size = num_directions * hidden_size

        for i in range(num_layers):
            # create each layer as a separate LSTM object to allow finer control
            # of the data flow from each one to the next
            lstm = LSTM(input_size, hidden_size, 1, bidirectional=bidirectional)
            self.lstms.append(lstm)

            highway = nn.Linear(input_size, actual_hidden_size)
            highway.bias.data.zero_()
            self.highways.append(highway)

            gate = nn.Linear(input_size, actual_hidden_size)
            gate.bias.data.zero_()
            self.gates.append(gate)

            input_size = actual_hidden_size

    def _packed_dropout(self, states):
        """Apply dropout to packed states"""
        # shared_states is a packed tuple; (data, lengths)
        states_data = states.data
        states_lengths = states.batch_sizes
        states_data = self.dropout(states_data)
        states = nn.utils.rnn.PackedSequence(states_data, states_lengths)
        return states

    def forward(self, x):
        h_output = []
        c_output = []
        for lstm, gate, highway in zip(self.lstms, self.gates, self.highways):
            x = self._packed_dropout(x)
            packed_hidden, (ht, ct) = lstm(x)
            h_output.append(ht)
            c_output.append(ct)

            padded_x, lengths = pad_packed_sequence(x, batch_first=True)
            padded_hidden, _ = pad_packed_sequence(packed_hidden,
                                                   batch_first=True)

            g = torch.sigmoid(gate(padded_x))
            t = torch.tanh(highway(padded_x))
            hidden = padded_hidden + g * t

            x = pack_padded_sequence(hidden, lengths, batch_first=True)

        h_output = torch.cat(h_output, 0)
        c_output = torch.cat(c_output, 0)

        return x, (h_output, c_output)
