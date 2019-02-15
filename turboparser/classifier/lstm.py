import torch
from torch import nn


class LSTM(nn.Module):
    """
    A wrapper of the torch LSTM with a few built-in functionalities.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=True, batch_first=True,
                            num_layers=num_layers, dropout=dropout)

        # shape is (num_directions, batch, num_units)
        shape = (2, 1, hidden_size)
        self.initial_h = nn.Parameter(torch.zeros(shape))
        self.initial_c = nn.Parameter(torch.zeros(shape))

    @property
    def hidden_size(self):
        return self.lstm.hidden_size

    @property
    def input_size(self):
        return self.lstm.input_size

    def forward(self, x):
        """
        :param x: a packed padded sequence
        :return:
        """
        # the packed padded sequence is (actual data, lengths)
        # lengths is sorted descending
        batch_size = x[1][0].item()

        # initial states must be (num_directions * layers, batch, num_units)
        batch_initial_h = self.initial_h.expand(-1, batch_size, -1)
        batch_initial_c = self.initial_c.expand(-1, batch_size, -1)
        return self.lstm(x, (batch_initial_h, batch_initial_c))
