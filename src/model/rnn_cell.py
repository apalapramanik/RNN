/Users/apalapramanik/Documents/LSTM/train.py /Users/apalapramanik/Documents/LSTM/src /Users/apalapramanik/Documents/LSTM/requirements.txt /Users/apalapramanik/Documents/LSTM/README.md /Users/apalapramanik/Documents/LSTM/data/Users/apalapramanik/Documents/LSTM/train.py /Users/apalapramanik/Documents/LSTM/src /Users/apalapramanik/Documents/LSTM/requirements.txt /Users/apalapramanik/Documents/LSTM/README.md /Users/apalapramanik/Documents/LSTM/dataimport torch
import torch.nn as nn


class RNNCell(nn.Module):
    """
    Single vanilla RNN cell implemented from scratch.

    Equation:
        h_t = tanh(W_x x_t + W_h h_{t-1})
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.linear_x = nn.Linear(input_dim, hidden_dim)
        self.linear_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h):
        """
        x : (B, input_dim)
        h : (B, hidden_dim)
        """
        h_next = torch.tanh(self.linear_x(x) + self.linear_h(h))
        return h_next
