import torch
import torch.nn as nn

from models.base import BaseModule


class RecurrentNet(BaseModule):
    def __init__(self,
                 hidden_dim,
                 num_layers,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        channels, width, length = self.input_dim
        self._seq_dim = length

        self._input_dim = width
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._rnn = nn.RNN(input_size=self._input_dim,
                           hidden_size=self._hidden_dim,
                           num_layers=self._num_layers,
                           batch_first=True,
                           nonlinearity='tanh')

        self._classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._hidden_dim, self.num_classes)
        )

    def forward(self, x):
        device = x.device
        x = x.view(-1, self._seq_dim, self._input_dim)

        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self._num_layers, x.size(0), self._hidden_dim).requires_grad_().to(device)

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        x, hn = self._rnn(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> hidden_dim, 28, 10
        # out[:, -1, :] --> hidden_dim, 10 --> just want last time step hidden states
        x = self._classifier(x[:, -1, :])
        return x
