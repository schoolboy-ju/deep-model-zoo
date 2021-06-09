from torch import nn
from models.base import BaseModule


class Autoencoder(BaseModule):
    def __init__(self,
                 *args,
                 **kwargs):
        super(Autoencoder, self).__init__(*args, **kwargs)
        self._in_channel, self._width, self._length = self.input_dim

        self._flatten = nn.Flatten()
        self._encoder = nn.Sequential(
            nn.Linear(self._width * self._length, 128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=4),
        )
        self._decoder = nn.Sequential(
            nn.Linear(in_features=4, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self._width * self._length),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self._flatten(x)
        encoded = self._encoder(x)
        decoded = self._decoder(encoded)
        decoded = decoded.view(-1, self._in_channel, self._width, self._length)
        return encoded, decoded
