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


class CNNAutoencoder(BaseModule):
    def __init__(self,
                 *args,
                 **kwargs):
        super(CNNAutoencoder, self).__init__(*args, **kwargs)
        self._in_channel, self._width, self._length = self.input_dim

        self._flatten = nn.Flatten()
        self._encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=self._in_channel, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
        )
        self._encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self._decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self._decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=1,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self._encoder_1(x)
        encoded = self._encoder_2(encoded)
        decoded = self._decoder_1(encoded)
        encoded = encoded.view(batch_size, -1)
        decoded = self._decoder_2(decoded)
        return encoded, decoded


class ExampleClassifier(nn.Module):
    def __init__(self,
                 encoded_dim,
                 num_classes: int):
        super(ExampleClassifier, self).__init__()
        self._encoded_dim = encoded_dim

        self._linear = nn.Sequential(
            nn.Linear(in_features=self._encoded_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        return self._linear(x)
