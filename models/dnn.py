from torch import nn

from models.base import BaseModule


class NeuralNetwork(BaseModule):

    def __init__(self,
                 *args,
                 **kwargs):
        super(NeuralNetwork, self).__init__(*args, **kwargs)

        _, width, length = self.input_dim

        self._flatten = nn.Flatten()
        self._linear_relu = nn.Sequential(
            nn.Linear(width * length, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self._flatten(x)
        logits = self._linear_relu(x)
        return logits
