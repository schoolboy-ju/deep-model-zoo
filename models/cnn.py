import torch.nn as nn

from models.base import BaseModule


class ConvNet(BaseModule):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        in_channel, _, _ = self.input_dim
        self._conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 30 * 8, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, x):
        x = self._conv(x)
        x = self._classifier(x)
        return x
