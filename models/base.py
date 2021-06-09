from torch import nn


class BaseModule(nn.Module):
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_dim(self):
        return self._input_dim

    def __init__(self,
                 input_dim,
                 num_classes):
        super(BaseModule, self).__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes
