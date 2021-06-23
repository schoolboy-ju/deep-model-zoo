from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class ExampleDatamodule(object):
    @property
    def dims(self):
        return 1, 28, 28

    @property
    def num_classes(self):
        return 10

    def __init__(self,
                 data_path: str = 'datasets',
                 batch_size: int = 32):
        self._data_path = data_path
        self._batch_size = batch_size

        self._train_data = None
        self._eval_data = None

    def setup(self):
        self._train_data = datasets.FashionMNIST(
            root=self._data_path,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        self._eval_data = datasets.FashionMNIST(
            root=self._data_path,
            train=False,
            download=True,
            transform=ToTensor(),
        )

    @property
    def train_dataloader(self):
        return DataLoader(self._train_data, batch_size=self._batch_size)

    @property
    def eval_dataloader(self):
        return DataLoader(self._eval_data, batch_size=self._batch_size)
