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

        self._train_dataset = None
        self._valid_dataset = None

        self._train_dataset = datasets.FashionMNIST(
            root=self._data_path,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        self._valid_dataset = datasets.FashionMNIST(
            root=self._data_path,
            train=False,
            download=True,
            transform=ToTensor(),
        )

    @property
    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size)

    @property
    def valid_dataloader(self):
        return DataLoader(self._valid_dataset, batch_size=self._batch_size)
