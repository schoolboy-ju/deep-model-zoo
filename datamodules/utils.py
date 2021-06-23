import random

import torch
from torch.utils.data import Sampler
import torchvision


class ImbalancedDataSamplerBase(Sampler):
    def __init__(self,
                 dataset,
                 indices: list = None,
                 num_samples: int = None,
                 callback_get_label=None):
        # If indices is not provided,
        # All elements in the dataset will be considered.
        self._indices = list(range(len(dataset))) if indices is None else indices

        # Define custom callback
        self._callback_get_label = callback_get_label

        # If num_samples is not provided,
        # Draw 'len(self._indices)' samples in each iteration.
        self._num_samples = len(self._indices) if num_samples is None else num_samples

        # Distribution of classes in the dataset.
        self._label_to_count = {}
        self._indices_each_label = {}
        for idx in self._indices:
            label = self._get_label(dataset, idx)
            if label in self._label_to_count:
                self._label_to_count[label] += 1
                self._indices_each_label[label].append(idx)
            else:
                self._label_to_count[label] = 1
                self._indices_each_label[label] = [idx]

        self._labels = [self._get_label(dataset, idx) for idx in self._indices]

        # Weight for each sample.
        weights = [1.0 / self._label_to_count[self._get_label(dataset, idx)] for idx in self._indices]
        self._weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self._callback_get_label:
            return self._callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def indices(self) -> list:
        return self._indices

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def weights(self) -> list:
        return self._weights

    @property
    def indices_each_label(self) -> dict:
        return self._indices_each_label

    @property
    def label_to_count(self) -> dict:
        return self._label_to_count


class WeightedRandomSampler(ImbalancedDataSamplerBase):
    def __init__(self, *args, **kwargs):
        super(WeightedRandomSampler, self).__init__(*args, **kwargs)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class OverSampler(ImbalancedDataSamplerBase):
    def __init__(self, *args, **kwargs):
        super(OverSampler, self).__init__(*args, **kwargs)
        count_each_class = max(self.label_to_count.values())
        times_list = [count_each_class // count for count in self.label_to_count.values()]
        labels_each_class_list = list(self.indices_each_label.values())

        self._output_index_list = []
        for label_list, times in zip(labels_each_class_list, times_list):
            self._output_index_list.extend(label_list * times)

        random.shuffle(self._output_index_list)

        self._num_samples = len(self._output_index_list)

    def __iter__(self):
        return (self.indices[i] for i in self._output_index_list)

    def __len__(self):
        return self.num_samples
