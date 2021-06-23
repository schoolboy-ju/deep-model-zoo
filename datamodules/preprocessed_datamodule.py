import os
import pickle

import pandas as pd
from sklearn import preprocessing
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from datamodules import utils

PREPROCESSED_ROOT_PATH = os.path.join('datasets', 'preprocessed')


class PreprocessedDataset(Dataset):

    @property
    def dims(self):
        return self._dims

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def label_list(self):
        return self._encoded_label_int

    @staticmethod
    def _read_data(src_path):
        with open(src_path, 'rb') as f:
            ret = pickle.load(f)
        return ret

    @staticmethod
    def get_label(dataset, idx):
        return dataset.label_list[idx]

    def __init__(self,
                 dataset_name: str,
                 is_train: bool = True):
        super(PreprocessedDataset, self).__init__()
        dataset_root_path = os.path.join(PREPROCESSED_ROOT_PATH, dataset_name)

        if is_train:
            metadata_path = os.path.join(dataset_root_path, 'train_split.csv')
            self._metadata = pd.read_csv(metadata_path, '\t')
            self._data_root_path = os.path.join(dataset_root_path, 'preprocessed', 'train')
        else:
            metadata_path = os.path.join(dataset_root_path, 'test_split.csv')
            self._metadata = pd.read_csv(metadata_path, '\t')
            self._data_root_path = os.path.join(dataset_root_path, 'preprocessed', 'test')

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self._metadata['label'])
        self._encoded_label_int = label_encoder.transform(self._metadata['label'])
        self._num_classes = len(self._metadata['label'].unique())

        # Get a sample data to get data dimension
        sample_data = self._read_data(os.path.join(self._data_root_path, self._metadata['file_name'][0]))
        self._dims = sample_data.shape

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        src_path = os.path.join(self._data_root_path, self._metadata['file_name'][index])
        data = self._read_data(src_path=src_path)
        label = self._encoded_label_int[index]
        return data, label


class PreprocessedDatamodule(object):
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def dims(self):
        return self._dims

    def __init__(self,
                 train_batch_size: int,
                 valid_batch_size: int,
                 num_workers: int,
                 dataset_name: str,
                 num_valid: int = 500,
                 use_over_sampler: bool = True):

        self._train_batch_size = train_batch_size
        self._valid_batch_size = valid_batch_size
        self._num_workers = num_workers

        # TODO(joohyun): test dataset
        self._train_dataset = PreprocessedDataset(dataset_name=dataset_name,
                                                  is_train=True)
        self._valid_dataset = PreprocessedDataset(dataset_name=dataset_name,
                                                  is_train=False)

        self._train_sampler = None
        self._valid_sampler = None
        if use_over_sampler:
            self._train_sampler = utils.OverSampler(self._train_dataset,
                                                    callback_get_label=self._train_dataset.get_label)
            self._valid_sampler = utils.WeightedRandomSampler(self._valid_dataset,
                                                              num_samples=num_valid,
                                                              callback_get_label=self._valid_dataset.get_label)

        self._dims = self._train_dataset.dims
        self._num_classes = self._train_dataset.num_classes

    @property
    def train_dataloader(self):
        if self._train_sampler is not None:
            return DataLoader(self._train_dataset,
                              batch_size=self._train_batch_size,
                              num_workers=self._num_workers,
                              pin_memory=True,
                              sampler=self._train_sampler)

        return DataLoader(self._train_dataset,
                          batch_size=self._train_batch_size,
                          num_workers=self._num_workers,
                          pin_memory=True,
                          shuffle=True)

    @property
    def valid_dataloader(self):
        if self._valid_sampler is not None:
            return DataLoader(self._valid_dataset,
                              batch_size=self._valid_batch_size,
                              num_workers=self._num_workers,
                              pin_memory=True,
                              sampler=self._valid_sampler)

        return DataLoader(self._valid_dataset,
                          batch_size=self._valid_batch_size,
                          num_workers=self._num_workers,
                          pin_memory=True,
                          shuffle=True)
