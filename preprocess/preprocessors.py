import csv
import os
import pickle
import random
import time
import shutil

import librosa
import yaml

from tqdm import tqdm
import pandas as pd

from preprocess import PROJ_PATH
from preprocess import utils


class PreprocessorBase(object):

    @property
    def dataset_root_path(self) -> str:
        return self._dataset_root_path

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def class_list(self) -> list:
        return self._class_list

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_data_each_chunk(self) -> int:
        return self._num_data_each_chunk

    @property
    def src_path_dict(self) -> dict:
        return self._src_path_dict

    @property
    def dest_path(self):
        return self._dest_path

    @property
    def train_size(self) -> float:
        return self._train_split_ratio / (self._train_split_ratio + self._test_split_ratio)

    @property
    def test_size(self) -> float:
        return self._test_split_ratio / (self._train_split_ratio + self._test_split_ratio)

    @property
    def num_data(self) -> int:
        return self._num_data

    @staticmethod
    def _make_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def _get_now():
        return time.time()

    @staticmethod
    def chunks(data_batch, n: int = 1000):
        for chunk_start_idx in range(0, len(data_batch), n):
            try:
                yield data_batch[chunk_start_idx: chunk_start_idx + n]
            except IndexError:
                yield data_batch[chunk_start_idx:]

    def _setup_dataset_path(self, dataset_name):
        prep_root_path = os.path.join(PROJ_PATH,
                                      'datasets',
                                      'preprocessed',
                                      '{}_{}'.format(dataset_name, int(self._get_now())))
        self._make_path(prep_root_path)
        return prep_root_path

    def __init__(self,
                 dataset_root_path: str,
                 train_split_ratio: int = 7,
                 test_split_ratio: int = 3):
        self._dataset_root_path = dataset_root_path
        self._dataset_name = dataset_root_path.split('/')[-1]
        self._class_list = os.listdir(self._dataset_root_path)
        self._dest_path = self._setup_dataset_path(self._dataset_name)

        self._train_split_ratio = train_split_ratio
        self._test_split_ratio = test_split_ratio

        # Data file lists
        self._src_path_dict = utils.get_path_dict(self._dataset_root_path, self._class_list)

        # Get num datasets
        self._num_data = 0
        for src_list in self._src_path_dict.values():
            self._num_data += len(src_list)

        self._num_classes = len(self._class_list)

    @property
    def config(self):
        return {
            'dataset_root_path': self._dataset_root_path,
            'dataset_name': self._dataset_name,
            'class_list': self._class_list,
            'dest_path': self._dest_path,
            'train_split_ratio': self._train_split_ratio,
            'test_split_ratio': self._test_split_ratio,
            'num_data': self._num_data,
            'num_classes': self._num_classes,
        }

    def _build_metadata(self, meta_file_name: str):
        metadata_path = os.path.join(self.dest_path, meta_file_name + '.csv')
        f_meta = open(metadata_path, 'w', encoding='utf-8', newline='')
        f_meta_csv_writer = csv.writer(f_meta, delimiter='\t')
        field_names = ['file_name', 'label', 'duration', 'src_path']
        f_meta_csv_writer.writerow(field_names)
        return f_meta, f_meta_csv_writer, metadata_path

    def _write_row_on_metadata(self, metadata_writer, src_path, label):
        signal = self.read_data(src_path, self._sample_rate)
        duration = signal.shape[-1] / self._sample_rate
        file_name = src_path.split('/')[-1]
        if file_name.endswith('.wav') or file_name.endswith('.asc'):
            file_name = file_name[:-4]
        metadata_writer.writerow([file_name, label, duration, src_path])

    def _random_split(self):
        f_train, train_metadata_writer, train_metadata_path = self._build_metadata('train_split')
        f_test, test_metadata_writer, test_metadata_path = self._build_metadata('test_split')
        total_train_size = 0
        for label, src_path_list in self.src_path_dict.items():
            random.shuffle(x=src_path_list)
            train_size = int(len(src_path_list) * self.train_size)
            total_train_size += train_size
            for src_path in src_path_list[:train_size]:
                self._write_row_on_metadata(metadata_writer=train_metadata_writer,
                                            src_path=src_path,
                                            label=label)
            for src_path in src_path_list[train_size:]:
                self._write_row_on_metadata(metadata_writer=test_metadata_writer,
                                            src_path=src_path,
                                            label=label)
        f_train.close()
        f_test.close()
        return train_metadata_path, test_metadata_path, total_train_size, self.num_data - total_train_size

    def _copy_data_of_path_list_to_split_path(self, train_meta_path, test_meta_path):
        save_path = os.path.join(self.dest_path, 'raw', 'train')
        os.makedirs(save_path)
        train_meta = pd.read_csv(train_meta_path, '\t')
        for path in train_meta['file_name']:
            shutil.copy(path, os.path.join(self.dest_path, save_path))

        save_path = os.path.join(self.dest_path, 'raw', 'test')
        os.makedirs(save_path)
        test_meta = pd.read_csv(test_meta_path, '\t')
        for path in test_meta['file_name']:
            shutil.copy(path, save_path)

    def _save_preprocessed(self, path_to_save, dataframe):
        signal = self.read_data(dataframe['src_path'].item(), self._sample_rate)
        feature = self.preprocess(signal)
        with open(os.path.join(path_to_save, str(dataframe['file_name'].item())), 'wb') as f:
            pickle.dump(feature, f)

    def run_save(self, save_raw_split: bool = False):
        train_meta_path, test_meta_path, train_size, test_size = self._random_split()

        if save_raw_split:
            self._copy_data_of_path_list_to_split_path(train_meta_path, test_meta_path)

        preprocessed_train_path = os.path.join(self.dest_path, 'preprocessed', 'train')
        os.makedirs(preprocessed_train_path)
        for df in tqdm(pd.read_csv(train_meta_path, sep='\t', chunksize=1), desc=self.dataset_name, total=train_size):
            self._save_preprocessed(path_to_save=preprocessed_train_path,
                                    dataframe=df)

        preprocessed_test_path = os.path.join(self.dest_path, 'preprocessed', 'test')
        os.makedirs(preprocessed_test_path)
        for df in tqdm(pd.read_csv(test_meta_path, sep='\t', chunksize=1), desc=self.dataset_name, total=test_size):
            self._save_preprocessed(path_to_save=preprocessed_test_path,
                                    dataframe=df)

    def preprocess(self, src):
        raise NotImplementedError

    def read_data(self, src_path, sample_rate):
        raise NotImplementedError


class AudioPreprocessor(PreprocessorBase):
    @property
    def n_hops(self):
        return utils.ms_to_closest_power_2_sample_len(sample_rate=self._sample_rate, x=self._hop_length)

    @property
    def n_window(self):
        return utils.ms_to_closest_power_2_sample_len(sample_rate=self._sample_rate, x=self._window_length)

    def __init__(self,
                 sample_rate: int,
                 n_mels: int,
                 window_length: [float, int],
                 hop_length: [float, int],
                 *args,
                 **kwargs):
        super(AudioPreprocessor, self).__init__(*args, **kwargs)
        self._sample_rate = sample_rate

        self._n_mels = n_mels

        self._window_length = window_length
        self._hop_length = hop_length

    @property
    def config(self):
        return {
            'sample_rate': self._sample_rate,
            'n_mels': self._n_mels,
            'window_length': self._window_length,
            'hop_length': self._hop_length,
            'n_window': self.n_window,
            'n_hops': self.n_hops,
            **super(AudioPreprocessor, self).config
        }

    def read_data(self, src_path, sample_rate):
        return utils.read_audio(src_path, sample_rate)

    def preprocess(self, signal):
        return utils.get_mel_spectrogram(signal,
                                         sample_rate=self._sample_rate,
                                         n_mels=self._n_mels,
                                         window_len=self._window_length,
                                         hop_len=self._hop_length)


class VibePreprocessor(PreprocessorBase):

    @property
    def lower_bound_hz(self):
        self._lower_bound_hz

    @property
    def upper_bound_hz(self):
        self._upper_bound_hz

    @property
    def n_hops(self):
        return utils.ms_to_closest_power_2_sample_len(sample_rate=self._sample_rate, x=self._hop_length)

    @property
    def n_window(self):
        return utils.ms_to_closest_power_2_sample_len(sample_rate=self._sample_rate, x=self._window_length)

    def __init__(self,
                 sample_rate: int,
                 window_length: [float, int],
                 hop_length: [float, int],
                 lower_bound_hz: int = 1500,
                 upper_bound_hz: int = 4000,
                 *args,
                 **kwargs):
        super(VibePreprocessor, self).__init__(*args, **kwargs)
        self._sample_rate = sample_rate

        self._lower_bound_hz = lower_bound_hz
        self._upper_bound_hz = upper_bound_hz

        self._window_length = window_length
        self._hop_length = hop_length

    @property
    def config(self):
        return {
            'sample_rate': self._sample_rate,
            'lower_bound': self._lower_bound_hz,
            'upper_bound': self._upper_bound_hz,
            'window_length': self._window_length,
            'hop_length': self._hop_length,
            'n_window': self.n_window,
            'n_hops': self.n_hops,
            **super(VibePreprocessor, self).config
        }

    def preprocess(self, signal):
        return utils.get_log_spectrogram(signal=signal,
                                         sample_rate=self._sample_rate,
                                         window_len=self.n_window,
                                         hop_len=self.n_hops,
                                         lower_bound_hz=self._lower_bound_hz,
                                         upper_bound_hz=self._upper_bound_hz)

    def read_data(self, src_path, sample_rate):
        with open(src_path, 'rb') as f:
            data = pickle.load(f)
        return data


class PreprocessorManager(object):
    @property
    def preprocessor(self):
        return self._preprocessor

    def __init__(self,
                 data_type: str,
                 *args,
                 **kwargs):
        assert data_type in ['audio', 'vibe']
        self._data_type = data_type

        if data_type == 'audio':
            self._preprocessor = AudioPreprocessor(*args, **kwargs)
        elif data_type == 'vibe':
            self._preprocessor = VibePreprocessor(*args, **kwargs)
        else:
            raise ValueError("Not yet.")

    @property
    def config(self):
        return {
            'data_type': self._data_type,
            **self._preprocessor.config
        }

    def preprocess(self, signal):
        self._preprocessor.preprocess(signal=signal)

    def read_data(self, src_path, sample_rate):
        self._preprocessor.read_data(src_path=src_path, sample_rate=sample_rate)

    def run_save(self):
        self._preprocessor.run_save()

    @property
    def dest_path(self):
        return self._preprocessor.dest_path

    def save_config(self):
        config_save_path = os.path.join(self.dest_path, 'config.yml')
        with open(config_save_path, 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)
