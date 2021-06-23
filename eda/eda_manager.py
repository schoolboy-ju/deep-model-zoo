import os.path
import pickle
import statistics
import yaml

import librosa
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis

from eda import *


class EDAManager(object):
    @property
    def dataset_root_path(self) -> str:
        return self._dataset_root_path

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def class_list(self) -> list:
        return self._class_list

    @staticmethod
    def _get_path_dict(root_path, data_classes_list) -> dict:
        dataset_path_dict = {}
        for c in data_classes_list:
            path_list = os.listdir(os.path.join(root_path, c))
            dataset_path_dict[c] = [os.path.join(root_path, c, f) for f in path_list]
        return dataset_path_dict

    @staticmethod
    def _read_metadata(preprocessed_path):
        file_list = os.listdir(preprocessed_path)
        csv_list = list(filter(lambda file_name: file_name.endswith('.csv'), file_list))
        train_meta_path = list(filter(lambda file_name: file_name.startswith('train'), csv_list))[0]
        test_meta_path = list(filter(lambda file_name: file_name.startswith('test'), csv_list))[0]
        train_metadata = pd.read_csv(os.path.join(preprocessed_path, train_meta_path), delimiter='\t')
        test_metadata = pd.read_csv(os.path.join(preprocessed_path, test_meta_path), delimiter='\t')
        return train_metadata, test_metadata

    def __init__(self,
                 preprocessed_path: str):
        self._preprocessed_path = preprocessed_path
        self._dataset_name = preprocessed_path.split('/')[-1]
        self._train_metadata, self._test_metadata = self._read_metadata(preprocessed_path=preprocessed_path)
        self._label_list = pd.unique(self._train_metadata['label'])
        with open(os.path.join(preprocessed_path, 'config.yml')) as f:
            self._preprocess_config = yaml.load(f, yaml.FullLoader)

    def get_raw_sample_from_each_label(self) -> dict:
        print("Classes list: {}".format(self._label_list))
        ret_list = []
        for label, df in self._train_metadata.groupby('label'):
            tmp = {}
            path = df.sample(n=1)['src_path'].item()
            if self._preprocess_config['data_type'] == 'audio':
                signal, sr = librosa.load(path, sr=None)
            elif self._preprocess_config['data_type'] == 'vibe':
                with open(path, 'rb') as f:
                    signal = pickle.load(f)
                sr = self._preprocess_config['sample_rate']
            tmp['signal'] = signal
            tmp['sr'] = sr
            tmp['label'] = label
            ret_list.append(tmp)
        return ret_list

    def get_preprocessed_sample_from_each_label(self) -> dict:
        print("Classes list: {}".format(self._label_list))
        ret_list = []
        for label, df in self._train_metadata.groupby('label'):
            tmp = {}
            filename = df.sample(n=1)['file_name'].item()
            prep_data_path = os.path.join(self._preprocessed_path, 'preprocessed', 'train', str(filename))
            with (open(prep_data_path, 'rb')) as openfile:
                tmp['data'] = pickle.load(openfile)
            tmp['label'] = label
            ret_list.append(tmp)
        return ret_list

    def plot_label_bar_graph(self):
        label_series = self._train_metadata.groupby('label').size()
        label_series.sort_index(inplace=True)
        sns.barplot(x=label_series.index, y=label_series)
        plt.ylabel('counts')

    def plot_duration_box_plot(self, fontsize: int = 10, figsize: tuple = (4, 5)):
        grouped = self._train_metadata.groupby('label')
        grouped.boxplot(column='duration', rot=45, fontsize=fontsize, figsize=figsize)

    @staticmethod
    def signal_framing(sig, fs=32000, win_len=0.064, win_hop=0.032):
        """
        transform a signal into a series of overlapping frames.

        Args:
            sig            (array) : a mono audio signal (Nx1) from which to compute features.
            fs               (int) : the sampling frequency of the signal we are working with.
                                     Default is 32000.
            win_len        (float) : window length in sec.
                                     Default is 0.064.
            win_hop        (float) : step between successive windows in sec.
                                     Default is 0.032.

        Returns:
            array of frames.
            frame length.
        """
        # compute frame length and frame step (convert from seconds to samples)
        frame_length = win_len * fs
        frame_step = win_hop * fs
        signal_length = len(sig)
        frames_overlap = frame_length - frame_step

        # Make sure that we have at least 1 frame+
        num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)

        # Pad Signal to make sure that all frames have equal number of samples
        # without truncating any samples from the original signal
        if rest_samples != 0:
            pad_signal_length = int(frame_step - rest_samples)
            z = np.zeros(pad_signal_length)
            pad_signal = np.append(sig, z)
            num_frames += 1
        else:
            pad_signal = sig

        # make sure to use integers as indices
        frame_length = int(frame_length)
        frame_step = int(frame_step)
        num_frames = int(num_frames)

        # compute indices
        idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
        idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step),
                       (frame_length, 1)).T
        indices = idx1 + idx2
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        return frames

    @staticmethod
    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def feat_rms(signal):

        if len(signal.shape) == 1:
            return np.sqrt(np.mean(signal ** 2))
        ret = []
        for s in signal:
            ret.append(np.sqrt(np.mean(s ** 2)))
        return np.array(ret)

    @staticmethod
    def feat_max(signal):
        if len(signal.shape) == 1:
            return np.max(signal)
        ret = []
        for s in signal:
            ret.append(np.max(s))
        return np.array(ret)

    @staticmethod
    def feat_variance(signal):
        if len(signal.shape) == 1:
            return statistics.variance(signal)
        ret = []
        for s in signal:
            ret.append(statistics.variance(s))
        return np.array(ret)

    @staticmethod
    def feat_peak_to_peak(signal):
        if len(signal.shape) == 1:
            return np.max(signal) - np.min(signal)
        ret = []
        for s in signal:
            ret.append(np.max(s) - np.min(s))
        return np.array(ret)

    @staticmethod
    def feat_skewness(data):
        if len(data.shape) == 1:
            return skew(data)
        ret = []
        for d in data:
            ret.append(skew(d))
        return np.array(ret)

    @staticmethod
    def feat_kurtosis(feature):
        if len(feature.shape) == 1:
            return kurtosis(feature)
        ret = []
        for s in feature:
            ret.append(kurtosis(s))
        return np.array(ret)
