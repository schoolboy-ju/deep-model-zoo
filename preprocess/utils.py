import math
import os
import pickle
import time

import librosa
from tqdm import tqdm
import numpy as np

from preprocess import PROJ_PATH


def ms_to_closest_power_2_sample_len(sample_rate, x):
    if type(x) == float:
        n_samples = int(sample_rate * x)
    elif type(x) == int:
        n_samples = x
    else:
        raise TypeError

    """
        Return the closest power of 2 by checking whether
        the second binary number is a 1.
    """
    op = math.floor if bin(n_samples)[3] != "1" else math.ceil
    return 2 ** (op(math.log(n_samples, 2)))


def get_frequency_map(n_fft, sample_rate):
    t = n_fft * 1.0 / sample_rate
    if sample_rate % 2 == 0:
        return np.arange(n_fft // 2 + 1) / t
    else:
        return np.arange((n_fft - 1) // 2 + 1) / t


def read_audio(src_path, sample_rate):
    signal, _ = librosa.load(src_path, sr=sample_rate)
    return signal


def get_path_dict(root_path, data_classes_list) -> dict:
    dataset_path_dict = {}
    for c in data_classes_list:
        path_list = os.listdir(os.path.join(root_path, c))
        dataset_path_dict[c] = [os.path.join(root_path, c, f) for f in path_list]
    return dataset_path_dict


def get_mel_spectrogram(signal,
                        sample_rate: int,
                        n_mels: int,
                        window_len: [float, int],
                        hop_len: [float, int]):
    num_channels = 1
    if len(signal.shape) == 2:
        num_channels = len(signal)
    n_fft = ms_to_closest_power_2_sample_len(sample_rate, window_len)
    n_hop = ms_to_closest_power_2_sample_len(sample_rate, hop_len)

    ret = []
    for ch in range(num_channels):
        spec_gram = librosa.feature.melspectrogram(y=signal,
                                                   n_mels=n_mels,
                                                   n_fft=n_fft,
                                                   hop_length=n_hop)
        ret.append(librosa.power_to_db(spec_gram, ref=np.max))
    ret = np.array(ret)
    return ret


def get_log_spectrogram(signal,
                        sample_rate: int,
                        window_len: [float, int],
                        hop_len: [float, int],
                        lower_bound_hz: int = 0,
                        upper_bound_hz: int = None):
    num_channels = 1
    if len(signal.shape) == 2:
        num_channels = len(signal)
    n_fft = ms_to_closest_power_2_sample_len(sample_rate, window_len)
    n_hop = ms_to_closest_power_2_sample_len(sample_rate, hop_len)

    ret = []
    for ch in range(num_channels):
        spec_gram = np.abs(librosa.stft(signal[ch],
                                        n_fft=n_fft,
                                        hop_length=n_hop,
                                        window='hann'))
        log_spec_gram = librosa.amplitude_to_db(spec_gram, ref=np.max)
        if upper_bound_hz is not None:
            freq_map = get_frequency_map(n_fft, sample_rate)
            log_spec_gram = log_spec_gram[(freq_map > lower_bound_hz) & (freq_map <= upper_bound_hz), :]
        ret.append(log_spec_gram)
    ret = np.array(ret)
    return ret


""" Utils for .asc file """

DATA_PREFIX_WITH_HEADER = ('X02VG4KNKH',
                           'Re_X02VG4KNKH',
                           'X02VG4KNLH',
                           'X10VG4KNLH',
                           'X18VG4KNLH')
NUM_HEAD_LINES = 7


def read_data_lines(lines,
                    sample_rate: int,
                    diagnosing_part: str,
                    with_header: bool = False) -> list:
    assert diagnosing_part in ['snap_ring', 'bsa']
    if not with_header:
        if diagnosing_part == 'snap_ring':
            time_duration = 5.0
            num_samples = round(time_duration * sample_rate)
            return np.array(([line.split('\t')[1:] for line in lines[-num_samples:]]), dtype=np.float32).T

        elif diagnosing_part == 'bsa':
            time_duration = 2.0
            num_samples = round(time_duration * sample_rate)
            return np.array(([line.split('\t')[1:] for line in lines[:num_samples]]), dtype=np.float32).T

        else:
            raise ValueError

    head = lines[:NUM_HEAD_LINES]
    data = lines[NUM_HEAD_LINES:]
    timestamps = [line.split('\t')[0] for line in data]

    if diagnosing_part == 'snap_ring':
        trigger_time = head[-1].split(',')[-2]
        time_duration = 5.0
    elif diagnosing_part == 'bsa':
        trigger_time = head[-1].split(',')[1]
        time_duration = 2.0
    else:
        raise ValueError

    start_idx = [i for i, ts in enumerate(timestamps) if trigger_time in ts][0]
    end_idx = int(start_idx + round(time_duration * sample_rate))
    return np.array(([line.split('\t')[1:] for line in data[start_idx:end_idx]]), dtype=np.float32).T


def read_asc(src_path, sample_rate, diagnosing_part):
    file_name = src_path.split('/')[-1]
    f = open(src_path, 'r', encoding='cp949')
    lines = f.readlines()
    if file_name.startswith(DATA_PREFIX_WITH_HEADER):
        data_lines = read_data_lines(lines,
                                     sample_rate=sample_rate,
                                     diagnosing_part=diagnosing_part,
                                     with_header=True)
    else:
        data_lines = read_data_lines(lines,
                                     sample_rate=sample_rate,
                                     diagnosing_part=diagnosing_part,
                                     with_header=False)
    f.close()
    return data_lines


def crop_vibe_data(
        sample_rate: int,
        diagnosing_part: str,
        src_path: str,
        dest_path: str = None):
    # TODO(joohyun): for windows os
    dataset_name = src_path.split('/')[-1]

    class_list = os.listdir(src_path)
    if dest_path is None:
        dest_path_root = os.path.join(PROJ_PATH,
                                      'datasets',
                                      '{}_{}'.format(dataset_name, int(time.time())))

    if not os.path.exists(dest_path_root):
        os.makedirs(dest_path_root)

    path_lists = [os.path.join(dest_path_root, label) for label in class_list]
    for path in path_lists:
        if not os.path.exists(path):
            os.makedirs(path)

    src_path_dict = get_path_dict(src_path, class_list)
    for label, src_path_list in src_path_dict.items():
        path_to_save = os.path.join(dest_path_root, label)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        for src_path in tqdm(src_path_list, desc=label):
            data = read_asc(src_path,
                            sample_rate=sample_rate,
                            diagnosing_part=diagnosing_part)
            file_name = src_path.split('/')[-1]
            if file_name.endswith('.asc'):
                file_name = file_name[:-4]
            with open(os.path.join(path_to_save, file_name), 'wb') as f:
                pickle.dump(data, f)
