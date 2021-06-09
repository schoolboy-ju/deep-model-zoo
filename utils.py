import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sn
import torch
from torchmetrics.functional.classification import confusion_matrix
import torchvision


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrixBuffer(object):
    @property
    def confusion_matrix(self):
        return self._buffer

    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._buffer = torch.zeros(self._num_classes, self._num_classes)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self._buffer += confusion_matrix(preds=preds,
                                         target=target,
                                         num_classes=self._num_classes)

    def reset(self):
        self._buffer = torch.zeros(self._num_classes, self._num_classes)


def conf_mat_to_img_tensor(num_classes, conf_mat_tensor):
    df_cm = pd.DataFrame(conf_mat_tensor.numpy().astype(np.int),
                         index=np.arange(num_classes),
                         columns=np.arange(num_classes))

    figure = plt.figure()
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws=dict(size=16), fmt='d')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close(figure)
    buf.seek(0)
    img = Image.open(buf)
    return torchvision.transforms.ToTensor()(img)
