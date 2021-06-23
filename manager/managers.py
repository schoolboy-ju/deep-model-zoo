import torch
from torch import nn
import torchmetrics
from tqdm import tqdm

from utils import AverageMeter


class ManagerBase(object):
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def model(self):
        return self._model.to(self._device)

    @property
    def datamodule(self):
        return self._datamodule

    @property
    def device(self):
        return self._device

    @property
    def config(self):
        return {
            'num_classes': self._num_classes,
            'feature_dim': self._feature_dim,
            'num_epochs': self._num_epochs,
        }

    def __init__(self,
                 device,
                 num_classes: int,
                 feature_dim: [tuple, int],
                 num_epochs: int,
                 datamodule,
                 model: nn.Module):
        self._device = device

        self._num_classes = num_classes
        self._feature_dim = feature_dim
        self._datamodule = datamodule
        self._num_epochs = num_epochs
        self._model = model

        self._train_dataloader = self._datamodule.train_dataloader
        self._valid_dataloader = self._datamodule.valid_dataloader

    def train_step(self, curr_epoch):
        raise NotImplementedError

    def train_epoch(self, curr_epoch):
        raise NotImplementedError

    def valid_step(self, curr_epoch):
        raise NotImplementedError

    def valid_epoch(self, curr_epoch):
        raise NotImplementedError

    def run(self):
        valid_outputs = None
        for epoch in range(self._num_epochs):
            train_outputs = self.train_epoch(curr_epoch=epoch,
                                             dataloader=self._train_dataloader)
            valid_outputs = self.valid_epoch(curr_epoch=epoch,
                                             dataloader=self._valid_dataloader)


class ClassifierManager(ManagerBase):

    def __init__(self,

                 loss_fn: nn.Module,
                 optimizer,
                 *args,
                 **kwargs):
        super(ClassifierManager, self).__init__(*args, **kwargs)

        self._loss_fn = loss_fn.to(self.device)
        self._optimizer = optimizer

        self._train_loss = AverageMeter()
        self._train_accuracy = AverageMeter()
        self._valid_loss = AverageMeter()
        self._valid_accuracy = AverageMeter()

    def config(self):
        return {
            **super(ClassifierManager, self).config,
        }

    def train_step(self, data, target):
        logits = self.model(data)
        loss = self._loss_fn(logits, target)
        return logits, loss

    def train_epoch(self, curr_epoch, dataloader):
        self.model.train()
        with tqdm(total=len(dataloader)) as p_bar:
            for time_step, (x, y) in enumerate(dataloader):
                num_data = len(x)
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.train_step(x, y)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                pred = torch.argmax(logits, dim=1)
                acc = torchmetrics.functional.accuracy(pred, y)

                self._train_loss.update(loss.detach().cpu(), n=num_data)
                self._train_accuracy.update(acc.detach().cpu(), n=num_data)

                p_bar.update(1)
                p_bar.set_description('Train | Epoch: {}'.format(curr_epoch + 1))
                p_bar.set_postfix_str('loss: {:.5f}, acc: {:.2f}%'.format(self._train_loss.avg,
                                                                          self._train_accuracy.avg * 100))
                p_bar.refresh()

        outputs = dict(loss=self._train_loss.avg, acc=self._train_accuracy.avg)

        self._train_loss.reset()
        self._train_accuracy.reset()

        return outputs

    def valid_step(self, data, target):
        logits = self.model(data)
        loss = self._loss_fn(logits, target)
        return logits, loss

    def valid_epoch(self, curr_epoch, dataloader):
        self.model.eval()
        with torch.no_grad():
            for time_step, (x, y) in enumerate(dataloader):
                num_data = len(x)
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.valid_step(x, y)
                pred = torch.argmax(logits, dim=1)
                acc = torchmetrics.functional.accuracy(pred, y)
                self._valid_loss.update(loss.detach().cpu(), n=num_data)
                self._valid_accuracy.update(acc.detach().cpu(), n=num_data)

        outputs = dict(loss=self._valid_loss.avg, acc=self._valid_accuracy.avg)
        print("Valid | Epoch: {} | loss: {:.5f}, acc: {:.2f}%".format(curr_epoch + 1,
                                                                      self._valid_loss.avg,
                                                                      self._valid_accuracy.avg * 100))

        self._valid_loss.reset()
        self._valid_accuracy.reset()

        return outputs


class AutoencoderManager(ManagerBase):

    def __init__(self,
                 loss_fn: nn.Module,
                 optimizer,
                 classifier: nn.Module = None,
                 classifier_optim=None,
                 classifier_loss_fn=None,
                 classifier_train_epoch: int = 20,
                 *args,
                 **kwargs):
        super(AutoencoderManager, self).__init__(*args, **kwargs)

        self._loss_fn = loss_fn.to(self.device)
        self._optimizer = optimizer

        self._train_loss = AverageMeter()
        self._valid_loss = AverageMeter()

        if classifier is not None:
            self._classifier = classifier.to(self.device)
            self._classifier_loss_fn = classifier_loss_fn.to(self.device)
            self._classifier_optim = classifier_optim
            self._num_classifier_train_epoch = classifier_train_epoch

            self._classifier_train_loss = AverageMeter()
            self._classifier_train_acc = AverageMeter()
            self._classifier_valid_loss = AverageMeter()
            self._classifier_valid_acc = AverageMeter()

    def config(self):
        return {
            **super(AutoencoderManager, self).config,
        }

    def train_step(self, data):
        _, decoded = self.model(data)
        loss = self._loss_fn(decoded, data)
        return loss

    def train_epoch(self, curr_epoch, dataloader):
        self.model.train()
        with tqdm(total=len(dataloader)) as p_bar:
            for time_step, (x, _) in enumerate(dataloader):
                num_data = len(x)
                x = x.to(self.device)
                loss = self.train_step(x)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                self._train_loss.update(loss.detach().cpu(), n=num_data)

                p_bar.update(1)
                p_bar.set_description('Train | Epoch: {}'.format(curr_epoch + 1))
                p_bar.set_postfix_str('loss: {:.5f}'.format(self._train_loss.avg))
                p_bar.refresh()

        outputs = dict(loss=self._train_loss.avg)

        self._train_loss.reset()

        return outputs

    def valid_step(self, data):
        _, decoded = self.model(data)
        loss = self._loss_fn(decoded, data)
        return loss

    def valid_epoch(self, curr_epoch, dataloader):
        self.model.eval()
        with torch.no_grad():
            for time_step, (x, _) in enumerate(dataloader):
                num_data = len(x)
                x = x.to(self.device)
                loss = self.valid_step(x)
                self._valid_loss.update(loss.detach().cpu(), n=num_data)

        outputs = dict(loss=self._valid_loss.avg)
        print("Valid | Epoch: {}: loss: {:.5f}".format(curr_epoch + 1, self._valid_loss.avg))
        self._valid_loss.reset()

        return outputs

    def run_classifier_train(self):
        assert self._classifier is not None, "Set classifier first."

        for epoch in range(self._num_classifier_train_epoch):
            train_outputs = self._classifier_train_epoch(curr_epoch=epoch,
                                                         dataloader=self._train_dataloader)
            valid_outputs = self._classifier_valid_epoch(curr_epoch=epoch,
                                                         dataloader=self._valid_dataloader)

    def _classifier_train_step(self, encoded_data, target):
        logits = self._classifier(encoded_data)
        loss = self._classifier_loss_fn(logits, target)
        return logits, loss

    def _classifier_train_epoch(self, curr_epoch, dataloader):
        self._classifier.train()
        self.model.eval()

        with tqdm(total=len(dataloader)) as p_bar:
            for time_step, (x, y) in enumerate(dataloader):
                num_data = len(x)
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    x, _ = self.model(x)
                logits, loss = self._classifier_train_step(x, y)

                self._classifier_optim.zero_grad()
                loss.backward()
                self._classifier_optim.step()

                pred = torch.argmax(logits, dim=1)
                acc = torchmetrics.functional.accuracy(pred, y)

                self._classifier_train_loss.update(loss.detach().cpu(), n=num_data)
                self._classifier_train_acc.update(acc.detach().cpu(), n=num_data)

                p_bar.update(1)
                p_bar.set_description('Train | Epoch: {}'.format(curr_epoch + 1))
                p_bar.set_postfix_str('loss: {:.5f}, acc: {:.2f}%'.format(self._classifier_train_loss.avg,
                                                                          self._classifier_train_acc.avg * 100))
                p_bar.refresh()

        outputs = dict(loss=self._classifier_train_loss.avg, acc=self._classifier_train_acc.avg)

        self._classifier_train_loss.reset()
        self._classifier_train_acc.reset()

        return outputs

    def _classifier_valid_step(self, data, target):
        logits = self._classifier(data)
        loss = self._classifier_loss_fn(logits, target)
        return logits, loss

    def _classifier_valid_epoch(self, curr_epoch, dataloader):
        self._classifier.eval()
        self.model.eval()

        with torch.no_grad():
            for time_step, (x, y) in enumerate(dataloader):
                num_data = len(x)
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    x, _ = self.model(x)
                logits, loss = self._classifier_valid_step(x, y)
                pred = torch.argmax(logits, dim=1)
                acc = torchmetrics.functional.accuracy(pred, y)
                self._classifier_valid_loss.update(loss.detach().cpu(), n=num_data)
                self._classifier_valid_acc.update(acc.detach().cpu(), n=num_data)

        outputs = dict(loss=self._classifier_valid_loss.avg, acc=self._classifier_valid_acc.avg)
        print("Valid | Epoch: {} | loss: {:.5f}, acc: {:.2f}%".format(curr_epoch + 1,
                                                                      self._classifier_valid_loss.avg,
                                                                      self._classifier_valid_acc.avg * 100))

        self._classifier_valid_loss.reset()
        self._classifier_valid_acc.reset()

        return outputs
