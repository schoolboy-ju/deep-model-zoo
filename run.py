import torch
from torch import nn

from example_dataloader import ExampleDatamodule
from models.dnn import NeuralNetwork
from models.cnn import ConvNet
from models.rnn import RecurrentNet
from models.autoencoder import Autoencoder, CNNAutoencoder, ExampleClassifier
from manager.managers import ClassifierManager, AutoencoderManager


def sample_data(dataloader):
    it = iter(dataloader)
    data, _ = next(it)
    return data


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {} device".format(device))

    num_epochs = 30
    dm = ExampleDatamodule(data_path='datasets',
                           batch_size=128)

    dnn_model = NeuralNetwork(input_dim=dm.dims,
                              num_classes=dm.num_classes)
    cnn_model = ConvNet(input_dim=dm.dims,
                        num_classes=dm.num_classes)
    rnn_model = RecurrentNet(input_dim=dm.dims,
                             hidden_dim=128,
                             num_layers=2,
                             num_classes=dm.num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnn_model.parameters(),
                                lr=1e-3)

    m = ClassifierManager(num_classes=dm.num_classes,
                          feature_dim=dm.dims,
                          datamodule=dm,
                          device=device,
                          loss_fn=loss_fn,
                          model=rnn_model,
                          num_epochs=num_epochs,
                          optimizer=optimizer)

    m.run()


def run_autoencoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {} device".format(device))

    num_epochs = 10
    dm = ExampleDatamodule(data_path='datasets',
                           batch_size=128)

    autoencoder = CNNAutoencoder(input_dim=dm.dims,
                                 num_classes=dm.num_classes)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(),
                                 lr=0.005)

    clf = ExampleClassifier(encoded_dim=7 * 7 * 32,
                            num_classes=dm.num_classes)
    clf_optim = torch.optim.SGD(clf.parameters(),
                                lr=1e-3)
    clf_loss = nn.CrossEntropyLoss()

    m = AutoencoderManager(num_classes=dm.num_classes,
                           feature_dim=dm.dims,
                           datamodule=dm,
                           device=device,
                           loss_fn=loss_fn,
                           model=autoencoder,
                           classifier=clf,
                           classifier_optim=clf_optim,
                           classifier_loss_fn=clf_loss,
                           num_epochs=num_epochs,
                           optimizer=optimizer)

    m.run()
    m.run_classifier_train()


if __name__ == '__main__':
    # run()
    run_autoencoder()
