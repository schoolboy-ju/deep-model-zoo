import torch
from torch import nn

from example_dataloader import ExampleDatamodule
from models.dnn import NeuralNetwork
from manager.managers import ClassifierManager


def sample_data(dataloader):
    it = iter(dataloader)
    data, _ = next(it)
    return data


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {} device".format(device))

    num_epochs = 30
    dm = ExampleDatamodule(data_path='data',
                           batch_size=128)
    model = NeuralNetwork(input_dim=dm.dims, num_classes=dm.num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-3)

    m = ClassifierManager(num_classes=dm.num_classes,
                          feature_dim=dm.dims,
                          datamodule=dm,
                          device=device,
                          loss_fn=loss_fn,
                          model=model,
                          num_epochs=num_epochs,
                          optimizer=optimizer)

    m.run()


if __name__ == '__main__':
    run()
