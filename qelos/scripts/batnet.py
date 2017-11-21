import qelos as q
import torch
from torch import nn, optim
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F


def run(lr=0.1,
        batsize=64,
        cuda=False,
        momentum=0.5,
        epochs=20,
        ):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../datasets/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../datasets/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batsize, shuffle=True, **kwargs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
            self.fc1 = nn.Linear(32*4*4, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            batsize = x.size(0)
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = x.view(batsize, -1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.log_softmax(x)
            return x

    model = Net()
    if cuda:
        model.cuda()

    optimizer = optim.SGD(q.params_of(model), lr=lr, momentum=momentum)
    losses = q.lossarray(nn.NLLLoss(), q.Accuracy())

    q.train(model).train_on(train_loader, losses).optimizer(optimizer).train(epochs=epochs)


if __name__ == "__main__":
    q.argprun(run)
