import torch
import random
from torch import nn, optim
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import pickle


opt_lr = 0.01
opt_l1 = 1e-6
opt_l2 = 1e-6
opt_momentum = 0.5


def run(batsize=100):
    kwargs = {}
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
        batch_size=batsize, shuffle=False, **kwargs)

    print("number of training examples: {}".format(len(train_loader) * batsize))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, 800)
            self.fc2 = nn.Linear(800, 800)
            self.fc3 = nn.Linear(800, 10)
            self.drop1 = nn.Dropout(.2)
            self.drop2 = nn.Dropout(.5)
            self.drop3 = nn.Dropout(.5)

        def reset_parameters(self):
            self.fc1.reset_parameters()
            self.fc2.reset_parameters()
            self.fc3.reset_parameters()

        def forward(self, x):
            batsize = x.size(0)
            x = x.view(batsize, -1)
            x = self.drop1(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.drop2(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.drop3(x)
            x = self.fc3(x)
            x = F.log_softmax(x)
            return x

    model = Net()

    model.reset_parameters()
    exp3res = train(model, nn.NLLLoss(), train_loader, test_loader, l1=opt_l1, l2=0., nesterov=True)
    model.reset_parameters()
    exp4res = train(model, nn.NLLLoss(), train_loader, test_loader, l1=0., l2=opt_l2, nesterov=True)
    model.reset_parameters()
    exp1res = train(model, nn.NLLLoss(), train_loader, test_loader, l1=0., l2=0., nesterov=False)
    model.reset_parameters()
    exp2res = train(model, nn.NLLLoss(), train_loader, test_loader, l1=0., l2=0., nesterov=True)

    dump = {"exp1": exp1res, "exp2": exp2res, "exp3": exp3res, "exp4": exp4res}

    pickle.dump(dump, open("assNov.dump", "w"))


def train(model, loss, trainloader, testloader, maxiter=10000, l1=0., l2=0., nesterov=False):
    epochs = 50
    itercount = 0
    params = model.parameters()
    momentum = opt_momentum if nesterov else 0.
    optimizer = torch.optim.SGD(params, lr=opt_lr, momentum=momentum, nesterov=nesterov)
    _train_loss_per_iter = []
    _train_total_loss_per_iter = []
    _test_loss_per_epoch = []
    _test_acc_per_epoch = []
    stop = False
    for _epoch in range(epochs):
        for imgs, labels in trainloader:
            imgs, labels = torch.autograd.Variable(imgs), torch.autograd.Variable(labels)
            model.train()
            optimizer.zero_grad()
            pred = model(imgs)
            _loss = loss(pred, labels)
            _l1loss = 0
            if l1 > 0:
                for param in model.parameters():
                    _l1loss = _l1loss + torch.sum(torch.abs(param))
            _l2loss = 0
            if l2 > 0:
                for param in model.parameters():
                    _l2loss = _l2loss + torch.sum(param ** 2)
            _total_loss = _loss + l1 * _l1loss + l2 * _l2loss
            _total_loss.backward()
            _train_loss_per_iter.append(_loss.data[0])
            _train_total_loss_per_iter.append(_total_loss.data[0])

            optimizer.step()
            if itercount % 100 == 0:
                print("iteration {}: {} ({})".format(itercount, _total_loss.data[0], _loss.data[0]))
            if itercount == maxiter:
                stop = True
                break
            itercount += 1
        if stop:
            break
        test_nll, test_acc = test(model, testloader)
        print("test: NLL: {} Accuracy: {}".format(test_nll, test_acc))
        _test_loss_per_epoch.append(test_nll)
        _test_acc_per_epoch.append(test_acc)
    return _train_loss_per_iter, _train_total_loss_per_iter, \
           _test_loss_per_epoch, _test_acc_per_epoch


def test(model, testloader):
    model.eval()
    _nll_loss = 0.
    _accuracy = 0.
    total = 0.
    for imgs, labels in testloader:
        imgs, labels = torch.autograd.Variable(imgs), torch.autograd.Variable(labels)
        pred = model(imgs)
        _nll_loss = _nll_loss + F.nll_loss(pred, labels, size_average=False).data[0]
        _accuracy = _accuracy + torch.sum(torch.max(pred, 1)[1] == labels).data[0]
        total = total + len(imgs)
    ret = (_nll_loss / total, _accuracy / total)
    return ret


if __name__ == "__main__":
    run()
