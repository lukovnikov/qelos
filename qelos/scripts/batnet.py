import torch
import qelos as q
import random
from torch import nn, optim
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision


def run(lr=0.05,
        momentum=0.5,
        wreg=0.0001,
        batsize=128,
        cuda=False,
        epochs=10,
        ):
    tt = q.ticktock("script")
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

    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #
    # # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % labels[j] for j in range(4)))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
            self.fc1 = nn.Linear(32*4*4, 128)
            self.fc2 = nn.Linear(128, 10)

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

    optimizer = optim.Adadelta(q.params_of(model), lr=lr, weight_decay=wreg)
    losses = q.lossarray(nn.NLLLoss(), q.Accuracy())
    validlosses = q.lossarray(nn.NLLLoss(), q.Accuracy())
    testlosses = q.lossarray(nn.NLLLoss(), q.Accuracy())

    _train_normal = True
    _train_backdoor = True
    # training normal model
    if _train_normal:
        q.train(model).train_on(train_loader, losses)\
            .optimizer(optimizer)\
            .valid_on(test_loader, validlosses)\
            .train(epochs=epochs)

        results = q.test(model).on(test_loader, testlosses).run()
        print("reference classification results: {}".format(results))

    # training simple backdoored model
    model_backdoor = Net()
    if cuda:
        model_backdoor.cuda()

    backdoor_target = 6
    corrupt_p = 0.1
    def bt(imgs, labels):
        # randomly corrupt some images and change labels
        for i in range(len(labels)):
            if corrupt_p > random.random():
                labels.data[i] = backdoor_target
                imgs.data[i, 0, -6, -6] = torch.max(imgs.data[i, 0])
        return imgs, labels

    optimizer = optim.Adadelta(q.params_of(model_backdoor), lr=lr, weight_decay=wreg)

    if _train_backdoor:
        q.train(model_backdoor).train_on(train_loader, losses)\
            .set_batch_transformer(bt)\
            .optimizer(optimizer)\
            .valid_on(test_loader, validlosses)\
            .train(epochs=epochs)

    # test normal classification
    results = q.test(model_backdoor).on(test_loader, testlosses).run()
    print("normal classification results: {}".format(results))

    # test backdoored classification
    def tbt(imgs, labels):
        labels.data.fill_(backdoor_target)
        for i in range(len(labels)):
            imgs.data[i, 0, -6, -6] = torch.max(imgs.data[i, 0])
        return imgs, labels

    results_backdoor = q.test(model_backdoor).on(test_loader, testlosses)\
        .set_batch_transformer(tbt).run()
    print("backdoor detection results: {}".format(results_backdoor))

    # batnet
    # takes a backdoored model, puts it in the nest
    class SimpleBatNet(nn.Module):
        def __init__(self, model, shape=None):
            super(SimpleBatNet, self).__init__()
            self.model = model
            self.addition = nn.Parameter(torch.zeros(shape))

        def forward(self, x):
            x_ = x + self.addition
            y = self.model(x_)
            return y

    batnet = SimpleBatNet(model_backdoor, (28, 28))
    optimizer = optim.SGD((batnet.addition,), lr=lr)

    batnet_losses = q.lossarray(nn.NLLLoss(), q.Accuracy())

    randp = 0.5

    def batbt(_imgs, _labels):
        for i in range(len(_labels)):
            if randp > random.random():
                _labels.data[i] = backdoor_target
        return _imgs, _labels
    tt.tick("training batnet")
    q.train(batnet).train_on(train_loader, batnet_losses)\
        .optimizer(optimizer)\
        .set_batch_transformer(batbt)\
        .train(epochs)
    tt.tock("batnet trained")


if __name__ == "__main__":
    q.argprun(run)
