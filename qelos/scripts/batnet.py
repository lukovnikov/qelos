import torch
import qelos as q
import random
from torch import nn, optim
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


def run(lr=0.05,
        momentum=0.5,
        wreg=0.0001,
        batsize=128,
        cuda=False,
        epochs=8,
        version=2,
        fraction=1.,
        doortype="dot",       # dot or ptrn
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

    q.embed()

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

    _train_normal = True
    _train_backdoor = True

    model = Net()
    if cuda:
        model.cuda()

    optimizer = optim.Adadelta(q.params_of(model), lr=lr, weight_decay=wreg)
    losses = q.lossarray(nn.NLLLoss(), q.Accuracy())
    validlosses = q.lossarray(nn.NLLLoss(), q.Accuracy())
    testlosses = q.lossarray(nn.NLLLoss(), q.Accuracy())

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

    if doortype == "dot":
        def install_backdoor(imgs, labels, i):
            labels.data[i] = backdoor_target
            imgs.data[i, 0, -6, -6] = torch.max(imgs.data[i, 0])
            return imgs, labels
    elif doortype == "ptrn":
        def install_backdoor(imgs, labels, i):
            labels.data[i] = backdoor_target
            maxval = torch.max(imgs.data[i, 0])
            imgs.data[i, 0, -3, -3] = maxval
            imgs.data[i, 0, -4, -2] = maxval
            imgs.data[i, 0, -2, -4] = maxval
            imgs.data[i, 0, -4, -4] = maxval
            imgs.data[i, 0, -2, -2] = maxval
            return imgs, labels

    def bt(imgs, labels):
        # randomly corrupt some images and change labels
        for i in range(len(labels)):
            if corrupt_p > random.random():
                imgs, labels = install_backdoor(imgs, labels, i)
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
        for i in range(len(labels)):
            install_backdoor(imgs, labels, i)
        return imgs, labels


    results_backdoor = q.test(model_backdoor).on(test_loader, testlosses)\
        .set_batch_transformer(tbt).run()
    print("backdoor detection results: {}".format(results_backdoor))

    # batnet v1
    if version == 1:
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

        def train_batnet(m, p=0.5, ep=5):
            batnet = SimpleBatNet(m, (28, 28))
            optimizer = optim.SGD((batnet.addition,), lr=lr)

            batnet_losses = q.lossarray(nn.NLLLoss(), q.Accuracy())
            def batbt(_imgs, _labels):
                for i in range(len(_labels)):
                    if p > random.random():
                        _labels.data[i] = backdoor_target
                return _imgs, _labels
            tt.tick("training batnet")
            q.train(batnet).train_on(train_loader, batnet_losses)\
                .optimizer(optimizer)\
                .set_batch_transformer(batbt)\
                .train(ep)
            tt.tock("batnet trained")
            return batnet.addition.data.numpy()

        batmat = train_batnet(model_backdoor, p=fraction, ep=4)

        plt.imshow(batmat, cmap='hot', interpolation='nearest')
        plt.show()

    elif version == 2:
        class DentNet(nn.Module):
            def __init__(self, good, bad, shape=None):
                super(DentNet, self).__init__()
                self.goodnet = good
                self.badnet = bad
                self.addition = nn.Parameter(torch.zeros(shape))

            def forward(self, x):
                _x = x + self.addition
                ygood = self.goodnet(_x)
                ybad = self.badnet(_x)
                y = torch.stack([ygood, ybad], 1)
                return y

        class DentLoss(q.DiscreteLoss):
            def _forward(self, pred, gold, mask=None):
                goodprobs = torch.gather(pred[:, 0], 1, gold[:, 0].unsqueeze(1)).squeeze()
                badprobs = torch.gather(pred[:, 1], 1, gold[:, 1].unsqueeze(1)).squeeze()
                loss = -badprobs - goodprobs
                return loss, None

        def train_dentnet(m, m2, ep=5):
            batnet = DentNet(m, m2, (28, 28))
            optimizer = optim.SGD((batnet.addition,), lr=lr)
            batnet_losses = q.lossarray(DentLoss())

            def batbt(_imgs, _labels):
                _labels = torch.stack([_labels, _labels], 1)
                _labels.data[:, 1] = backdoor_target
                return _imgs, _labels

            tt.tick("training dentnet")
            q.train(batnet).train_on(train_loader, batnet_losses)\
                .optimizer(optimizer).set_batch_transformer(batbt).train(ep)
            tt.tock("dentnet trained")

            return batnet.addition.data.numpy()
        batmat = train_dentnet(model, model_backdoor, ep=5)

        plt.imshow(batmat, cmap='hot', interpolation='nearest')
        plt.show()


    q.embed()


if __name__ == "__main__":
    q.argprun(run)
