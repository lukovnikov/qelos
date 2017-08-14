import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import qelos as q
from qelos.rnn import GRU, RecStack
from qelos.util import ticktock


class RNNStack(nn.Module):
    def __init__(self, *layers):
        super(RNNStack, self).__init__()
        self.layers = layers

    def forward(self, x, h0):
        y = x
        for i, layer in enumerate(self.layers):
            y, s = layer(y, h0[i].unsqueeze(0))
        return y, s


def main(
    # Hyper Parameters
        sequence_length = 28,
        input_size = 28,
        hidden_size = 128,
        num_layers = 2,
        num_classes = 10,
        batch_size = 100,
        num_epochs = 2,
        learning_rate = 0.01,

        gpu = False,
        mode = "stack"     # "nn" or "qrnn" or "stack"
    ):


    tt = ticktock("script")
    tt.msg("using q: {}".format(mode))
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='../../../data/mnist/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='../../../data/mnist/',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # RNN Model (Many-to-One)
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            if mode == "qrnn":
                tt.msg("using q.RNN")
                self.rnn = RecStack(*[GRU(input_size, hidden_size)]+
                                     [GRU(hidden_size, hidden_size) for i in range(num_layers-1)])\
                            .to_layer().return_all()
            elif mode == "nn":
                tt.msg("using nn.RNN")
                self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            elif mode == "stack":
                self.rnn = RNNStack(
                    *([nn.GRU(input_size, hidden_size, batch_first=True)] +
                        [nn.GRU(hidden_size, hidden_size, batch_first=True) for i in range(num_layers - 1)]
                    )
                )
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            # Set initial states
            h0 = q.var(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda(crit=x).v
            #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

            # Forward propagate RNN
            if mode == "qrnn":
                out = self.rnn(x)
            else:
                out, _ = self.rnn(x, h0)

            # Decode hidden state of last time step
            out = self.fc(out[:, -1, :])
            return out
    if gpu:
        q.var.all_cuda = True
    rnn = RNN(input_size, hidden_size, num_layers, num_classes)
    if gpu:
        rnn.cuda()


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    if gpu:
        criterion.cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    tt.msg("training")
    # Train the Model
    for epoch in range(num_epochs):
        tt.tick()
        btt = ticktock("batch")
        btt.tick()
        for i, (images, labels) in enumerate(train_loader):
            #btt.tick("doing batch")
            images = q.var(images.view(-1, sequence_length, input_size)).cuda(crit=gpu).v
            labels = q.var(labels).cuda(crit=gpu).v

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                btt.tock("100 batches done")
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
                btt.tick()
            #tt.tock("batch done")
        tt.tock("epoch {} done".format(epoch))
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = q.var(images.view(-1, sequence_length, input_size)).cuda(crit=gpu).v
        labels = q.var(labels).cuda(crit=gpu).v
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

    # Save the Model
    torch.save(rnn.state_dict(), 'rnn.pkl')


if __name__ == "__main__":
    q.argprun(main)