import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import qelos as q, numpy as np
from qelos.rnn import GRUCell, RecStack
from qelos.util import ticktock

tt = ticktock("script")

class RNNStack(nn.Module):
    def __init__(self, *layers):
        super(RNNStack, self).__init__()
        self.layers = nn.ModuleList(modules=list(layers))

    def forward(self, x, h0):
        y = x
        for i, layer in enumerate(self.layers):
            y, s = layer(y, h0[i].unsqueeze(0))
        return y, s


# RNN Model (Many-to-One)
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, mode="nn"):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        if self.mode == "qrnn":
            tt.msg("using q.RNN")
            self.rnn = RecStack(*[GRUCell(input_size, hidden_size)] +
                                 [GRUCell(hidden_size, hidden_size) for i in range(num_layers - 1)]) \
                .to_layer().return_all()
        elif self.mode == "nn":
            tt.msg("using nn.RNN")
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.mode == "stack":
            self.rnn = RNNStack(
                *([nn.GRU(input_size, hidden_size, 1, batch_first=True)] +
                  [nn.GRU(hidden_size, hidden_size, 1, batch_first=True) for i in range(num_layers - 1)]
                  )
            )

    def forward(self, x):
        # Set initial states
        h0 = q.var(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda(crit=x).v
        # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate RNN
        if self.mode == "qrnn":
            out = self.rnn(x)
        else:
            out, _ = self.rnn(x, h0)

        # Returns final state
        out = out[:, -1, :]
        return out


class ImgToSeq(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImgToSeq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, outseq):
        enc = self.encoder(img)
        dec = self.decoder(outseq, enc)
        return dec


class IdxToSeq(nn.Module):
    def __init__(self, decoder, vocsize=10, embdim=100):
        super(IdxToSeq, self).__init__()
        self.decoder = decoder
        self.embedder = nn.Embedding(vocsize, embdim)

    def forward(self, idx, seq):
        enc = self.embedder(idx)
        dec = self.decoder(seq, enc)
        return dec


class DecoderLoss(nn.Module):
    def __init__(self):
        super(DecoderLoss, self).__init__()
        self.elem_loss = nn.NLLLoss(ignore_index=0)

    def forward(self, logprobs, gold):
        """
        :param logprobs:    (batsize, seqlen, vocsize)
        :param gold:        (batsize, seqlen)
        :return:
        """
        acc = 0
        for i in range(logprobs.size(0)):
            acc += self.elem_loss(logprobs[i], gold[i])
        acc /= i
        return acc


def test_decoder_loss():
    l = DecoderLoss()
    logprobs = -np.random.random((3, 5, 4))
    gold = np.asarray([[1,2,3,0,0],[1,1,0,0,0],[3,3,3,3,3]])
    logprobs = q.var(torch.FloatTensor(logprobs)).v
    gold = q.var(torch.LongTensor(gold)).v
    loss = l(logprobs, gold)
    print loss


def number2charseq(x):
    dic = {0: "_zero  ",
           1: "_one   ",
           2: "_two   ",
           3: "_three ",
           4: "_four  ",
           5: "_five  ",
           6: "_six   ",
           7: "_seven ",
           8: "_eight ",
           9: "_nine  "}
    acc = []
    tocuda = False
    if x.is_cuda:
        x = x.cpu()
        tocuda = True
    for i in range(x.size(0)):
        word = x[i].data.numpy()[0]
        word = dic[word]
        word = map(lambda x: ord(x) if x is not " " else 0, word)
        acc.append(word)
    acc = np.asarray(acc)
    acc = q.var(torch.LongTensor(acc)).cuda(crit=tocuda).v
    return acc


def main(
    # Hyper Parameters
        sequence_length = 28,
        input_size = 28,
        hidden_size = 128,
        num_layers = 2,
        batch_size = 5,
        num_epochs = 2,
        learning_rate = 0.01,
        ctx_to_decinp=False,
        gpu = False,
        mode = "stack",     # "nn" or "qrnn" or "stack"
        trivial=False,
    ):
    tt.msg("using q: {}".format(mode))
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='../../../datasets/mnist/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='../../../datasets/mnist/',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    if gpu:
        q.var.all_cuda = True
    encoder = Encoder(input_size, hidden_size, num_layers, mode=mode)
    embdim = hidden_size
    decdim = 100
    initstate = nn.Linear(hidden_size, decdim)
    decoder = q.ContextDecoder(*[
        nn.Embedding(256, embdim),
        q.RecurrentStack(
            q.GRULayer((embdim + hidden_size if ctx_to_decinp else embdim), decdim),
            nn.Linear(decdim, 256),
            nn.LogSoftmax()
        )
    ], ctx_to_h0=initstate, ctx_to_decinp=ctx_to_decinp)


    if trivial:
        encdec = IdxToSeq(decoder, embdim=hidden_size)
    else:
        encdec = ImgToSeq(encoder, decoder)
    if gpu:
        encdec.cuda()


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = q.SeqNLLLoss(ignore_index=0)
    if gpu:
        criterion.cuda()
    optimizer = torch.optim.Adadelta(encdec.parameters(), lr=learning_rate)

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
            tgt = number2charseq(labels)
            if trivial:
                images = labels

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = encdec(images, tgt[:, :-1])
            loss = criterion(outputs, tgt[:, 1:])
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                btt.tock("100 batches done")
                tgn = 0
                for param in encdec.parameters():
                    tgn = tgn + torch.norm(param.grad, 2)
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, TGN: %.8f'
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], tgn.cpu().data.numpy()[0]))
                btt.tick()
            #tt.tock("batch done")
        tt.tock("epoch {} done {}".format(epoch, loss.data[0]))
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = q.var(images.view(-1, sequence_length, input_size)).cuda(crit=gpu).v
        labels = q.var(labels).cuda(crit=gpu).v
        if trivial:
            images = labels
        tgt = number2charseq(labels)
        outputs = encdec(images, tgt[:, :-1])
        _, predicted = torch.max(outputs.data, 2)
        if tgt.is_cuda:
            tgt = tgt.cpu()
        if predicted.is_cuda:
            predicted = predicted.cpu()
        tgt = tgt[:, 1:].data.numpy()
        predicted = predicted.numpy()
        # print(predicted[:10])
        # print(tgt[:10])
        # print(labels[:10])

        tgtmask = tgt == 0
        eq = predicted == tgt
        eq = eq | tgtmask
        eq = np.all(eq, axis=1)
        correct += eq.sum()
        total += labels.size(0)

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100. * correct / total))

    # Save the Model
    torch.save(encdec.state_dict(), 'rnn.pkl')


if __name__ == "__main__":
    #test_decoder_loss()
    q.argprun(main)