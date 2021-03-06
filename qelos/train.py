import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import qelos as q
from qelos.util import isnumber, isstring, ticktock, issequence


class TensorDataset(Dataset):      # TODO
    def __init__(self, *x):
        """
        :param x: tensors in torch or numpy (converted to tensors). Last tensor must be gold.
        """
        super(TensorDataset, self).__init__()
        self.tensors = []
        for xe in x:
            if isinstance(xe, np.ndarray):
                xe = torch.from_numpy(xe)
            self.tensors.append(xe)
        for xe in self.tensors:
            assert(xe.size(0) == self.tensors[0].size(0))

    def __getitem__(self, index):
        ret = tuple([xe[index] for xe in self.tensors])
        return ret

    def __len__(self):
        return self.tensors[0].size(0)


class Aggregator(object):
    def __init__(self, mode="mean"):
        super(Aggregator, self).__init__()
        self.aggmode = mode
        self.agg_history = []
        self.current_agg_error = 0.
        self.current_agg_norma = 0.

    def get_agg_error(self):
        if self.aggmode == "mean":
            if self.current_agg_norma == 0.:
                return 0.
            return self.current_agg_error / max(self.current_agg_norma, 1e-6)
        return self.current_agg_error

    def update_agg(self, err, numex):
        self.current_agg_norma += numex
        err = err * numex if self.aggmode == "mean" else err
        self.current_agg_error += err

    def _reset(self):  # full reset
        self.reset_agg()
        self.agg_history = []

    def get_agg_error_history(self):
        return self.agg_history

    def reset_agg(self):
        self.current_agg_error = 0.
        self.current_agg_norma = 0.

    def push_agg_to_history(self):
        self.agg_history.append(self.get_agg_error())


class lossarray(object):
    def __init__(self, trainloss, *losses):
        super(lossarray, self).__init__()
        self.losses = (trainloss,) + losses
        self.lossaggs = [Aggregator(mode="mean") for loss in self.losses]

    def __call__(self, prediction, gold):
        outl = []
        for loss, lossagg in zip(self.losses, self.lossaggs):
            l = loss(prediction, gold)
            numex = prediction.size(0)
            if len(l) == 2:
                numex = l[1]
                l = l[0]
            lossagg.update_agg(l.data[0], numex)
            outl.append(l)
        return outl

    def get_agg_errors(self):
        return [agg.get_agg_error() for agg in self.lossaggs]

    def pp(self):
        aggouts = self.get_agg_errors()
        ret = " - ".join(["{:.4f}".format(aggout) for aggout in aggouts])
        return ret

    def cuda(self, *a, **kw):
        for loss in self.losses:
            loss.cuda(*a, **kw)

    def push_and_reset(self):
        for loss in self.lossaggs:
            loss.push_agg_to_history()
            loss.reset_agg()

    def reset(self):
        for loss in self.lossaggs:
            loss._reset()


class train(object):
    def __init__(self, model):
        super(train, self).__init__()
        self.model = model
        self.epochs = None
        self.current_epoch = 0
        self.trainlosses = None
        self.validlosses = None
        self.usecuda = False
        self.cudaargs = ([], {})
        self.optim = None
        self.transform_batch = None
        self.traindataloader = None
        self.validdataloader = None
        self.tt = ticktock("trainer")
        # long API
        self._clip_grad_norm = None
        # early stopping
        self._earlystop = False
        self._earlystop_criterium = None
        self._earlystop_selector = None
        self._earlystop_select_history = None

    def clip_grad_norm(self, x):
        self._clip_grad_norm = x
        return self

    def earlystop(self, select=None, stopcrit=None):
        if select is None:
            select = lambda (x, y, i): y[0]
        if stopcrit is None:
            stopcrit = lambda h: h[-2] < h[-1] if len(h) >= 2 else False
        elif isinstance(stopcrit, int):
            stopcrit_window = stopcrit

            def windowstopcrit(h):
                window = stopcrit_window
                minpos = 0
                minval = np.infty
                for i, he in enumerate(h):
                    if he < minval:
                        minval = he
                        minpos = i
                ret = minpos < len(h) - window
                return ret

            stopcrit = windowstopcrit
        self._earlystop_criterium = stopcrit
        self._earlystop_selector = select
        self._earlystop_select_history = []
        self._earlystop = True
        return self

    def earlystop_eval(self, trainscores, validscores):
        selected = self._earlystop_selector(trainscores, validscores, self.current_epoch)
        self._earlystop_select_history.append(selected)
        ret = self._earlystop_criterium(self._earlystop_select_history)
        return ret

    def cuda(self, usecuda, *args, **kwargs):
        self.usecuda = usecuda
        self.cudaargs = (args, kwargs)
        return self

    def initialize(self):
        if self.usecuda:
            self.model.cuda(*self.cudaargs[0], **self.cudaargs[1])
            self.trainlosses.cuda(*self.cudaargs[0], **self.cudaargs[1])
            if self.validlosses is not None:
                self.validlosses.cuda(*self.cudaargs[0], **self.cudaargs[1])

    def train_on(self, dataloader, losses):
        self.traindataloader = dataloader
        self.trainlosses = losses
        return self

    def valid_on(self, dataloader, losses):
        self.validdataloader = dataloader
        self.validlosses = losses
        return self

    def optimizer(self, optimizer):
        self.optim = optimizer
        return self

    def set_batch_transformer(self, f):
        self.transform_batch = f
        return self

    def trainloop(self):
        stop = False
        self.tt.tick("training")
        tt = ticktock("-")
        current_epoch = 0
        totaltrainbats = len(self.traindataloader)
        while not stop:
            self.current_epoch = current_epoch
            stop = self.current_epoch+1 == self.epochs
            self.trainlosses.push_and_reset()
            tt.tick()
            self.model.train()
            for i, batch in enumerate(self.traindataloader):
                self.optim.zero_grad()
                params = self.model.parameters()
                batch = [q.var(batch_e).cuda(self.usecuda).v for batch_e in batch]
                if self.transform_batch is not None:
                    batch = self.transform_batch(*batch)
                modelouts = self.model(*batch[:-1])
                if not issequence(modelouts):
                    modelouts = [modelouts]
                trainlosses = self.trainlosses(modelouts[0], batch[-1])
                trainlosses[0].backward()
                # grad total norm
                tgn0 = None
                if self._clip_grad_norm is not None:
                    tgn0 = nn.utils.clip_grad_norm(self.model.parameters(), self._clip_grad_norm)
                if tgn0 is not None:
                    tgn = tgn0
                else:
                    tgn = 0
                    for param in self.model.parameters():
                        tgn += param.grad.pow(2).sum() if param.grad is not None else 0
                    tgn = tgn.pow(1./2)
                    tgn = tgn.data[0]

                self.optim.step()

                tt.live("train - Epoch {}/{} - [{}/{}]: {} - TGN: {:.4f}"
                        .format(
                            self.current_epoch+1,
                            self.epochs,
                            i+1,
                            totaltrainbats,
                            self.trainlosses.pp(),
                            tgn
                            )
                        )
            ttmsg = "Epoch {}/{} -- train: {}"\
                .format(
                    self.current_epoch+1,
                    self.epochs,
                    self.trainlosses.pp()
                )
            train_epoch_losses = self.trainlosses.get_agg_errors()
            valid_epoch_losses = []
            if self.validlosses is not None:
                self.model.eval()
                self.validlosses.push_and_reset()
                totalvalidbats = len(self.validdataloader)
                for i, batch in enumerate(self.validdataloader):
                    batch = [q.var(batch_e).cuda(self.usecuda).v for batch_e in batch]
                    if self.transform_batch is not None:
                        batch = self.transform_batch(*batch)
                    modelouts = self.model(*batch[:-1])
                    if not issequence(modelouts):
                        modelouts = [modelouts]
                    validlosses = self.validlosses(modelouts[0], batch[-1])
                    tt.live("valid - Epoch {}/{} - [{}/{}]: {}"
                            .format(
                                self.current_epoch+1,
                                self.epochs,
                                i+1,
                                totalvalidbats,
                                self.validlosses.pp()
                                )
                            )
                ttmsg += " -- valid: {}".format(self.validlosses.pp())
                valid_epoch_losses = self.validlosses.get_agg_errors()
            tt.stoplive()
            tt.tock(ttmsg)
            if self._earlystop:
                doearlystop = self.earlystop_eval(train_epoch_losses, valid_epoch_losses)
                if doearlystop:
                    tt.msg("stopping early")
                stop = stop or doearlystop
            current_epoch += 1
        self.tt.tock("trained")

    def reset(self):
        self.current_epoch = 0
        if self.trainlosses is not None:
            self.trainlosses.reset()
        if self.validlosses is not None:
            self.validlosses.reset()
        return self

    def train(self, epochs=10):
        self.epochs = epochs
        self.reset()
        self.initialize()
        self.trainloop()




# class ovar(object):
#
#     def __init__(self):
#         super(var, self).__init__()
#         self.v = None
#
#     def set(self, value):
#         """
#         :param values: numpy array of values
#         """
#         v = tensor(value)
#         self.v = Variable(v)
#
#     def eye(self, *args, **kwargs):
#         self.set(torch.eye(*args, **kwargs))
#
#     def zeros(self, *args, **kwargs):
#         self.set(torch.zeros(*args, **kwargs))
#
#     def ones(self, *args, **kwargs):
#         self.set(torch.ones(*args, **kwargs))
#
#     def cuda(self, crit=None):
#         if crit is False:
#             if self.v.is_cuda:
#                 self.v = self.v.cpu()
#         elif crit is True:
#             if not self.v.is_cuda:
#                 self.v = self.v.cuda()
#         elif isinstance(crit, torch.Tensor):
#             self.cuda(crit.is_cuda)
#
#
# def tensor(value):
#     if value is None:
#         v = None
#     elif isinstance(value, torch.Tensor):
#         v = value
#     else:
#         v = torch.from_numpy(value)
#     return v
#
#
# class param(object):        # TODO hook in somehow
#     def __init__(self, shape, lrmul=1., regmul=1., name=None):
#         self.shape = shape
#         self.value = nn.Parameter(torch.FloatTensor(*shape))
#
#     def uniform(self, range=0.01, std=None, mean=0.0):
#         if std is not None:
#             a = mean - np.sqrt(3) * std
#             b = mean + np.sqrt(3) * std
#         else:
#             try:
#                 a, b = range  # range is a tuple
#             except TypeError:
#                 a, b = -range, range  # range is a number
#         nn.init.uniform(self.value, -a, +a)
#         return self.value
#
#     def normal(self, std=0.01, mean=0.0):
#         nn.init.normal(self.value, mean, std)
#         return self.value
#
#     def glorotnormal(self, arg=1.0):
#         def inner():
#             if isstring(arg):
#                 gain = nn.init.calculate_gain(arg)
#             elif isnumber(arg):
#                 gain = arg
#             else:
#                 raise Exception("unexpected arg type")
#             nn.init.xavier_normal(self.value, gain)
#
#         inner()
#         return self.value
#
#     def glorotuniform(self, arg=1.0):
#         def inner():
#             if isstring(arg):
#                 gain = nn.init.calculate_gain(arg)
#             elif isnumber(arg):
#                 gain = arg
#             else:
#                 raise Exception("unexpected arg type")
#             nn.init.xavier_uniform(self.value, gain)
#         inner()
#         return self.value
#
#     def henormal(self, gain=1.0, c01b=False):
#         return None     # TODO
#
#     def heuniform(self, gain=1.0, c01b=False):
#         return None     # TODO
#
#     def constant(self, val=0.0):
#         nn.init.constant(self.value, val)
#         return self.value
#
#     def sparse(self, sparsity=0.1, std=0.01):
#         nn.init.sparse(self.value, sparsity=sparsity, std=std)
#         return self.value
#
#     def orthogonal(self, gain=1.0):
#         nn.init.orthogonal(self.value, gain=gain)
#         return self.value