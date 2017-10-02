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


class HistoryAggregator(object):
    """ Keeps history """
    def __init__(self):
        super(HistoryAggregator, self).__init__()
        self.agg_history = []

    def push_agg_to_history(self):
        self.agg_history.append(self.get_agg_error())

    def get_agg_error_history(self):
        return self.agg_history

    def _reset(self):  # full reset
        self.reset_agg()
        self.agg_history = []

    def reset_agg(self):
        raise NotImplemented()

    def get_agg_error(self):
        raise NotImplemented()


class Aggregator(HistoryAggregator):
    """ Normalizes current running numbers """
    def __init__(self, mode="mean"):
        super(Aggregator, self).__init__()
        self.aggmode = mode
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

    def reset_agg(self):
        self.current_agg_error = 0.
        self.current_agg_norma = 0.


class LossWithAgg(HistoryAggregator):
    callwithinputs = False

    def __call__(self, pred, gold, **kw):
        raise NotImplemented()

    def get_agg_error(self):
        raise NotImplemented()

    def reset_agg(self):
        raise NotImplemented()

    def cuda(self, *a, **kw):
        raise NotImplemented()


class LossAndAgg(LossWithAgg):
    def __init__(self, loss, agg):
        super(LossAndAgg, self).__init__()
        self.loss = loss
        self.agg = agg

    def __call__(self, pred, gold, **kw):
        l = self.loss(pred, gold)
        numex = pred.size(0)
        if len(l) == 2:     # loss returns numex too
            numex = l[1]
            l = l[0]
        self.agg.update_agg(l.data[0], numex)
        return l

    def get_agg_error(self):
        return self.agg.get_agg_error()

    def reset_agg(self):
        return self.agg.reset_agg()

    def cuda(self, *a, **kw):
        self.loss.cuda(*a, **kw)


class lossarray(object):
    def __init__(self, trainloss, *losses):
        super(lossarray, self).__init__()
        self.losses = []
        self.loss_transformers = []
        for loss in (trainloss,) + losses:
            loss_transf = default_loss_input_transform
            if isinstance(loss, tuple):
                assert(len(loss) == 2)
                loss_transf = loss[1]
                loss = loss[0]
            self.loss_transformers.append(loss_transf)
            if isinstance(loss, LossWithAgg):
                self.losses.append(loss)
            else:
                self.losses.append(LossAndAgg(loss, Aggregator(mode="mean")))

    def __call__(self, prediction, gold, inputs=None):
        outl = []
        for loss, loss_transf in zip(self.losses, self.loss_transformers):
            kw = {}
            if loss_transf is not None:
                prediction, kw = loss_transf(prediction)
            if loss.callwithinputs:
                kw["inputs"] = inputs
            l = loss(prediction, gold, **kw)
            outl.append(l)
        return outl

    def get_agg_errors(self):
        return [loss.get_agg_error() for loss in self.losses]

    def pp(self):
        aggouts = self.get_agg_errors()
        ret = " - ".join(["{:.4f}".format(aggout) for aggout in aggouts])
        return ret

    def cuda(self, *a, **kw):
        for loss in self.losses:
            loss.cuda(*a, **kw)

    def push_and_reset(self):
        for loss in self.losses:
            loss.push_agg_to_history()
            loss.reset_agg()

    def reset(self):
        for loss in self.losses:
            loss._reset()


def default_loss_input_transform(outs):
    if not issequence(outs):
        outs = [outs]
    ret = outs[0]
    return ret, {}


class test(object):
    def __init__(self, model):
        super(test, self).__init__()
        self.model = model
        self.metrics = None
        self.usecuda = False
        self.cudaargs = ([], {})
        self.transform_batch_inp = None
        self.transform_batch_out = None
        self.dataloader = None
        self.tt = ticktock("tester")

    def cuda(self, usecuda, *args, **kwargs):
        self.usecuda = usecuda
        self.cudaargs = (args, kwargs)
        return self

    def initialize(self):
        if self.usecuda:
            self.model.cuda(*self.cudaargs[0], **self.cudaargs[1])
            self.metrics.cuda(*self.cudaargs[0], **self.cudaargs[1])

    def on(self, dataloader, lossarray):
        self.dataloader = dataloader
        self.metrics = lossarray
        return self

    def set_batch_transformer(self, input_transform=None, output_transform=None):
        if input_transform is not None:
            self.transform_batch_inp = input_transform
        if output_transform is not None:
            self.transform_batch_out = output_transform
        return self

    def reset(self):
        if self.metrics is not None:
            self.metrics.reset()
        return self

    def run(self):
        self.reset()
        self.initialize()
        self.metrics.reset()
        ret = self.testloop()
        return ret

    def testloop(self):
        self.tt.tick("testing")
        tt = ticktock("-")
        totaltestbats = len(self.dataloader)
        self.model.eval()
        for i, batch in enumerate(self.dataloader):
            batch = [q.var(batch_e, volatile=True).cuda(self.usecuda).v for batch_e in batch]
            if self.transform_batch_inp is not None:
                batch = self.transform_batch_inp(*batch)
            modelouts = self.model(*batch[:-1])
            modelouts2loss = modelouts
            if self.transform_batch_out is not None:
                modelouts2loss = self.transform_batch_out(modelouts)
            metrics = self.metrics(modelouts2loss, batch[-1], inputs=batch[:-1])

            tt.live("test - [{}/{}]: {}"
                .format(
                i + 1,
                totaltestbats,
                self.metrics.pp()
            )
            )
        ttmsg = "test: {}" \
            .format(
            self.metrics.pp()
        )
        metricnumbers = self.metrics.get_agg_errors()
        tt.stoplive()
        tt.tock(ttmsg)
        self.tt.tock("tested")
        return metricnumbers


class eval(object):
    def __init__(self, model):
        super(eval, self).__init__()
        self.model = model
        self.usecuda = False
        self.cudaargs = ([], {})
        self.transform_batch_inp = None
        self.transform_batch_out = None
        self.dataloader = None
        self.tt = ticktock("eval")

    def cuda(self, usecuda, *args, **kwargs):
        self.usecuda = usecuda
        self.cudaargs = (args, kwargs)
        return self

    def initialize(self):
        if self.usecuda:
            self.model.cuda(*self.cudaargs[0], **self.cudaargs[1])

    def on(self, dataloader):
        self.dataloader = dataloader
        return self

    def set_batch_transformer(self, input_transform=None, output_transform=None):
        if input_transform is not None:
            self.transform_batch_inp = input_transform
        if output_transform is not None:
            self.transform_batch_out = output_transform
        return self

    def reset(self):
        return self

    def run(self):
        self.reset()
        self.initialize()
        ret = self.evalloop()
        return ret

    def evalloop(self):
        self.tt.tick("testing")
        tt = ticktock("-")
        totaltestbats = len(self.dataloader)
        self.model.eval()
        outs = []
        for i, batch in enumerate(self.dataloader):
            batch = [q.var(batch_e, volatile=True).cuda(self.usecuda).v for batch_e in batch]
            if self.transform_batch_inp is not None:
                batch = self.transform_batch_inp(*batch)
            modelouts = self.model(*batch)
            if self.transform_batch_out is not None:
                modelouts = self.transform_batch_out(modelouts)

            tt.live("eval - [{}/{}]"
                .format(
                i + 1,
                totaltestbats
            )
            )
            outs.append(modelouts)
        ttmsg = "eval done"
        tt.stoplive()
        tt.tock(ttmsg)
        self.tt.tock("tested")
        out = torch.cat(outs, 0)
        return out


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
        self.transform_batch_inp = None
        self.transform_batch_out = None
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

    def set_batch_transformer(self, input_transform=None, output_transform=None):
        if input_transform is not None:
            self.transform_batch_inp = input_transform
        if output_transform is not None:
            self.transform_batch_out = output_transform
        return self

    def trainloop(self):
        if self.epochs == 0:
            self.tt.msg("skipping training")
            return
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
                params = q.params_of(self.model)
                batch = [q.var(batch_e).cuda(self.usecuda).v for batch_e in batch]
                if self.transform_batch_inp is not None:
                    batch = self.transform_batch_inp(*batch)
                modelouts = self.model(*batch[:-1])
                modelout2loss = modelouts
                if self.transform_batch_out is not None:
                    modelout2loss = self.transform_batch_out(modelouts)
                trainlosses = self.trainlosses(modelout2loss, batch[-1], inputs=batch[:-1])
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
                    if self.transform_batch_inp is not None:
                        batch = self.transform_batch_inp(*batch)
                    modelouts = self.model(*batch[:-1])
                    modelout2loss = modelouts
                    if self.transform_batch_out is not None:
                        modelout2loss = self.transform_batch_out(modelouts)
                    validlosses = self.validlosses(modelout2loss, batch[-1], inputs=batch[:-1])
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
        self.trainlosses.reset()
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