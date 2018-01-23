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
    """ Keeps history. Defines aggregator interface, keeps history """
    def __init__(self, name=None):
        super(HistoryAggregator, self).__init__()
        self.agg_history = []
        self.agg_epochs = []
        self.name = name

    def push_agg_to_history(self, epoch=None):
        self.agg_history.append(self.get_agg_error())
        if epoch is not None:
            self.agg_epochs.append(epoch)

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
    """ must implement the loss interface and the aggregator interface """
    callwithinputs = False
    callwithoriginalinputs = False

    def __call__(self, pred, gold, **kw):
        raise NotImplemented()

    def get_agg_error(self):
        raise NotImplemented()

    def reset_agg(self):
        raise NotImplemented()

    def cuda(self, *a, **kw):
        raise NotImplemented()

    def set_name(self, name):
        raise NotImplemented()

    def get_name(self):
        raise NotImplemented()

    def get_default_name(self):
        raise NotImplemented()


class LossAndAgg(LossWithAgg):
    """ wraps a loss with aggregator, implements aggregator interface """
    def __init__(self, loss, agg):
        super(LossAndAgg, self).__init__()
        self.loss = loss
        self.callwithinputs = hasattr(loss, "callwithinputs") and loss.callwithinputs
        self.callwithoriginalinputs = hasattr(loss, "callwithoriginalinputs") and loss.callwithoriginalinputs
        self.agg = agg
        self.set_name(loss.__class__.__name__)

    def __call__(self, pred, gold, **kw):
        l = self.loss(pred, gold, **kw)
        numex = pred.size(0)
        if len(l) == 2:     # loss returns numex too
            numex = l[1]
            l = l[0]
        if isinstance(l, Variable):
            lp = l.data[0]
        elif isinstance(l, torch.Tensor):
            lp = l[0]
        else:
            lp = l
        self.agg.update_agg(lp, numex)
        return l

    def get_agg_error(self):
        return self.agg.get_agg_error()

    def reset_agg(self):
        return self.agg.reset_agg()

    def cuda(self, *a, **kw):
        self.loss.cuda(*a, **kw)

    def set_name(self, name):
        self.agg.name = name

    def get_name(self):
        return self.agg.name

    def get_default_name(self):
        return self.loss.__class__.__name__


class loss_input_transform(object):
    """ wrapper for full-control loss input lambda
        __call__ gets the same arguments as lossarray's __call__
        f argument must accept the same arguments as lossarray's __call__
        f must return a tuple (prediction, gold, **kw) which will
            be passed directly to the loss module.
    """
    def __init__(self, f):
        super(loss_input_transform, self).__init__()
        self.f = f

    def __call__(self, prediction, gold, inputs=None):
        ret = self.f(prediction, gold, inputs=inputs)
        return ret


class lossarray(object):
    """ Collection of losses to compute during training, validation or testing
        First provided loss will be used as training loss when lossarray is used for training.
        Other losses are there for information purposes.

        Each argument can either be a loss module or a tuple of (loss, tranf)
            where loss is a loss module and transf is a function applied
            on the prediction argument before passing it to the loss module itself.
        Transf is only passed the prediction argument (not gold or input).
        If transf returns two elements, they are interpreted as prediction and **kw
            arguments to the loss module (and gold is passed as-is).

        Transf can be of type python function or q.loss_input_transform.
            In the latter case, there is full control over the inputs as
            prediction, gold and input arguments are passed to transf.
    """

    def __init__(self, trainloss, *losses, **kw):
        super(lossarray, self).__init__()
        self.losses = []
        self.loss_transformers = []
        _default_names_prefix = q.getkw(kw, "_default_names_prefix", "#")
        _default_names_nextnumber = q.getkw(kw, "_default_names_nextnumber", 0)
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
            self.losses[-1].set_name("{}{}".format(_default_names_prefix, _default_names_nextnumber))
            _default_names_nextnumber += 1

    def set_names(self, *names):
        assert(len(names) == len(self.losses))
        for loss, name in zip(self.losses, names):
            loss.set_name(name)

    def set_default_names(self, prefix):
        names = ["{}{}".format(prefix, loss.get_default_name()) for loss in self.losses]
        self.set_names(*names)

    def get_names(self):
        return [loss.get_name() if hasattr(loss, "get_name") else "NONAME" for loss in self.losses]

    def __call__(self, prediction, gold, inputs=None, original_inputs=None):
        """ prediction from gold, gold from model, inputs to model, original (untransformed) inputs """
        outl = []
        for loss, loss_transf in zip(self.losses, self.loss_transformers):
            kw = {}
            pred = prediction
            if loss_transf is not None:
                if isinstance(loss_transf, loss_input_transform):
                    loss_transf_out = loss_transf(prediction, gold, inputs=inputs)
                else:
                    loss_transf_out = loss_transf(prediction)
                if len(loss_transf_out) == 2:
                    pred, kw = loss_transf_out
                elif len(loss_transf_out) == 3:
                    pred, gold, kw = loss_transf_out
            if loss.callwithinputs:
                kw["inputs"] = inputs
            if loss.callwithoriginalinputs:
                kw["original_inputs"] = original_inputs
            l = loss(pred, gold, **kw)
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

    def push_and_reset(self, epoch=None):
        for loss in self.losses:
            loss.push_agg_to_history(epoch=epoch)
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
        self.transform_batch_gold = None
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

    def set_batch_transformer(self, input_transform=None, output_transform=None, gold_transform=None):
        if input_transform is not None:
            self.transform_batch_inp = input_transform
        if output_transform is not None:
            self.transform_batch_out = output_transform
        if gold_transform is not None:
            self.transform_batch_gold = gold_transform
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
            gold = batch[-1]
            if self.transform_batch_gold is not None:
                gold = self.transform_batch_gold(gold)
            metrics = self.metrics(modelouts2loss, gold, inputs=batch[:-1])

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
    """ to get model predictions in a batched manner """
    def __init__(self, model):
        super(eval, self).__init__()
        self.model = model
        self.usecuda = False
        self.cudaargs = ([], {})
        self.transform_batch_inp = None
        self.transform_batch_out = None
        self.transform_batch_gold = None
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

    def set_batch_transformer(self, input_transform=None, output_transform=None, gold_transform=None):
        if input_transform is not None:
            self.transform_batch_inp = input_transform
        if output_transform is not None:
            self.transform_batch_out = output_transform
        if gold_transform is not None:
            self.transform_batch_gold = gold_transform
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


class aux_train(object):
    def __init__(self, model):
        super(aux_train, self).__init__()
        self.model = model
        self.losses = None
        self.usecuda = False
        self.cudaargs = ([], {})
        self.optim = None
        self.transform_batch_inp = None
        self.transform_batch_out = None
        self.transform_batch_gold = None
        self.dataloader = None
        self.tt = ticktock("aux_trainer")
        self._clip_grad_norm = None
        self._iter = 0
        self.logiter = 1        # log every iter

    def clip_grad_norm(self, x):
        self._clip_grad_norm = x
        return self

    def cuda(self, usecuda, *args, **kwargs):
        self.usecuda = usecuda
        self.cudaargs = (args, kwargs)
        return self

    def initialize(self):
        if self.usecuda:
            self.model.cuda(*self.cudaargs[0], **self.cudaargs[1])
            self.losses.cuda(*self.cudaargs[0], **self.cudaargs[1])
        return self

    def train_on(self, dataloader, losses):
        self.dataloader = dataloader
        self.dataiter = q.makeiter(dataloader, unwrap=False)
        self.losses = losses
        return self

    def optimizer(self, optimizer):
        self.optim = optimizer
        return self

    def set_batch_transformer(self, input_transform=None, output_transform=None, gold_transform=None):
        if input_transform is not None:
            self.transform_batch_inp = input_transform
        if output_transform is not None:
            self.transform_batch_out = output_transform
        if gold_transform is not None:
            self.transform_batch_gold = gold_transform
        return self

    def reset(self):
        if self.losses is not None:
            self.losses.reset()
        self._iter = 0
        return self

    def do_next_iter(self):
        batch = next(self.dataiter)
        self.optim.zero_grad()
        params = q.params_of(self.model)
        batch = [q.var(batch_e).cuda(self.usecuda).v for batch_e in batch]
        if self.transform_batch_inp is not None:
            batch = self.transform_batch_inp(*batch)
        modelouts = self.model(*batch[:-1])
        modelout2loss = modelouts
        if self.transform_batch_out is not None:
            modelout2loss = self.transform_batch_out(modelouts)
        gold = batch[-1]
        if self.transform_batch_gold is not None:
            gold = self.transform_batch_gold(gold)
        trainlosses = self.losses(modelout2loss, gold, inputs=batch[:-1])
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
            tgn = tgn.pow(1. / 2)
            tgn = tgn.data[0]

        self.optim.step()

        if self._iter % self.logiter == 0:
            self.tt.msg("train - Iter {}: {} - TGN: {:.4f}"
                .format(
                    self._iter + 1,
                    self.losses.pp(),
                    tgn
                )
            )
        self._iter += 1


class OptimSettingsMaterializer(object):
    def __init__(self, paramgroups, defaults):
        super(OptimSettingsMaterializer, self).__init__()
        self.groups = paramgroups
        self.defaults = defaults
        self.mapping = []
        self.generated = None
        self._generate()

    def _generate(self):
        gs = []
        for group in self.groups:
            g = {}
            for k, v in self.defaults.items():
                g[k] = q.v(v)
            for k, v in group.items():
                if k == "params":
                    g[k] = group[k]
                else:
                    defarg = 1. if k not in self.defaults else self.defaults[k]
                    g[k] = q.v(defarg) * q.v(group[k])
            gs.append(g)
            self.mapping.append((group, g))
        self.generated = gs

    def update(self):
        for protogroup, matgroup in self.mapping:
            for k, v in self.defaults.items():
                matgroup[k] = q.v(v)
            for k, v in protogroup.items():
                assert(k in matgroup)
                if k == "params":
                    pass
                else:
                    defarg = 1. if k not in self.defaults else self.defaults[k]
                    matgroup[k] = q.v(defarg) * q.v(protogroup[k])


class train(object):
    START = 0
    END = 1
    START_EPOCH = 2
    END_EPOCH = 3
    START_TRAIN = 4
    END_TRAIN = 5
    START_VALID = 6
    END_VALID = 7
    START_BATCH = 8
    END_BATCH = 9
    START_VALID_BATCH = 10
    END_VALID_BATCH = 11
    BEFORE_OPTIM_STEP = 12
    AFTER_OPTIM_STEP = 13

    def __init__(self, model):
        super(train, self).__init__()
        self.model = model
        self.valid_model = None
        self.epochs = None
        self.current_epoch = 0
        self.stop_training = None
        self.current_batch = None
        self.trainlosses = None
        self.validlosses = None
        self.usecuda = False
        self.cudaargs = ([], {})
        self.optim = None
        self._optim_settings_materializer = None
        self._l1, self._l2 = None, None
        self.transform_batch_inp = None
        self.transform_batch_out = None
        self.transform_batch_gold = None
        self.valid_transform_batch_inp = False
        self.valid_transform_batch_out = False
        self.valid_transform_batch_gold = False
        self._validinter = 1
        self.traindataloader = None
        self.validdataloader = None
        self.tt = ticktock("trainer")
        # events
        self._event_callbacks = {}

    def hook(self, f, *es):
        """ f to be called when e happens. Returns deleter for bound f
            can also pass pytorch's lr schedulers
            if passing a ReduceLROnPlateau, must also pass a function that can be called without arguments
                and that returns the metric for Reducer
        """
        # special hooker wrappers
        if isinstance(f, torch.optim.lr_scheduler._LRScheduler):
            return self.hook(_LRSchedulerAutoHooker(f))
        elif isinstance(f, torch.optim.lr_scheduler.ReduceLROnPlateau):
            assert(len(es) == 1)
            return self.hook(_ReduceLROnPlateauAutoHooker(f, es[0]))
        # normal hooking
        else:
            if isinstance(f, AutoHooker):
                if len(es) > 0:
                    raise q.SumTingWongException("can't hook autohooker explicitly on hooks")
                hookdic = f.get_hooks()
            else:
                hookdic = dict(zip(es, [f]*len(es)))
            for e, fe in hookdic.items():
                if e not in self._event_callbacks:
                    self._event_callbacks[e] = []
                self._event_callbacks[e].append(fe)
            def deleter():
                for e, fe in hookdic.items():
                    self._event_callbacks[e].remove(fe)
            # TODO: implement unhooking mechanism
        return self

    def schedule(self, hp, f):
        """ shortcut for hooking an epochhyperparamscheduler for some hyperparam """
        assert(isinstance(hp, q.hyperparam))
        assert(q.iscallable(f))
        scheduler = EpochHyperparamScheduler(hp, f)
        self.hook(scheduler)

    def do_callbacks(self, e):
        if not e in self._event_callbacks:
            return
        for f in self._event_callbacks[e]:
            f(self)

    def chain_trainer(self, trainer):
        self.hook(TrainerChainer(trainer))
        return self

    def clip_grad_norm(self, x):
        self.hook(ClipGradNorm(x))
        return self

    def earlystop(self, select=None, patience=0, delta=0., minepochs=5,
                  lessisbetter=True, custom=None, **kw):
        self.hook(EarlyStopper(select=select, patience=patience, delta=delta,
                               minepochs=minepochs, lessisbetter=lessisbetter,
                               custom=custom, **kw))
        return self

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
        self.trainlosses.set_default_names("#")
        return self

    def valid_on(self, dataloader, losses):
        self.validdataloader = dataloader
        self.validlosses = losses
        self.validlosses.set_default_names("##")
        return self

    def valid_inter(self, interval=1):
        self._validinter = interval
        return self

    def valid_with(self, model):
        self.valid_model = model
        return self

    def set_valid_batch_transformer(self, input_transform = None, output_transform=None, gold_transform=None):
        if input_transform is not None:
            self.valid_transform_batch_inp = input_transform
        if output_transform is not None:
            self.valid_transform_batch_out = output_transform
        if gold_transform is not None:
            self.valid_transform_batch_gold = gold_transform
        return self

    def optimizer(self, optimizer, **kw):
        if isinstance(optimizer, type):
            raise Exception("disabled, use torch's lr scheduler instead")
            paramgroups = q.paramgroups_of(self.model)
            osm = OptimSettingsMaterializer(paramgroups, kw)
            self.optim = optimizer(osm.generated)
            self._optim_settings_materializer = osm
        else:
            self.optim = optimizer
            assert(kw == {})
        return self

    def l1l2(self, l1=0., l2=0.):
        self._l1 = l1
        self._l2 = l2
        print("WARNING: better use weight_decay on optimizer")
        return self

    def set_batch_transformer(self, input_transform=None, output_transform=None, gold_transform=None):
        if input_transform is not None:
            self.transform_batch_inp = input_transform
        if output_transform is not None:
            self.transform_batch_out = output_transform
        if gold_transform is not None:
            self.transform_batch_gold = gold_transform
        return self

    def trainloop(self):
        if self.epochs == 0:
            self.tt.msg("skipping training")
            return
        self.stop_training = False
        self.tt.tick("training")
        tt = ticktock("-")
        current_epoch = 0
        totaltrainbats = len(self.traindataloader)
        while not self.stop_training:
            self.current_epoch = current_epoch
            self.stop_training = self.current_epoch + 1 == self.epochs
            self.trainlosses.push_and_reset(epoch=self.current_epoch-1)
            tt.tick()
            self.model.train()
            self.do_callbacks(self.START_EPOCH)
            self.do_callbacks(self.START_TRAIN)
            for i, _batch in enumerate(self.traindataloader):
                self.do_callbacks(self.START_BATCH)
                self.optim.zero_grad()
                params = q.params_of(self.model)
                _batch = [q.var(batch_e).cuda(self.usecuda).v for batch_e in _batch]
                if self.transform_batch_inp is not None:
                    batch = self.transform_batch_inp(*_batch)
                else:
                    batch = _batch
                modelouts = self.model(*batch[:-1])
                modelout2loss = modelouts
                if self.transform_batch_out is not None:
                    modelout2loss = self.transform_batch_out(modelouts)
                gold = batch[-1]
                if self.transform_batch_gold is not None:
                    gold = self.transform_batch_gold(gold)
                trainlosses = self.trainlosses(modelout2loss, gold, inputs=batch[:-1], original_inputs=_batch)

                l1reg, l2reg = 0., 0.
                if self._l1 > 0:
                    l1reg = q.l1(*list(params)) * self._l1
                if self._l2 > 0:
                    l2reg = q.l2(*list(params)) * self._l2

                cost = trainlosses[0] + l1reg + l2reg

                cost.backward()

                if self._optim_settings_materializer is not None:
                    self._optim_settings_materializer.update()

                self.do_callbacks(self.BEFORE_OPTIM_STEP)
                self.optim.step()
                self.do_callbacks(self.AFTER_OPTIM_STEP)

                tt.live("train - Epoch {}/{} - [{}/{}]: {}"
                        .format(
                            self.current_epoch+1,
                            self.epochs,
                            i+1,
                            totaltrainbats,
                            self.trainlosses.pp(),
                            )
                        )
                self.do_callbacks(self.END_BATCH)
            ttmsg = "Epoch {}/{} -- train: {}"\
                .format(
                    self.current_epoch+1,
                    self.epochs,
                    self.trainlosses.pp()
                )
            train_epoch_losses = self.trainlosses.get_agg_errors()
            self.do_callbacks(self.END_TRAIN)
            valid_epoch_losses = []
            if self.validlosses is not None and self.current_epoch % self._validinter == 0:
                self.do_callbacks(self.START_VALID)
                model = self.valid_model if self.valid_model is not None else self.model
                model.eval()
                self.validlosses.push_and_reset()
                totalvalidbats = len(self.validdataloader)
                for i, _batch in enumerate(self.validdataloader):
                    self.do_callbacks(self.START_VALID_BATCH)
                    _batch = [q.var(batch_e).cuda(self.usecuda).v for batch_e in _batch]
                    _tbi = self.valid_transform_batch_inp if self.valid_transform_batch_inp is not False else self.transform_batch_inp
                    _tbo = self.valid_transform_batch_out if self.valid_transform_batch_out is not False else self.transform_batch_out
                    _tbg = self.valid_transform_batch_gold if self.valid_transform_batch_gold is not False else self.transform_batch_gold
                    if _tbi is not None:
                        batch = _tbi(*_batch)
                    else:
                        batch = _batch
                    modelouts = model(*batch[:-1])
                    modelout2loss = modelouts
                    if _tbo is not None:
                        modelout2loss = _tbo(modelouts)
                    gold = batch[-1]
                    if _tbg is not None:
                        gold = _tbg(gold)
                    validlosses = self.validlosses(modelout2loss, gold,
                                                   inputs=batch[:-1], original_inputs=_batch)
                    tt.live("valid - Epoch {}/{} - [{}/{}]: {}"
                            .format(
                                self.current_epoch+1,
                                self.epochs,
                                i+1,
                                totalvalidbats,
                                self.validlosses.pp()
                                )
                            )
                    self.do_callbacks(self.END_VALID_BATCH)
                ttmsg += " -- valid: {}".format(self.validlosses.pp())
                valid_epoch_losses = self.validlosses.get_agg_errors()
                self.do_callbacks(self.END_VALID)
            tt.stoplive()
            tt.tock(ttmsg)
            self.do_callbacks(self.END_EPOCH)
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
        self.do_callbacks(self.START)
        self.trainlosses.reset()
        self.trainloop()
        self.do_callbacks(self.END)


class AutoHooker(object):
    def get_hooks(self):
        raise NotImplemented()


class _LRSchedulerAutoHooker(AutoHooker):
    def __init__(self, s, **kw):
        super(_LRSchedulerAutoHooker, self).__init__(**kw)
        self.s = s

    def get_hooks(self):
        return {train.START_EPOCH: self.on_start_epoch}

    def on_start_epoch(self, model, **kw):
        self.s.step(epoch=model.current_epoch)


class _ReduceLROnPlateauAutoHooker(AutoHooker):
    def __init__(self, s, critf, **kw):
        super(_ReduceLROnPlateauAutoHooker, self).__init__(**kw)
        self.s = s
        self.critf = critf

    def get_hooks(self):
        return {train.END_EPOCH: self.on_end_epoch}

    def on_end_epoch(self, model, **kw):
        self.s.step(self.critf(), epoch=model.current_epoch)


class ClipGradNorm(AutoHooker):
    def __init__(self, norm, **kw):
        super(ClipGradNorm, self).__init__(**kw)
        self._norm = norm

    def get_hooks(self):
        return {train.BEFORE_OPTIM_STEP: self.on_before_optim_step}

    def on_before_optim_step(self, trainer, **kw):
        model = trainer.model
        tgn0 = None
        if self._norm is not None:
            tgn0 = nn.utils.clip_grad_norm(model.parameters(), self._norm)
        if tgn0 is not None:
            tgn = tgn0
        else:
            tgn = 0
            for param in model.parameters():
                tgn += param.grad.pow(2).sum() if param.grad is not None else 0
            tgn = tgn.pow(1./2)
            tgn = tgn.data[0]
        return tgn


class TrainerChainer(AutoHooker):
    def __init__(self, trainer, **kw):
        super(TrainerChainer, self).__init__(**kw)
        self._trainer = trainer

    def get_hooks(self):
        return {train.END_BATCH: self.on_end_batch, train.START: self.on_start}

    def on_end_batch(self, *x, **kw):
        self._trainer.do_next_iter()

    def on_start(self, *x, **kw):
        self._trainer.reset()
        self._trainer.initialize()


class EarlyStopper(AutoHooker):
    def __init__(self, select=None, patience=0, delta=0.,
                 minepochs=5, lessisbetter=True, custom=None, **kw):
        super(EarlyStopper, self).__init__(**kw)
        if select is None:
            select = lambda trainlosses, validlosses, epochnumber: validlosses[0] if len(validlosses) > 0 else None
        self.monitor = select
        self.patience = patience
        self.delta = delta
        self.minepochs = minepochs
        self.history = []
        self.customf = custom
        self.lessisbetter = lessisbetter

    def get_hooks(self):
        return {train.END_EPOCH: self.on_end_epoch}

    def on_end_epoch(self, trainer, **kw):
        train_epoch_losses = trainer.trainlosses.get_agg_errors()
        valid_epoch_losses = trainer.validlosses.get_agg_errors() if trainer.validlosses is not None else []
        i = trainer.current_epoch

        monval = self.monitor(train_epoch_losses, valid_epoch_losses, i)
        if monval is None:
            return

        self.history.append(monval)

        # make early stopping decision based on history (including latest)
        stop = False
        stop |= self.customf(self.history)
        if len(self.history) >= 2 + self.patience and i > self.minepochs:
            last, check = self.history[-1], self.history[-2-self.patience]
            inbetween = self.history[-self.patience-1:-1]
            if self.lessisbetter:
                _stop = last > check + self.delta
                _cancel_stop = check > min(inbetween)
            else:
                _stop = last < check + self.delta
                _cancel_stop = check < max(inbetween)
            _stop &= not _cancel_stop
            stop |= stop
        trainer.stop_training |= stop


class HyperparamScheduler(AutoHooker):
    def __init__(self, hp, **kw):
        super(HyperparamScheduler, self).__init__(**kw)
        self._hp = hp

    def get_hooks(self):
        return {train.START_EPOCH: self.on_start_epoch}

    def on_start_epoch(self, trainer, **kw):
        pass


class EpochHyperparamScheduler(HyperparamScheduler):
    def __init__(self, hp, f, **kw):
        """ f takes epoch number and max epochs and hp and returns new value for hp """
        super(EpochHyperparamScheduler, self).__init__()
        self._f = f

    def get_hooks(self):
        return {train.START_EPOCH: self.on_start_epoch}

    def on_start_epoch(self, trainer, **kw):
        newval = self._f(trainer.current_epoch, maxepoch=trainer.epochs, hp=self._hp)
        self._hp.v = newval





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