import qelos as q
import os
import json
import warnings
import pandas as pd
import glob


OPT_PREFIX = "zexp"
OPT_SETTINGS_NAME = "settings.json"
OPT_LOG_NAME = "log{}.lines"


def get_default_log_path(prefix):
    prefix = prefix if prefix is not None else OPT_PREFIX
    i = 0
    found = False
    while not found:
        canpath = "{}{}".format(prefix, i)
        if os.path.exists(canpath):
            i += 1
            continue
        else:
            found = True
    return canpath


def get_train_dump_path(p, logname=None):
    found = False
    if logname is not None:
        canpath = p + "/" + logname
        found = True
    i = 0
    while not found:
        canpath = p + "/" + OPT_LOG_NAME.format(i if i > 0 else "")
        if os.path.exists(canpath):
            i += 1
            continue
        else:
            found = True
    return canpath


class Logger(q.AutoHooker):
    def __init__(self, p=None, prefix=None, **kw):
        super(Logger, self).__init__()
        assert(p is None or prefix is None)
        self.p = p if p is not None else get_default_log_path(prefix)
        if os.path.exists(self.p):
            raise q.SumTingWongException("path '{}' already exists".format(p))
        else:
            os.makedirs(self.p)
        self._current_train_file = None
        self._current_numbers = []

    def save_settings(self, **kw):
        p = self.p + "/" + OPT_SETTINGS_NAME
        with open(p, "w") as f:
            json.dump(kw, f, sort_keys=True)

    def load_settings(self):
        p = self.p + "/" + OPT_SETTINGS_NAME
        with open(p) as f:
            r = json.load(f)
        return r

    def update_settings(self, **kw):
        settings = self.load_settings()
        settings.update(kw)
        self.save_settings(**settings)

    def get_hooks(self):
        return {q.train.START: self.on_start,
                q.train.END_TRAIN: self.on_end_train,
                q.train.END_VALID: self.on_end_valid,
                q.train.END_EPOCH: self.on_end_epoch,
                q.train.END: self.on_end}

    def on_start(self, trainer, **kw):
        self.start_train_logging(trainer=trainer, **kw)

    def start_train_logging(self, names=None, trainer=None, logname=None, overwrite=False, **kw):
        p = get_train_dump_path(self.p, logname)
        # make writer
        if os.path.exists(p):
            if not overwrite:
                raise q.SumTingWongException("file already exists")
            else:
                warnings.warn("training log file already exists. overwriting {}".format(p))
        self._current_train_file = open(p, "w+")
        assert(names is None or trainer is None)
        names = ["Epoch"]
        if trainer is not None:
            names += trainer.trainlosses.get_names()
            if trainer.validlosses is not None:
                names += trainer.validlosses.get_names()
        line = "\t".join(names) + "\n"
        self._current_train_file.write(line)
        self._current_train_file.flush()

    def on_end(self, trainer, **kw):
        self.stop_train_logging(trainer, **kw)

    def stop_train_logging(self, trainer, **kw):
        self._current_train_file.close()

    def on_end_train(self, trainer, **kw):
        self._current_numbers = trainer.trainlosses.get_agg_errors()

    def on_end_valid(self, trainer, **kw):
        self._current_numbers += trainer.validlosses.get_agg_errors()

    def on_end_epoch(self, trainer, **kw):
        self.flush_trainer_loss_history(trainer, **kw)

    def flush_trainer_loss_history(self, trainer, **kw):
        cur_epoch = trainer.current_epoch * 1.
        numbers = [cur_epoch] + self._current_numbers
        line = "\t".join(["{:f}".format(n) for n in numbers]) + "\n"
        self._current_train_file.write(line)
        self._current_train_file.flush()

    @classmethod
    def load_path(cls, p):
        """ loads one log (from dir specified by p) """
        settings = {}
        settingsp = p + "/" + OPT_SETTINGS_NAME
        if os.path.exists(settingsp):
            with open(p + "/" + OPT_SETTINGS_NAME) as f:
                settings = json.load(f)
        datap = p + "/" + OPT_LOG_NAME.format("")
        data = pd.DataFrame.from_csv(datap, sep="\t")
        return ExpLog(settings, data)

    @classmethod
    def load_multiple(cls, expr):
        """ expr is unix path expression (check glob specs) """
        matched_paths = glob.glob(expr)
        for path in matched_paths:
            yield cls.load_path(path)


class ExpLog(object):
    def __init__(self, settings, data, **kw):
        super(ExpLog, self).__init__(**kw)
        self.settings = settings
        self.data = data



