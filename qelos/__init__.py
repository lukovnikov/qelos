from qelos.train import lossarray, train, TensorDataset
from qelos.rnn import GRUCell, LSTMCell, RNU, RecStack, RNNLayer, BiRNNLayer, GRULayer, LSTMLayer, RecurrentStack
from qelos.loss import SeqNLLLoss
from qelos.seq import Decoder, DecoderCell, ContextDecoderCell, AttentionDecoderCell, Attention, ContextDecoder, AttentionDecoder
from qelos.basic import Softmax, LogSoftmax, BilinearDistance, CosineDistance, DotDistance, Forward, ForwardDistance, \
    Distance, Lambda, Stack, TrilinearDistance, LNormDistance, SeqBatchNorm1d, CReLU, Identity
from qelos.containers import ModuleList
from qelos.util import ticktock, argprun, isnumber, issequence, iscollection, \
    iscallable, isstring, isfunction, StringMatrix, tokenize
from qelos.qutils import name2fn, var, val
from qelos.word import WordEmb, PretrainedWordEmb, ComputedWordEmb, WordLinout, PretrainedWordLinout, ComputedWordLinout
from qelos.gan import GANTrainer
from qelos.exceptions import SumTingWongException
from IPython import embed
