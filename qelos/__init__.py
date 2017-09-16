from qelos.train import lossarray, train, TensorDataset, LossWithAgg
from qelos.rnn import GRUCell, LSTMCell, SRUCell, RNU, RecStack, RNNLayer, BiRNNLayer, GRULayer, LSTMLayer, RecurrentStack, BidirGRULayer, BidirLSTMLayer, Recurrent, Reccable, PositionwiseForward
from qelos.loss import SeqNLLLoss, SeqAccuracy, SeqElemAccuracy
from qelos.seq import Decoder, DecoderCell, ContextDecoderCell, AttentionDecoderCell, Attention, ContextDecoder, AttentionDecoder
from qelos.basic import Softmax, LogSoftmax, BilinearDistance, CosineDistance, DotDistance, Forward, ForwardDistance, \
    Distance, Lambda, Stack, TrilinearDistance, LNormDistance, SeqBatchNorm1d, CReLU, Identity, argmap, argsave, LayerNormalization
from qelos.containers import ModuleList
from qelos.word import WordEmb, PretrainedWordEmb, ComputedWordEmb, WordLinout, PretrainedWordLinout, ComputedWordLinout
from qelos.gan import GANTrainer
from qelos.exceptions import SumTingWongException, HoLeePhukException, BaDumTssException
from IPython import embed
from qelos.aiayn import MultiHeadAttention as MultiHeadAttention, Encoder as AYNEncoder, Decoder as AYNDecoder, Transformer as AYNTransformer


from qelos.util import ticktock, argprun, isnumber, issequence, iscollection, \
    iscallable, isstring, isfunction, StringMatrix, tokenize, dtoo, emit, get_emitted, \
    wordids2string, wordmat2wordchartensor, slicer_from_flatcharseq, split
from qelos.qutils import name2fn, var, val, seq_pack, seq_unpack, dataload, params_of
