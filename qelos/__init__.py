from qelos.train import lossarray, train, test, eval, TensorDataset, LossWithAgg, aux_train

from qelos.rnn import GRUCell, LSTMCell, SRUCell, RNU, RecStack, RNNLayer, \
    BiRNNLayer, GRULayer, LSTMLayer, RecurrentStack, BidirGRULayer, \
    BidirLSTMLayer, Recurrent, Reccable, PositionwiseForward, \
    TimesharedDropout, CatLSTMCell

from qelos.loss import SeqNLLLoss, SeqAccuracy, SeqElemAccuracy, \
    RankingLoss, SeqRankingLoss, CrossEntropyLoss, \
    SeqCrossEntropyLoss, PairRankingLoss

from qelos.seq import Decoder, DecoderCell, ContextDecoderCell, \
    AttentionDecoderCell, Attention, ContextDecoder, AttentionDecoder, \
    HierarchicalAttentionDecoderCell, ModularDecoderCell, \
    DecoderTop, ContextDecoderTop, StaticContextDecoderTop, AttentionContextDecoderTop, \
    DecoderCore, FreeRunner, TeacherForcer, DecoderRunner

from qelos.basic import Softmax, LogSoftmax, BilinearDistance, \
    CosineDistance, DotDistance, Forward, ForwardDistance, \
    Distance, Lambda, Stack, TrilinearDistance, LNormDistance, \
    SeqBatchNorm1d, CReLU, Identity, argmap, argsave, persist_kwargs, \
    LayerNormalization, wire

from qelos.containers import ModuleList

from qelos.word import WordEmb, PretrainedWordEmb, ComputedWordEmb, \
    WordLinout, PretrainedWordLinout, ComputedWordLinout, ZeroWordEmb, \
    ZeroWordLinout

from qelos.gan import GANTrainer

from qelos.exceptions import SumTingWongException, HoLeePhukException, \
    BaDumTssException, LonLeeException, WiNoDoException

from IPython import embed

from qelos.aiayn import MultiHeadAttention as MultiHeadAttention, \
    Encoder as AYNEncoder, Decoder as AYNDecoder, \
    Transformer as AYNTransformer, AddSinPositionVectors, \
    EncoderLayer as AynEncoderLayer

from qelos.cnn import SeqConv
from qelos.furnn import MemGRUCell, TwoStackCell, DynamicOracleRunner


from qelos.util import ticktock, argprun, isnumber, issequence, iscollection, \
    iscallable, isstring, isfunction, StringMatrix, tokenize, dtoo, emit, get_emitted, \
    wordids2string, wordmat2wordchartensor, slicer_from_flatcharseq, split, log, kw2dict, \
    save_sparse_tensor, load_sparse_tensor, makeiter, getkw

from qelos.qutils import name2fn, var, val, seq_pack, seq_unpack, dataload, \
    params_of, batchablesparse2densemask, rec_clone, intercat

from tfun import TFModule