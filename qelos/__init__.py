from qelos.train import lossarray, train, test, eval, TensorDataset, LossWithAgg, \
    aux_train, loss_input_transform, AutoHooker, HyperparamScheduler

from qelos.rnn import GRUCell, LSTMCell, SRUCell, RNU, RecStack, RNNLayer, \
    BiRNNLayer, GRULayer, LSTMLayer, RecurrentStack, BidirGRULayer, \
    BidirLSTMLayer, Recurrent, Reccable, PositionwiseForward, \
    TimesharedDropout, CatLSTMCell, ReccableLambda, RecurrentLambda, RecDropout

from qelos.loss import SeqNLLLoss, SeqAccuracy, SeqElemAccuracy, \
    RankingLoss, SeqRankingLoss, CrossEntropyLoss, \
    SeqCrossEntropyLoss, PairRankingLoss, Accuracy, DiscreteLoss, Loss, MacroBLEU

from qelos.seq import Decoder, DecoderCell, ContextDecoderCell, \
    AttentionDecoderCell, Attention, ContextDecoder, AttentionDecoder, \
    HierarchicalAttentionDecoderCell, ModularDecoderCell, \
    DecoderTop, ContextDecoderTop, StaticContextDecoderTop, AttentionContextDecoderTop, \
    DecoderCore, FreeRunner, TeacherForcer, DecoderRunner

from qelos.basic import Softmax, LogSoftmax, BilinearDistance, \
    CosineDistance, DotDistance, Forward, ForwardDistance, \
    Distance, Lambda, Stack, TrilinearDistance, LNormDistance, \
    SeqBatchNorm1d, CReLU, Identity, argmap, argsave, persist_kwargs, \
    LayerNormalization, wire, Dropout

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

from qelos.qutils import name2fn, var, val, v, seq_pack, seq_unpack, dataload, \
    params_of, batchablesparse2densemask, rec_clone, intercat, hyperparam, \
    add_tag, remove_tag, filter_by_tag, get_tags, add_qelos_key, has_qelos_key, get_qelos_key, \
    remove_qelos_key, gradmult, remove_gradmult, set_lr, remove_lr, set_l2, remove_l2, \
    paramgroups_of

# from tfun import TFModule

from qelos.reg import l1, l2