
###################################################################
# adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch by Yu-Hsiang Huang
###################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import qelos as q
from qelos.rnn import PositionwiseForward

MASKID = 0


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(MASKID).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


#### BASIC ##########################
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention. Old, don't use '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax()

    def forward(self, q, k, v, attn_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


''' Define the sublayers in encoder/decoder layer '''


class OldOriginalMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(OldOriginalMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = q.Attention().dot_gen().scale(d_model ** 0.5).dropout(dropout)    #ScaledDotProductAttention(d_model)
        self.layer_norm = q.LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, x, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = x

        mb_size, len_q, d_model = x.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = x.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        q.emit("mymha", {"q_s": q_s, "k_s": k_s, "v_s": v_s})

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.repeat(n_head, 1)
        else:
            attn_mask = attn_mask.repeat(n_head, 1, 1)
        attns = self.attention.attgen(k_s, q_s, mask=attn_mask)
        outputs = self.attention.attcon(v_s, attns)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns
        # return outputs + residual, attns


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(d_model, d_k * n_head))  # changes xavier init but probably no difference in the end
        self.w_ks = nn.Parameter(torch.FloatTensor(d_model, d_k * n_head))
        self.w_vs = nn.Parameter(torch.FloatTensor(d_model, d_v * n_head))

        self.attention = q.Attention().dot_gen().scale(d_model ** 0.5).dropout(dropout)    #ScaledDotProductAttention(d_model)
        self.layer_norm = q.LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, x, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = x

        mb_size, len_q, dim_q = x.size()
        mb_size, len_k, dim_k = k.size()
        mb_size, len_v, dim_v = v.size()

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.mm(x.view(-1, dim_q), self.w_qs).view(mb_size, len_q, n_head, d_k) \
            .permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.mm(k.view(-1, dim_k), self.w_ks).view(mb_size, len_k, n_head, d_k) \
            .permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.mm(v.view(-1, dim_v), self.w_vs).view(mb_size, len_v, n_head, d_v) \
            .permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v


        q.emit("mymha", {"q_s": q_s, "k_s": k_s, "v_s": v_s})

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.repeat(n_head, 1)
        else:
            attn_mask = attn_mask.repeat(n_head, 1, 1)
        attns = self.attention.attgen(k_s, q_s, mask=attn_mask)
        outputs = self.attention.attcon(v_s, attns)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns
        # return outputs + residual, attns


class OriginalMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(OriginalMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model, attn_dropout=dropout)
        self.layer_norm = q.LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, x, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = x

        mb_size, len_q, d_model = x.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # HACK
        if attn_mask.dim() == 2:        # original impl assumes dim 3 ???
            attn_mask = attn_mask.unsqueeze(0).repeat(mb_size, 1, 1)

        # treat as a (n_head) size batch
        q_s = x.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        q.emit("mha", {"q_s": q_s, "k_s": k_s, "v_s": v_s})

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns
        # return outputs + residual, attns


#### ENCODING #######################
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, src_emb, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_pos_vec=212, d_model=512, d_inner_hid=1024, dropout=0.1, cat_pos_enc=True):

        super(Encoder, self).__init__()
        self.cat_pos_enc = cat_pos_enc

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_pos_vec, padding_idx=MASKID)
        self.position_enc.weight.data = position_encoding_init(n_position, d_pos_vec)
        self.position_enc.weight.requires_grad = False

        self.src_word_emb = src_emb     #nn.Embedding(n_src_vocab, d_word_vec, padding_idx=MASKID)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos=None):
        # Word embedding look up
        enc_slf_attn_mask = None
        enc_input = self.src_word_emb(src_seq)
        if isinstance(enc_input, tuple) and len(enc_input) == 2:
            enc_input, enc_slf_attn_mask = enc_input

        if src_pos is None:
            src_pos = torch.arange(0, src_seq.size(1))\
                .unsqueeze(0).repeat(src_seq.size(0), 1).long()
            src_pos = q.var(src_pos).v

        # Position Encoding addition
        pos_input = self.position_enc(src_pos)
        if not self.cat_pos_enc:
            enc_input = enc_input + pos_input           # does the paper add position encodings? --> yes
        else:
            enc_input = torch.cat([enc_input, pos_input], 2)

        enc_outputs, enc_slf_attns = [], []

        enc_output = enc_input
        # enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            enc_outputs += [enc_output]
            enc_slf_attns += [enc_slf_attn]

        return enc_output       # enc_outputs, enc_slf_attns


#### DECODING ######################
class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):       # This decoder is teacher forced, non-reccable TODO: make proper decoder
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, tgt_emb, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_pos_vec=212, d_model=512, d_inner_hid=1024, dropout=0.1, cat_pos_enc=True):

        super(Decoder, self).__init__()
        self.cat_pos_enc = cat_pos_enc
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_pos_vec, padding_idx=MASKID)
        self.position_enc.weight.data = position_encoding_init(n_position, d_pos_vec)
        self.position_enc.weight.requires_grad = False

        self.tgt_word_emb = tgt_emb
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, src_enc, src_mask=None, tgt_pos=None):
        # Word embedding look up
        dec_input, dec_mask = self.tgt_word_emb(tgt_seq)
        if dec_mask is not None:
            dec_mask = dec_mask.unsqueeze(1).repeat(1, dec_input.size(1), 1)
        else:
            dec_mask = q.var(np.ones((tgt_seq.size(0), tgt_seq.size(1), tgt_seq.size(1)))).v

        if tgt_pos is None:
            tgt_pos = torch.arange(0, tgt_seq.size(1)) \
                .unsqueeze(0).repeat(tgt_seq.size(0), 1).long()
            tgt_pos = q.var(tgt_pos).v

        # Position Encoding addition
        pos_input = self.position_enc(tgt_pos)
        if not self.cat_pos_enc:
            dec_input = dec_input + pos_input
        else:
            dec_input = torch.cat([dec_input, pos_input], 2)

        dec_outputs, dec_slf_attns, dec_enc_attns = [], [], []

        # Decode
        # dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = -1*get_attn_subsequent_mask(tgt_seq)+1
        dec_slf_attn_mask = q.var(dec_mask.data.byte() * dec_slf_attn_sub_mask).v

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, src_enc, slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=src_mask)

            dec_outputs += [dec_output]
            dec_slf_attns += [dec_slf_attn]
            dec_enc_attns += [dec_enc_attn]

        return dec_output       #dec_outputs, dec_slf_attns, dec_enc_attns


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, src_emb, tgt_emb, tgt_lin, n_max_seq, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1, proj_share_weight=True):

        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_emb, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_pos_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(
            tgt_emb, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_pos_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.tgt_word_proj = tgt_lin
        self.dropout = nn.Dropout(dropout)

    # def get_trainable_parameters(self):
    #     ''' Avoid updating the position encoding '''
    #     enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
    #     dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
    #     freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
    #     return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_output = self.encoder(src_seq, src_pos)
        dec_output = self.decoder(
            tgt_seq, tgt_pos, src_seq, enc_output)

        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))


class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, indim, outdim, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(indim, outdim, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)



#### BOTTLES #######################
class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass


class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0]*size[1]))
        return out.view(-1, size[0], size[1])


class BottleLayerNormalization(BatchBottle, q.LayerNormalization):
    ''' Perform the reshape routine before and after a layer normalization'''
    pass


