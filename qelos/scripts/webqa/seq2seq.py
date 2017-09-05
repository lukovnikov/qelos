import qelos as q
from qelos.scripts.webqa.load import load_all
import torch
from torch import nn


def make_encoder(src_emb, embdim=100, dim=100, **kw):
    """ make encoder
    # concatenating bypass encoder:
    #       embedding  --> top GRU
    #                  --> 1st BiGRU
    #       1st BiGRU  --> top GRU
    #                  --> 2nd BiGRU
    #       2nd BiGRU  --> top GRU
    """
    encoder = q.RecurrentStack(
        src_emb,        # embs, masks
        q.argsave.spec(emb=0, mask=1),
        q.argmap.spec(0, mask=["mask"]),
        q.BidirGRULayer(embdim, dim),
        q.argsave.spec(bypass=0),
        q.argmap.spec(0, mask=["mask"]),
        q.BidirGRULayer(dim * 2, dim),
        q.argmap.spec(0, ["bypass"], ["emb"]),
        q.Lambda(lambda x, y, z: torch.cat([x, y, z], 2)),
        q.argmap.spec(0, mask=["mask"]),
        q.GRULayer(dim * 4 + embdim, dim).return_final(True),
        q.argmap.spec(1, ["mask"], 0),
    )   # returns (all_states, mask, final_state)
    return encoder


def make_decoder(emb, lin, embdim=100, dim=100, **kw):
    """ makes decoder
    # attention cell decoder that accepts VNT !!!
    """
    pass        # TODO


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, srcseq, tgtseq):
        enc = self.encoder(srcseq)
        encstates = self.encoder.get_states(srcseq.size(0))
        self.decoder.set_init_states(encstates[-1], encstates[-1])
        dec = self.decoder(tgtseq, enc)
        return dec


def run(lr=0.1,
        gradnorm=2.,
        epochs=100,
        wreg=1e-6,
        glovedim=50,
        encdim=100,
        decdim=100,
        merge_mode="cat",        # "cat" or "sum"
        rel_which="urlwords",     # "urlwords ... ..."
        rel_embdim=-1,
        batsize=128,
        cuda=False,
        gpu=1,
        ):
    if cuda:
        torch.cuda.set_device(gpu)

    tt = q.ticktock("script")
    # region I. data
    # load data and reps
    tt.tick("loading data and rep")
    rel_which = tuple(rel_which.split())
    rel_embdim = None if rel_embdim == -1 else rel_embdim
    (question_sm, query_sm, vnt_mat, tx_sep, qids), (src_emb, tgt_emb, tgt_lin) \
        = load_all(glovedim=glovedim, merge_mode=merge_mode,
                   rel_which=rel_which, rel_embdim=rel_embdim)
    tt.tock("loaded data and rep")
    tt.tick("making data loaders")
    # train/valid/test split:
    train_questions, test_questions = question_sm.matrix[:tx_sep], question_sm[tx_sep:]
    train_queries, test_queries = query_sm.matrix[:tx_sep], query_sm[tx_sep:]
    train_vnt, test_vnt = vnt_mat[:tx_sep], vnt_mat[tx_sep:]

    validportion = .2
    tv_sep = int(len(train_questions) * (1 - validportion))
    train_questions, valid_questions = train_questions[:tv_sep], train_questions[tv_sep:]
    train_queries, valid_queries = train_queries[:tv_sep], train_queries[tv_sep:]
    train_vnt, valid_vnt = train_vnt[:tv_sep], train_vnt[tv_sep:]

    # make data loaders
    train_dataloader = q.dataload(train_questions, train_vnt, train_queries, batch_size=batsize)
    valid_dataloader = q.dataload(valid_questions, valid_vnt, valid_queries, batch_size=batsize)
    test_dataloader = q.dataload(test_questions, test_vnt, test_queries, batch_size=batsize)
    tt.tock("made data loaders")
    # endregion

    # make main model
    src_emb_dim = glovedim
    tgt_emb_dim = glovedim
    encoder = make_encoder(src_emb, embdim=src_emb_dim, dim=encdim)
    decoder = make_decoder(tgt_emb, tgt_lin, embdim=tgt_emb_dim, dim=decdim)
    m = Model(encoder, decoder)

    # training settings
    losses = q.lossarray(q.SeqNLLLoss(), q.SeqAccuracy(), q.SeqElemAccuracy())
    validlosses = q.lossarray(q.SeqNLLLoss(), q.SeqAccuracy(), q.SeqElemAccuracy())

    optimizer = torch.optim.Adadelta(m.parameters(), lr=lr, weight_decay=wreg)

    # train
    q.train(m).cuda(cuda).train_on(train_dataloader, losses)\
        .set_batch_transformer(lambda a, b, c: (a, b, c[:, :-1], c[:, 1:]))\
        .valid_on(valid_dataloader, validlosses)\
        .optimizer(optimizer).clip_grad_norm(gradnorm)\
        .train(epochs)

    # test
    # TODO


if __name__ == "__main__":
    q.argprun(run)