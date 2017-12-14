import torch
from torch.autograd import Variable
import qelos as q
import scipy as sp
from scipy import optimize as spopt


EPS = 1e-6


class DataloaderIterator(object):
    def __init__(self, dataloader):
        super(DataloaderIterator, self).__init__()
        self.dataloader = dataloader
        self.dataloaderiter = iter(dataloader)
        self._epoch_count = 0

    def __next__(self):
        try:
            return next(self.dataloader)
        except StopIteration as e:
            """ done one epoch of data """
            self.dataloaderiter = iter(self.dataloader)
            return next(self)

    def reset(self):
        self._epoch_count = 0
        self.dataloaderiter = iter(self.dataloader)


class Cudable(object):
    def __init__(self, _usecuda=False):
        self.usecuda = _usecuda
        self.cudaargs = ([], {})

    def cuda(self, usecuda, *a, **kw):
        self.usecuda = usecuda
        self.cudaargs = (a, kw)
        return self


class InfNoise(Cudable):
    def __init__(self, batsize, noisedim=100, **kw):
        super(InfNoise, self).__init__(**kw)
        self.batsize = batsize
        self.noisedim = noisedim

    def __next__(self):
        noise = q.var(torch.zeros(self.batsize, self.noisedim)).cuda(self.usecuda).v
        noise.data.normal_(0, 1)
        return noise


class InfIterWithNoise(Cudable):
    """ default implementation that takes a dataloader
        and returns batches of data coupled with noise.
        Wraps data (and noise) in Variables """
    def __init__(self, dataloader, noisedim=100, **kw):
        super(InfIterWithNoise, self).__init__(**kw)
        self.dataloader = dataloader
        self.dataloaderiter = iter(self.dataloader)
        self._epoch_count = 0
        self.noisedim = noisedim

    def reset(self):
        self._epoch_count = 0
        self.dataloaderiter = iter(self.dataloader)

    def __next__(self):
        try:
            batch = next(self.dataloader)
            if not q.issequence(batch):
                batch = [batch]
            batsize = batch[0].size(0)
            batch = [q.var(batch_e).cuda(self.usecuda).v for batch_e in batch]
            noise = q.var(torch.zeros(batsize, self.noisedim)).cuda(self.usecuda).v
            noise.data.normal_(0, 1)
            return batch, [noise]
        except StopIteration as e:
            """ done one epoch of data """
            self.dataloaderiter = iter(self.dataloader)
            return next(self)


class GANTrainer(Cudable):
    opt_lr = 0.00001

    def __init__(self, discriminator, generator, **kw):
        super(GANTrainer, self).__init__(**kw)
        self.discriminator = discriminator
        self.generator = generator
        self.optD, self.optG = None, None
        self.dataD, self.dataG = None, None     # should be inf-iters
        self.valid_dataD, self.valid_dataG = None, None     # should be inf-iters
        self.valid_metrics = []
        self.valid_inter = 1
        # scheduling
        self.niterD, self.niterG = 1, 1
        self.niterD_burnin, self.niterD_burnin_interval = 0, 0
        self.niterD_burnin_number = 1
        self._phase = None

    def optimizers(self, optD, optG):
        self.optD = optD
        self.optG = optG
        return self

    def train_on(self, dataiterD, dataiterG):
        """
        :param dataiterD:   infinite data iterator to use for training discriminator
                            must provide data for both discriminator and generator
                            must return a pair of (possibly sequences) of data to feed to discriminator resp. generator
        :param dataiterG:   infinite data iterator to use for training generator
                            must only provide data for generator
        """
        self.dataD = dataiterD
        self.dataG = dataiterG
        return self

    def schedule(self, niterD=5, niterG=1, niterD_burnin=25, niterD_burnin_interval=500, niterD_burnin_number=100):
        self.niterD, self.niterG = niterD, niterG
        self.niterD_burnin, self.niterD_burnin_interval = niterD_burnin, niterD_burnin_interval
        self.niterD_burnin_number = niterD_burnin_number
        return self

    def valid_on(self, dataiterD, dataiterG, metrics, interval=10):
        self.valid_dataD = dataiterD
        self.valid_dataG = dataiterG
        self.valid_metrics = metrics
        self.valid_inter = interval
        return self

    def initialize(self):
        if self.usecuda:
            self.discriminator.cuda(*self.cudaargs[0], **self.cudaargs[1])
            self.generator.cuda(*self.cudaargs[0], **self.cudaargs[1])
        if self.optD is None:
            print("WARNING: discriminator optimizer was not set. Setting default (RMSProp, lr={})".format(self.opt_lr))
            self.optD = torch.optim.RMSprop(q.params_of(self.discriminator), lr=self.opt_lr)
        if self.optG is None:
            print("WARNING: generator optimizer was not set. Setting default (RMSProp, lr={})".format(self.opt_lr))
            self.optG = torch.optim.RMSprop(q.params_of(self.generator), lr=self.opt_lr)

    def reset(self):
        self.current_iter = 0

    def train(self, iters=1000):
        self.iters = iters
        self.reset()
        self.initialize()
        self.trainloop()

    def trainloop(self):
        for _iter in range(0, self.iters):
            self.current_iter = _iter
            self.train_discriminator()
            self.train_generator()
            if _iter % self.valid_inter == 0:
                self.run_valid()

    def _get_niterD(self, iteration, totaliterations):
        niterD = self.niterD
        if iteration < self.niterD_burnin or \
            (self.niterD_burnin_interval > 0
             and self.current_iter % self.niterD_burnin_interval == 0):
            niterD = self.niterD_burnin_number
        return niterD

    def _get_niterG(self, iteration, totaliterations):
        return self.niterG

    def train_discriminator(self):
        self._phase = "trainD"
        niterD = self._get_niterD(self.current_iter, self.iters)
        for j in range(niterD):
            self.discriminator.zero_grad()
            self._before_iterD()

            real, noise = next(self.dataD)

            if not q.issequence(real):
                real = [real]
            real = [q.var(real_e).cuda(self.usecuda).v
                    if not isinstance(real_e, Variable) else real_e
                    for real_e in real]
            if not q.issequence(noise):
                noise = [noise]
            noise = [q.var(noise_e).cuda(self.usecuda).v
                     if not isinstance(noise_e, Variable) else noise_e
                     for noise_e in noise]

            fake = self.generator(*noise)
            if not q.issequence(fake):
                fake = [fake]

            fake4D = self._get_fake4D(fake)

            scoreD_real = self.discriminator(*real)
            scoreD_fake = self.discriminator(*[fake_e.detach() for fake_e in fake4D])

            lossDadd = 0
            lossD = self._get_lossD(scoreD_real, scoreD_fake, real=real, fake=fake, noise=noise)
            if len(lossD) == 2:
                lossD, lossDadd = lossD

            costD = lossD + lossDadd
            costD.backward()
            self._log_trainD(costD=costD, lossD=lossD, lossDadd=lossDadd,
                             scoreD_real=scoreD_real, scoreD_fake=scoreD_fake,
                             real=real, fake=fake, noise=noise)
            self.optD.step()
            self._after_iterD()

    def _log_trainD(self, costD=None, lossD=None, lossDadd=None,
                          scoreD_real=None, scoreD_fake=None,
                          real=None, fake=None, noise=None):
        raise NotImplemented("use subclass")

    def train_generator(self):
        self._phase = "trainG"
        niterG = self._get_niterG(self.current_iter, self.iters)
        for j in range(niterG):
            self.generator.zero_grad()
            self._before_iterG()

            noise = next(self.dataG)
            if not q.issequence(noise):
                noise = [noise]
            noise = [q.var(noise_e).cuda(self.usecuda).v
                     if not isinstance(noise_e, Variable) else noise_e
                     for noise_e in noise]

            fake = self.generator(*noise)
            if not q.issequence(fake):
                fake = [fake]

            fake4D = self._get_fake4D(fake)

            scoreD_fake = self.discriminator(*fake4D)

            lossGadd = 0
            lossG = self._get_lossG(scoreD_fake, fake=fake, noise=noise)
            if len(lossG) == 2:
                lossG, lossGadd = lossG
            costG = lossG + lossGadd
            costG.backward()
            self.optG.step()
            self._log_trainG(costG=costG, lossG=lossG, lossGadd=lossGadd,
                             scoreD_fake=scoreD_fake, fake=fake, noise=noise)
            self._after_iterG()

    def _get_fake4D(self, fake):
        """ override this to only feed a selection of G's output to D """
        return fake

    def _log_trainG(self, costG=None, lossG=None, lossGadd=None,
                          scoreD_fake=None, fake=None, noise=None):
        raise NotImplemented("use subclass")

    def run_valid(self):
        raise NotImplemented("use subclass")

    def _get_lossD(self, score_real, score_fake, real=None, fake=None, noise=None):
        raise NotImplemented("use subclass")

    def _get_lossG(self, score_fake, fake=None, noise=None):
        raise NotImplemented("use subclass")

    # "hooks"
    def _before_iterD(self):    pass
    def _after_iterD(self):     pass
    def _before_iterG(self):    pass
    def _after_iterG(self):     pass


class OriginalGANTrainer(GANTrainer):
    def _get_lossD(self, score_real, score_fake, **kw):
        ret = -torch.log(score_real).mean() - torch.log(1 - score_fake).mean()
        return ret

    def _get_lossG(self, score_fake, **kw):
        ret = torch.log(1.0 - score_fake).mean()
        return ret


class WGANTrainer(GANTrainer):
    def __init__(self, d, g, noisedim=100, clamp_weights=(-0.01, +0.01), **kw):
        super(WGANTrainer, self).__init__(d, g, noisedim=noisedim, **kw)
        self.clamp_weights_rng = clamp_weights

    def _get_lossD(self, score_real, score_fake, **kw):
        ret = score_real.mean() - score_fake.mean()
        return ret

    def _get_lossG(self, score_fake, **kw):
        ret = -score_fake.mean()
        return ret

    def _before_iterD(self):
        for p in q.params_of(self.discriminator):
            p.data.clamp_(*self.clamp_weights_rng)


class IWGANTrainer(GANTrainer):
    def __init__(self, d, g, noisedim=100, grad_penalty=1, **kw):
        super(IWGANTrainer, self).__init__(d, g, noisedim=noisedim, **kw)
        self.grad_penalty_weight = grad_penalty

    def _get_lossD(self, score_real, score_fake, real=None, fake=None, noise=None):
        real, fake, noise = real[0], fake[0], noise[0]
        core = score_real.mean() - score_fake.mean()
        batsize = real.size(0)
        interp_alpha = real.data.new(batsize, 1)
        interp_alpha.uniform_(0, 1)
        interp_points = interp_alpha * real.data + (1 - interp_alpha) * fake.data
        interp_points = Variable(interp_points, requires_grad=True)
        scoreD_interp = self.discriminator(interp_points)
        scoreD_interp_grad = torch.autograd.grad(scoreD_interp.sum(),
                                                 interp_points,
                                                 create_graph=True)
        lip_grad_norm = scoreD_interp_grad.view(batsize, -1).norm(2, 1)
        lip_loss = self.grad_penalty_weight * ((lip_grad_norm - 1) ** 2).mean()
        return core, lip_loss

    def _get_lossG(self, score_fake, **kw):
        ret = -score_fake.mean()
        return ret


class DRAGANTrainer(GANTrainer):
    pass        # TODO


class ACGANTrainer(GANTrainer):
    pass        # TODO


class DiscoGANTrainer(GANTrainer):
    pass        # TODO


class InfoGANTrainer(GANTrainer):
    pass        # TODO


class fGANTrainer(GANTrainer):
    pass        # TODO


class LSGANTrainer(GANTrainer):
    pass        # TODO


# TODO: more GANS


class OldGANTrainer(object):
    def __init__(self,  mode="DRAGAN",      # WGAN, WGAN-GP, DRAGAN, DRAGAN-G, DRAGAN-LG, PAGAN
                        modeD="critic",  # disc or critic
                        one_sided=False,
                        penalty_weight=None,
                        perturb_both=False,
                        perturb_symmetric=False,
                        perturb_scale=1.,
                        clamp_weights_rng=(-0.01, 0.01),
                        optimizerD=None,
                        optimizerG=None,
                        logger=None,
                        data_iter=None,
                        valid_data_iter=None,
                        validation_metrics=[],  # fnr, emd
                        notgan=False,
                        validinter=1,
                        paganP=1):
        if one_sided:
            self.clip_fn = lambda x: x.clamp(min=0)
        else:
            self.clip_fn = lambda x: x
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.penalty_weight = penalty_weight
        self.perturb_both = perturb_both
        self.perturb_symmetric = perturb_symmetric
        self.perturb_scale = perturb_scale
        self.clamp_weights_rng = clamp_weights_rng
        self.noise_dim = 2
        self.mode = mode
        self.modeD = modeD
        if self.mode == "GAN":
            self.modeD = "disc"
        self.logger = logger
        self._dragan_lg_var = 0.6
        self.data_iter = data_iter
        self.valid_data_iter = valid_data_iter
        self.validinter = validinter
        self.validation_metrics = validation_metrics
        self.notgan = notgan
        # cache
        self.validdata = None
        # PAGAN
        self.paganP = paganP      # NOT THE P OF NORM!!
        self.pagandist = q.LNormDistance(2)

    def perturb(self, x):
        if self.mode == "DRAGAN":
            return self.perturb_dragan(x)
        elif self.mode == "DRAGAN-G":
            return self.perturb_dragan_g(x)
        elif self.mode == "DRAGAN-LG":
            return self.perturb_dragan_lg(x)

    def perturb_dragan(self, x):  # Variable (batsize, dim)
        perturbation = x.data.new(x.size())
        if not self.perturb_symmetric:
            perturbation.uniform_(0, to=1)
        else:
            perturbation.uniform_(-1, to=1)
        perturbation *= self.perturb_scale * x.data.std()
        interp_alpha = perturbation.new(perturbation.size(0), 1)
        interp_alpha.uniform_(0, to=1)
        perturbation = interp_alpha * perturbation
        perturbed = x.data + perturbation
        return Variable(perturbed, requires_grad=True)

    def perturb_dragan_g(self, x):
        perturbation = x.data.new(x.size())
        perturbation.normal_()
        perturbation *= 0.25 * self.perturb_scale * x.data.std()
        ret = x.data + perturbation
        return Variable(ret, requires_grad=True)

    def perturb_dragan_lg(self, x):
        perturbation = x.data.new(x.size())
        perturbation.log_normal_(0, self._dragan_lg_var)
        interp_alpha = perturbation.new(perturbation.size())
        interp_alpha.uniform_(-1, to=1)
        interp_alpha.sign_()
        perturbation *= interp_alpha * 0.25 * self.perturb_scale * x.data.std()
        ret = x.data + perturbation
        return Variable(ret, requires_grad=True)

    def train(self, netD, netG, niter=0, niterD=10, batsizeG=100,
              data_gen=None, valid_data_gen=None, cuda=False):
        print("status: pagan added")
        data_gen = data_gen if data_gen is not None else self.data_iter
        valid_data_gen = valid_data_gen if valid_data_gen is not None else self.valid_data_iter
        if cuda:
            netD.cuda()
            netG.cuda()

        valid_EMD = 0.
        valid_fake2real = 0.    # mean of distances from each fake to closest real
        valid_real2fake = 0.    # mean of distances from each real to closest fake
        valid_fakeandreal = 0.  # harmonic mean of above

        vnoise = q.var(torch.zeros(batsizeG, self.noise_dim)).cuda(cuda).v

        for _iter in range(niter):
            ##########  Update D net ##########
            lip_loss = None
            for p in netD.parameters():
                p.requires_grad = True

            if _iter < 25 or _iter % 500 == 0:
                _niterD = 100
            else:
                _niterD = niterD
            for j in range(_niterD):
                netD.zero_grad()
                if self.mode == "WGAN":
                    for p in netD.parameters():
                        p.data.clamp_(*self.clamp_weights_rng)
                data = next(data_gen)
                real = q.var(data).cuda(cuda).v
                num_examples = data.size(0)
                batsize = num_examples
                vnoise = Variable(real.data.new(num_examples, self.noise_dim))
                vnoise.data.normal_(0, 1)
                fake = netG(vnoise)
                scoreD_real_vec = netD(real)        # (batsize,)
                #scoreD_real = scoreD_real_vec.mean()
                # scoreD_real.backward(one, retain_graph=(lc> 0))
                scoreD_fake_vec = netD(fake.detach())        # TODO: netD(fake.detach())  ?
                #scoreD_fake = scoreD_fake_vec.mean()
                # scoreD_fake.backward(mone, retain_graph=(lc > 0))
                if self.mode == "GAN":
                    errD = - torch.log(scoreD_real_vec).mean() - torch.log(1 - scoreD_fake_vec).mean()
                    grad_points = None
                if self.mode == "WGAN":
                    errD = scoreD_fake_vec.mean() - scoreD_real_vec.mean()
                    grad_points = None
                elif self.mode == "PAGAN":
                    errD = scoreD_fake_vec.mean() - scoreD_real_vec.mean()
                    lip_loss = (scoreD_real_vec - scoreD_fake_vec).squeeze()
                    lip_loss_p = self.pagandist(real, fake) ** self.paganP
                    lip_loss = lip_loss / lip_loss_p.clamp(min=EPS)
                    lip_loss = self.penalty_weight * (self.clip_fn(lip_loss - 1)).mean()
                    errD = errD + lip_loss
                    grad_points = None
                elif self.mode == "WGAN-GP":
                    errD = scoreD_fake_vec.mean() - scoreD_real_vec.mean()
                    interp_alpha = real.data.new(num_examples, 1)
                    interp_alpha.uniform_(0, 1)
                    interp_points = interp_alpha * real.data + (1 - interp_alpha) * fake.data
                    grad_points = interp_points.clone()
                    interp_points = Variable(interp_points, requires_grad=True)
                    errD_interp_vec = netD(interp_points)
                    errD_gradient, = torch.autograd.grad(errD_interp_vec.sum(),
                                                         interp_points,
                                                         create_graph=True)
                    lip_grad_norm = errD_gradient.view(batsize, -1).norm(2, dim=1)
                    assert(lip_grad_norm.size() == (batsize,))
                    lip_loss = self.penalty_weight * (self.clip_fn(lip_grad_norm - 1) ** 2).mean()
                    errD = errD + lip_loss
                elif self.mode == "DRAGAN" or self.mode == "DRAGAN-G" or self.mode == "DRAGAN-LG":
                    if self.modeD == "disc":
                        errD = - torch.log(scoreD_real_vec).mean() - torch.log(1 - scoreD_fake_vec).mean()
                    elif self.modeD == "critic":
                        errD = scoreD_fake_vec.mean() - scoreD_real_vec.mean()

                    real_perturbed = self.perturb(real)
                    grad_points = real_perturbed.data
                    errD_real_perturbed_vec = netD(real_perturbed)
                    errD_gradient, = torch.autograd.grad(
                        errD_real_perturbed_vec.sum(), real_perturbed, create_graph=True)
                    lip_grad_norm = errD_gradient.view(batsize, -1).norm(2, dim=1)
                    assert(lip_grad_norm.size() == (batsize,))
                    lip_loss = self.penalty_weight * (self.clip_fn(lip_grad_norm - 1) ** 2).mean()
                    if self.perturb_both:
                        fake_perturbed = self.perturb(fake)
                        errD_fake_perturbed_vec = netD(fake_perturbed)
                        errD_fake_gradient, = torch.autograd.grad(
                            errD_fake_perturbed_vec.sum(), fake_perturbed, create_graph=True)
                        lip_fake_grad_norm = errD_fake_gradient.view(batsize, -1).norm(2, dim=1)
                        assert(lip_fake_grad_norm.size() == (batsize,))
                        lip_fake_loss = self.penalty_weight * (self.clip_fn(lip_fake_grad_norm - 1) ** 2).mean()
                        lip_loss = lip_loss * 0.5 + lip_fake_loss * 0.5
                    errD = errD + lip_loss
                    #errD = lip_loss
                errD.backward()
                self.optimizerD.step()

            ##########  Update G net ##########
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            vnoise.data.normal_(0, 1)
            fake = netG(vnoise)

            errG = netD(fake)
            if self.modeD == "critic":
                errG = -errG.mean()
            elif self.modeD == "disc":
                errG = torch.log(1.0 - errG).mean()

            errG.backward()
            self.optimizerG.step()

            ######### Validate G net ##########
            if valid_data_gen is not None:
                if _iter % self.validinter == 0 or self.validinter == 1:
                    validdata = next(valid_data_gen)
                    validreal = q.var(validdata, volatile=True).cuda(cuda).v
                    vnoise = q.var(validreal.data.new(validreal.size(0), self.noise_dim),
                                   volatile=True).v
                    vnoise.data.normal_(0, 1)
                    validfake = netG(vnoise)
                    # compute distance matrix
                    distmat = q.LNormDistance(2)(
                        validreal.unsqueeze(0),
                        validfake.unsqueeze(0)).squeeze(0)
                    if "emd" in self.validation_metrics:
                        npdistmat = distmat.cpu().data.numpy()
                        ass_x, ass_y = spopt.linear_sum_assignment(npdistmat)
                        valid_EMD = npdistmat[ass_x, ass_y].mean()

                    if "fnr" in self.validation_metrics:
                        #real2fake and fake2real
                        valid_fake2real, _ = torch.min(distmat, 0)
                        valid_real2fake, _ = torch.min(distmat, 1)
                        valid_fake2real = valid_fake2real.mean()
                        valid_real2fake = valid_real2fake.mean()
                        valid_fakeandreal = 2 * valid_fake2real * valid_real2fake / (valid_fake2real + valid_real2fake).clamp(min=1e-6)
                        valid_fakeandreal = valid_fakeandreal.data[0]
                        valid_fake2real = valid_fake2real.data[0]
                        valid_real2fake = valid_real2fake.data[0]

            if self.logger is not None:
                self.logger.log(_iter=_iter, niter=niter,
                                real=real.data, fake=fake.data, grad_points=grad_points,
                                errD=errD.data[0], errG=errG.data[0],
                                scoreD_real=scoreD_real_vec.mean().data[0],
                                scoreD_fake=scoreD_fake_vec.mean().data[0],
                                lip_loss=lip_loss.data[0] if lip_loss is not None else -999.,
                                valid_EMD=valid_EMD, valid_fake2real=valid_fake2real,
                                valid_real2fake=valid_real2fake, valid_fakeandreal=valid_fakeandreal,
                                when="after_G")

