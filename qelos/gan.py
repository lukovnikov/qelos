import torch
from torch.autograd import Variable
import qelos as q
import scipy as sp
from scipy import optimize as spopt


EPS = 1e-6


class GANTrainer(object):
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
                        paganP=1,
                        noise_dim=2):
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
        self.noise_dim = noise_dim
        self.mode = mode
        self.modeD = modeD
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
        # dynamic hyperparams
        self.dynamichyperparams = []

    def add_dyn_hyperparams(self, dynhparam):
        self.dynamichyperparams.append(dynhparam)
        return self

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
        interp_alpha = perturbation.new(perturbation.size(0),
                                        *([1]*(perturbation.dim()-1)))
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

    def train(self, netD, netG, niter=0, niterD=10, niterG=1, batsizeG=100,
              data_gen=None, valid_data_gen=None, cuda=False,
              netR=None, sample_real_for_gen=False, feed_real_to_gen=False):
        """
        :param netD:    nn.Module for discriminator function,
                        or, tuple of two nn.Modules,
                        first one used as discriminator during discriminator training,
                        second one used as discriminator during generator training
        :param netG:    nn.Module for generator function,
                        or, tuple of two nn.Modules,
                        first one used as generator during discriminator training,
                        second one used as generator during generator training
        :param niter:   total number of iterations to train (equals number of generator updates)
        :param niterD:  number of discriminator updates per generator update
        :param batsizeG: generator batch size
        :param data_gen: data generator of real examples (must have next()), must produce one tensor    # TODO: support multiple parts for data
        :param valid_data_gen: data generator for validation (must have next()), must produce one tensor
        :param cuda:
        :param netR:    network for processing real examples; trained with discriminator
                        allows specifying different network for processing real examples
        :return:
        """
        assert(not (sample_real_for_gen and feed_real_to_gen))
        print("status: supporting different nets for discriminator and generator training")
        data_gen = data_gen if data_gen is not None else self.data_iter
        valid_data_gen = valid_data_gen if valid_data_gen is not None else self.valid_data_iter

        if isinstance(netD, tuple):
            assert(len(netD) == 2)
            netD4D = netD[0]
            netD4G = netD[1]
        else:
            netD4D, netD4G = netD, netD

        if isinstance(netG, tuple):
            assert(len(netG) == 2)
            netG4D = netG[0]
            netG4G = netG[1]
        else:
            netG4D, netG4G = netG

        if cuda:
            netD4D.cuda()
            if netD4D != netD4G:
                netD4G.cuda()
            netG4D.cuda()
            if netG4G != netG4D:
                netG4G.cuda()
            if netR is not None:
                netR.cuda()

        valid_EMD = 0.
        valid_fake2real = 0.    # mean of distances from each fake to closest real
        valid_real2fake = 0.    # mean of distances from each real to closest fake
        valid_fakeandreal = 0.  # harmonic mean of above

        vnoise = q.var(torch.zeros(batsizeG, self.noise_dim)).cuda(cuda).v

        netD4Dparams = q.params_of(netD4D)
        netD4Gparams = q.params_of(netD4G)

        for _iter in range(niter):
            #### DYNAMIC HYPERPARAMS ####
            for dynhparam in self.dynamichyperparams:
                dynhparam.update(iter=_iter)

            ##########  Update D net ##########
            netD, netG = netD4D, netG4D
            lip_loss = None
            for p in netD4Dparams:
                p.requires_grad = True

            if _iter < 25 or _iter % 500 == 0:
                _niterD = 100
            else:
                _niterD = niterD

            _niterD = 1     # for debugging

            for j in range(_niterD):
                netD.zero_grad()
                if self.mode == "WGAN":
                    for p in netD.parameters():
                        p.data.clamp_(*self.clamp_weights_rng)
                data = next(data_gen)
                if feed_real_to_gen:
                    data4gen = data
                elif sample_real_for_gen:     # ensure data sizes consistency
                    data4gen = next(data_gen)
                    if data.size(0) != data4gen.size(0):
                        minimumsize = min(data.size(0), data4gen.size(0))
                        data = data[:minimumsize]
                        data4gen = data4gen[:minimumsize]

                real = q.var(data).cuda(cuda).v
                if netR is not None:
                    real = netR(real)
                num_examples = real.size(0)
                batsize = num_examples
                vnoise = q.var(torch.FloatTensor(num_examples, self.noise_dim)).cuda(real).v
                vnoise.data.normal_(0, 1)

                if not sample_real_for_gen:
                    fake = netG(vnoise)
                else:
                    real4gen = q.var(data4gen).cuda(cuda).v
                    fake = netG(vnoise, real4gen)

                scoreD_real_vec = netD(real)        # (batsize,)
                #scoreD_real = scoreD_real_vec.mean()
                # scoreD_real.backward(one, retain_graph=(lc> 0))
                scoreD_fake_vec = netD(fake.detach())        # TODO: netD(fake.detach())  ?
                #scoreD_fake = scoreD_fake_vec.mean()
                # scoreD_fake.backward(mone, retain_graph=(lc > 0))
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
                    interp_alpha = real.data.new(num_examples, *([1]*(real.dim()-1)))
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
            for j in range(niterG):
                netD, netG = netD4G, netG4G

                for p in netD4Gparams:
                    p.requires_grad = False
                netG.zero_grad()

                if sample_real_for_gen or feed_real_to_gen:
                    real4gen = next(data_gen)
                    real4gen = q.var(real4gen).cuda(cuda).v
                    vgnoise = q.var(torch.zeros(real4gen.size(0), self.noise_dim)).cuda(cuda).v
                    vgnoise.data.normal_(0, 1)
                    fake = netG(vgnoise, real4gen)
                else:
                    vgnoise = q.var(torch.zeros(batsizeG, self.noise_dim)).cuda(cuda).v
                    vgnoise.data.normal_(0, 1)
                    fake = netG(vgnoise)

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
                                lip_loss=lip_loss.data[0] if lip_loss is not None else 0.,
                                valid_EMD=valid_EMD, valid_fake2real=valid_fake2real,
                                valid_real2fake=valid_real2fake, valid_fakeandreal=valid_fakeandreal,
                                when="after_G")

