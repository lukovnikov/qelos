import torch
from torch.autograd import Variable
import qelos as q

class GANTrainer(object):
    def __init__(self,  mode="DRAGAN",      # WGAN, WGAN-GP, DRAGAN, DRAGAN-G, DRAGAN-LG
                        modeD="critic",  # disc or critic
                        one_sided=False,
                        penalty_weight=None,
                        perturb_both=False,
                        perturb_symmetric=False,
                        perturb_scale=1.,
                        clamp_weights_rng=(-0.01, 0.01),
                        optimizerD=None,
                        optimizerG=None,
                        logger=None):
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
        self.logger = logger
        self._dragan_lg_var = 0.6

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
              data_gen=None, cuda=False):
        if cuda:
            netD.cuda()
            netG.cuda()

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
                vnoise = Variable(real.data.new(num_examples, self.noise_dim))
                vnoise.data.normal_(0, 1)
                fake = netG(vnoise)
                scoreD_real_vec = netD(real)        # (batsize,)
                scoreD_real = scoreD_real_vec.mean()
                # scoreD_real.backward(one, retain_graph=(lc> 0))
                scoreD_fake_vec = netD(fake)
                scoreD_fake = scoreD_fake_vec.mean()
                # scoreD_fake.backward(mone, retain_graph=(lc > 0))
                if self.mode == "WGAN":
                    errD = scoreD_fake - scoreD_real
                elif self.mode == "WGAN-GP":
                    errD = scoreD_fake - scoreD_real
                    interp_alpha = real.data.new(num_examples, 1)
                    interp_alpha.uniform_(0, 1)
                    interp_points = interp_alpha * real.data + (1 - interp_alpha) * fake.data
                    grad_points = Variable(interp_points.clone())
                    interp_points = Variable(interp_points, requires_grad=True)
                    errD_interp_vec = netD(interp_points)
                    errD_gradient, = torch.autograd.grad(errD_interp_vec.sum(),
                                                         interp_points,
                                                         create_graph=True)
                    lip_grad_norm = errD_gradient.norm(2, dim=1)
                    lip_loss = self.penalty_weight * (self.clip_fn(lip_grad_norm - 1) ** 2).mean()
                    errD = errD + lip_loss
                elif self.mode == "DRAGAN" or self.mode == "DRAGAN-G" or self.mode == "DRAGAN-LG":
                    if self.modeD == "disc":
                        errD = - torch.log(scoreD_real) - torch.log(1 - scoreD_fake)
                    elif self.modeD == "critic":
                        errD = scoreD_fake - scoreD_real

                    real_perturbed = self.perturb(real)
                    grad_points = real_perturbed.data
                    errD_real_perturbed_vec = netD(real_perturbed)
                    errD_gradient, = torch.autograd.grad(
                        errD_real_perturbed_vec.sum(), real_perturbed, create_graph=True)
                    lip_grad_norm = errD_gradient.norm(2)
                    lip_loss = self.penalty_weight * (self.clip_fn(lip_grad_norm - 1) ** 2).mean()
                    if self.perturb_both:
                        fake_perturbed = self.perturb(fake)
                        errD_fake_perturbed_vec = netD(fake_perturbed)
                        errD_fake_gradient, = torch.autograd.grad(
                            errD_fake_perturbed_vec.sum(), fake_perturbed, create_graph=True)
                        lip_fake_grad_norm = errD_fake_gradient.norm(2)
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
            errG = -errG.mean()
            errG.backward()
            self.optimizerG.step()

            if self.logger is not None:
                self.logger.log(_iter=_iter, niter=niter,
                                real=real.data, fake=fake.data, grad_points=grad_points,
                                errD=errD.data[0], errG=errG.data[0],
                                scoreD_real=scoreD_real.data[0],
                                scoreD_fake=scoreD_fake.data[0],
                                lip_loss=lip_loss.data[0] if lip_loss is not None else 0.,
                                when="after_G")

