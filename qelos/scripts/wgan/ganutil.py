from matplotlib import pyplot, gridspec
import matplotlib.backends.backend_pdf as mplpdf
import numpy
import torch
from torch.autograd import Variable
import IPython
import qelos as q
from IPython.display import clear_output, display
from datetime import datetime as dt

# Generates and saves a plot of the true distribution, the generator, and the
# critic.
# largely form Improved Training of Wasserstein GAN code (see link above)
class ImageGenerator:
  def __init__(self, netG, netD, prefix='frame', noise_dim=2, save_frames_to_pdf=False):
    self.prefix = prefix
    self.frame_index = 1
    self.noise_dim = noise_dim
    self.netG = netG
    self.netD = netD
    self.frames2pdf = save_frames_to_pdf
    self.figs4pdf = []

  def __call__(self, true_dist, perturbed, losses):
    try:
        N_POINTS = 128
        RANGE = 2

        points = numpy.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:,:,0] = numpy.linspace(-RANGE, RANGE, N_POINTS)[:,None]
        points[:,:,1] = numpy.linspace(-RANGE, RANGE, N_POINTS)[None,:]
        points = points.reshape((-1,2))
        points = Variable(torch.from_numpy(points))
        if true_dist.is_cuda:
            points = points.cuda()

        noise = Variable(true_dist.new(true_dist.size(0), 2))
        noise.data.normal_(0, 1)

        fake = self.netG(noise)
        samples = fake.data.cpu().numpy()

        disc_points = self.netD(points)

        disc_map = disc_points.data.cpu().numpy()
        # disc_map = (disc_map - numpy.min(disc_map)) / numpy.max(disc_map) * 8
        # disc_map = numpy.log(disc_map + 0.25)

        fig = pyplot.figure(num=1, figsize=(10,15))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        pyplot.clf()
        if self.prefix is None:
            #pyplot.suplots(nrows=1, ncols=2)
            pyplot.subplot(gs[0])
        x = y = numpy.linspace(-RANGE, RANGE, N_POINTS)
        disc_map = disc_map.reshape((len(x), len(y))).T
        pyplot.contour(x, y, disc_map)

        true_dist = true_dist.cpu().numpy()
        perturbed = perturbed.cpu().numpy()

        # save
        ret = {"true_dist": true_dist,
               "perturbed": perturbed,
               "samples": samples,
               "contour": {"x": x, "y": y, "disc_map": disc_map}}

        # plot scatter
        pyplot.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',marker='+')
        pyplot.scatter(perturbed[:, 0], perturbed[:, 1], c='red', marker='+')
        pyplot.scatter(samples[:, 0],   samples[:, 1],   c='green', marker='*')
        if self.frames2pdf:
            self.figs4pdf.append(fig)
        if self.prefix is not None:
          pyplot.savefig(self.prefix+'{:05d}'.format(self.frame_index)+'.jpg')
        else:
          pyplot.subplot(gs[1])
          pyplot.plot(losses)
          clear_output(wait=True)
          display(pyplot.gcf())
        self.frame_index += 1

        return ret      # return saved

    except Exception as e:
        print("some exception occurred while plotting")

  def finalize(self, savedir="experiments/", savename=None, settings=None):
      if self.frames2pdf:
          # generate savename
          if savename is None:
              extraoptstring = ""
              if settings.lip_mode in "DRAGAN DRAGAN-G DRAGAN-LG":
                  extraoptstring += "_pboth={}".format(int(settings.perturb_both))
              if settings.lip_mode == "DRAGAN":
                  extraoptstring += "_psym={}".format(int(settings.perturb_symmetric))
              savename = "{}_os={}_pw={}{}_niter={}_at_{}".format(
                  settings.lip_mode, int(settings.onesided), settings.penalty_weight, extraoptstring, settings.niter,
                  str(dt.now()).replace(" ", "_"))
          savep = savedir + savename + ".pdf"
          # do save
          pdf = mplpdf.PdfPages(savep)
          for fig in self.figs4pdf:
              pdf.savefig(fig)
          pdf.close()