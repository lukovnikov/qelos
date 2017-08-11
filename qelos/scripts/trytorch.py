import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython import embed


x = Variable(torch.from_numpy(np.random.random((10, 10)).astype("float32")))
y = x ** 2
yd = y.data.numpy()
z = y.sum()
embed()
print y