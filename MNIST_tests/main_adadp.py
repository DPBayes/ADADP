








'''

A code for training in a differentially private manner a fully
connected network using ADADP.
Here the method is applied to the MNIST data set.

The ADADP algorithm is described in

Koskela, A. and Honkela, A.,
Learning rate adaptation for differentially private stochastic gradient descent.
arXiv preprint arXiv:1809.03832. (2018)

This code is due to Antti Koskela (@koskeant) and is based
on a code by Mikko Heikkilä (@mixheikk).

'''











import copy
import datetime
import numpy as np
import pickle
import sys
import time
import logging
from collections import OrderedDict as od
from matplotlib import pyplot as plt
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision

from torchvision import datasets, transforms

import linear

import adadp

import gaussian_moments as gm

import itertools
from types import SimpleNamespace
import px_expander





parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                    help='input batch size for training')
parser.add_argument('--noise_sigma', type=float, default=2.0, metavar='M',
                    help='noise_sigma')
parser.add_argument('--n_epochs', type=int, default=10, metavar='N',
                    help='n_epochs')
parser.add_argument('--run_id', type=int, default=1, metavar='N',
                    help='run_id')
parser.add_argument('--tol', type=int, default=1.0, metavar='t',
                    help='tolerance parameter')

args = parser.parse_args()



randomize_data = True
batch_size = args.batch_size # Note: overwritten by BO if used, last batch is skipped if not full size
batch_proc_size = 10 # needs to divide or => to batch size

n_hidden_layers = 1 # number of units/layer (same for all) is set in bo parameters
latent_dim = 512 # Note: overwritten by BO if used
output_dim = 10
log_interval = 6000//batch_size # Note: this is absolute interval, actual is this//batch_size




use_dp = True # dp vs non-dp model
scale_grads = True
grad_norm_max = 10
noise_sigma = args.noise_sigma
delta = 1e-5

tol = args.tol

n_epochs = args.n_epochs
l_rate = 0.01

run_id = args.run_id


np.random.seed(17*run_id+3)

input_dim = (28,28)




if torch.cuda.is_available() and torch.cuda.device_count() > 0:

  print('Using cuda')
  torch.cuda.manual_seed(11*run_id+19)
  use_cuda = True
  data_dir = './data/'



trainset = torchvision.datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                      ]))

testset = torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))]))


sampling_ratio = float(batch_size)/len(trainset)




# moments accountant
def update_privacy_pars(priv_pars):
  verify = False
  max_lmbd = 32
  lmbds = range(1, max_lmbd + 1)
  log_moments = []
  for lmbd in lmbds:
    log_moment = 0
    '''
    print('Here q = ' + str(priv_pars['q']))
    print('Here sigma = ' + str(priv_pars['sigma']))
    print('Here T = ' + str(priv_pars['T']))
    '''
    log_moment += gm.compute_log_moment(priv_pars['q'], priv_pars['sigma'], priv_pars['T'], lmbd, verify=verify)
    log_moments.append((lmbd, log_moment))
  priv_pars['eps'], _ = gm.get_privacy_spent(log_moments, target_delta=priv_pars['delta'])
  return priv_pars









class simpleExpandedDNN(nn.Module):
  def __init__(self, batch_size, batch_proc_size):
    super(simpleExpandedDNN, self).__init__()
    #self.lrelu = nn.LeakyReLU()
    self.relu = nn.ReLU()

    self.batch_proc_size = batch_proc_size
    self.batch_size = batch_size

    self.linears = nn.ModuleList([ linear.Linear(1*input_dim[0]*input_dim[1], latent_dim, bias=False, batch_size=batch_proc_size)])
    if n_hidden_layers > 0:
      for k in range(n_hidden_layers):
        self.linears.append( linear.Linear(latent_dim, latent_dim,bias=False,batch_size=batch_proc_size) )
    self.final_fc = linear.Linear(self.linears[-1].out_features, output_dim,bias=False, batch_size=batch_proc_size)
    self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                    shuffle=randomize_data, num_workers=4)
    self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=randomize_data, num_workers=4)

  def forward(self, x):

    x = torch.unsqueeze(x.view(-1, 1*input_dim[0]*input_dim[1]),1)

    for k_linear in self.linears:
      x = self.relu(k_linear(x))
    x = self.final_fc(x)
    return nn.functional.log_softmax(x.view(-1,output_dim),dim=1)




model = simpleExpandedDNN(batch_size=batch_size, batch_proc_size=batch_proc_size)

print('model: {}'.format(model))


for p in model.parameters():
  if p is not None:
    p.data.copy_( p[0].data.clone().repeat(batch_proc_size,1,1) )

if use_cuda:
  model = model.cuda()

loss_function = nn.NLLLoss(size_average=False)




#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=l_rate, momentum=0)
optimizer = adadp.ADADP(model.parameters())










def train(epoch, model, T):

  model.train()
  ii=0


  print('run_id: ' +str(run_id))

  for batch_idx, (data, target) in enumerate(model.train_loader):
    if data.shape[0] != batch_size:
      print('skipped last batch')
      continue

    optimizer.zero_grad()
    loss_tot = 0

    data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
    data, target = data.cuda(), target.cuda()


    if use_dp and scale_grads:
      cum_grads = od()
      for i,p in enumerate(model.parameters()):
        if p.requires_grad:
          cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]),requires_grad=False).cuda()

    for i_batch in range(batch_size//batch_proc_size):

      data_proc = data[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size,:]
      target_proc = target[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size]

      output = model(data_proc)

      loss = loss_function(output,target_proc)
      loss_tot += loss.data

      loss.backward()

      if use_dp and scale_grads:
        px_expander.acc_scaled_grads(model=model,C=grad_norm_max, cum_grads=cum_grads, use_cuda=use_cuda)
        optimizer.zero_grad()


    if use_dp:
      px_expander.add_noise_with_cum_grads(model=model, C=grad_norm_max, sigma=noise_sigma, cum_grads=cum_grads, use_cuda=use_cuda)

    # step1 corresponds to the first part of ADADP (i.e. only one step of size half),
    # step2 to the second part (error estimate + step size adaptation)

    if batch_idx%2 == 0:
      optimizer.step1()
    else:
      optimizer.step2(tol)

    #For SGD:
    #optimizer.step()


    T += 1

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(model.train_loader.dataset),
        100. * batch_idx / len(model.train_loader), loss_tot.item()/batch_size))


  return T








def test(model, epoch):

  model.eval()

  test_loss = 0
  correct = 0

  for data, target in model.test_loader:
    if data.shape[0] != model.batch_size:
      print('skipped last batch')
      continue

    data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
    data, target = data.cuda(), target.cuda()

    for i_batch in range(model.batch_size//batch_proc_size):

      data_proc = data[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size,:]
      target_proc = target[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size]
      data_proc = data_proc.cuda()
      target_proc = target_proc.cuda()

      output = model(data_proc)

      test_loss += F.nll_loss(output, target_proc, size_average=False).item()

      pred = output.data.max(1, keepdim=True)[1]

      correct += pred.eq(target_proc.data.view_as(pred)).cpu().sum()

  test_loss /= len(model.test_loader.dataset)

  acc = correct.numpy() / len(model.test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(model.test_loader.dataset),
    100. * acc))

  return test_loss, acc




priv_pars = od()
priv_pars['T'], priv_pars['eps'],priv_pars['delta'], priv_pars['sigma'], priv_pars['q'] = 0, 0, delta, noise_sigma, sampling_ratio





accs = []
epsilons = []

for epoch in range(1,n_epochs+1):

  loss, acc = test(model, epoch)

  accs.append(acc)

  print('Current privacy pars: {}'.format(priv_pars))
  priv_pars['T'] = train(epoch, model, priv_pars['T'])

  if use_dp and scale_grads and noise_sigma > 0:
    update_privacy_pars(priv_pars)

  epsilons.append(priv_pars['eps'])

# Save the test accuracies
np.save('accs_' +str(run_id) + '_' + str(noise_sigma) + '_' + str(batch_size),accs)
