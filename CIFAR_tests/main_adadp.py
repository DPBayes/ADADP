


'''

A code for training in a differentially private manner the fully connected
layers of a simple convolutive network using ADADP.
Here the method is applied to the Cifar-10 data set.
The parameters for the convolutive layers are loaded from a file "conv_layers.pt".

The ADADP algorithm is described in

Koskela, A. and Honkela, A.,  
Learning rate adaptation for differentially private stochastic gradient descent. 
arXiv preprint arXiv:1809.03832. (2018)

This code is due to Antti Koskela (@koskeant) and is based
on a code by Mikko Heikkinen (@mixheikk).

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
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='input batch size for training')
parser.add_argument('--noise_sigma', type=float, default=8.0, metavar='M',
                    help='noise_sigma')
parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                    help='n_epochs')
parser.add_argument('--run_id', type=int, default=1, metavar='N',
                    help='run_id')
parser.add_argument('--tol', type=float, default=1.0, metavar='t',
                    help='tolerance parameter')





args = parser.parse_args()


randomize_data = True
batch_size = args.batch_size 
batch_proc_size = 10 # needs to divide or => to batch size

n_hidden_layers = 1 # number of hidden layers in the feedforward network
latent_dim = 500 #width of the hidden layers
output_dim = 10
log_interval = 6000//batch_size 

use_dp = True 
grad_norm_max = 3
noise_sigma = args.noise_sigma
delta = 1e-5

tol = args.tol

n_epochs = args.n_epochs
l_rate = 0.01

run_id = args.run_id


np.random.seed(17*run_id+3)







  
if torch.cuda.is_available() and torch.cuda.device_count() > 0:

  print('Using cuda')
  torch.cuda.manual_seed(11*run_id+19)
  use_cuda = True
  data_dir = './data/'

  






  
transform = torchvision.transforms.Compose([])
  

transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform_train)
 
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform_test)

sampling_ratio = batch_size/len(trainset)

  








# moments accountant

def update_privacy_pars(priv_pars):
  verify = False
  max_lmbd = 32
  lmbds = range(1, max_lmbd + 1)
  log_moments = []
  for lmbd in lmbds:
    log_moment = 0
    log_moment += gm.compute_log_moment(priv_pars['q'], priv_pars['sigma'], priv_pars['T'], lmbd, verify=verify)
    log_moments.append((lmbd, log_moment))
  priv_pars['eps'], _ = gm.get_privacy_spent(log_moments, target_delta=priv_pars['delta'])
  return priv_pars






# The convolutional part of the network


class Net1(nn.Module):
  def __init__(self):
    super(Net1, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))       
    return x
    
model1 =  Net1().cuda()




# Load the pre-trained convolutive layers

tb_save = torch.load('conv_layers.pt')

for ii,p in enumerate(model1.parameters()):
  if(ii<4):
    p.data = tb_save[ii].clone()








# The fully connected part of the network

class Net2(nn.Module):
  def __init__(self, batch_size, batch_proc_size):
    super(Net2, self).__init__()
    self.relu = nn.ReLU()
      
    self.batch_proc_size = batch_proc_size
    self.batch_size = batch_size
      
    self.linears = nn.ModuleList([ linear.Linear(1600, latent_dim, bias=False, batch_size=batch_proc_size)])
    if n_hidden_layers > 0:
      for k in range(n_hidden_layers):
        self.linears.append( linear.Linear(latent_dim, latent_dim,bias=False,batch_size=batch_proc_size) )
    self.final_fc = linear.Linear(self.linears[-1].out_features, output_dim,bias=False, batch_size=batch_proc_size)
    self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                    shuffle=randomize_data, num_workers=4)
    self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                    shuffle=randomize_data, num_workers=4)
      
  def forward(self, x):
    x = torch.unsqueeze(x.view(-1, 1600),1)
    for k_linear in self.linears:
      x = self.relu(k_linear(x))
    x = self.final_fc(x)   
    return nn.functional.log_softmax(x.view(-1,output_dim),dim=1)



model2 = Net2(batch_size=batch_size, batch_proc_size=batch_proc_size)

for p in model2.parameters():
  if p is not None:
    p.data.copy_( p[0].data.clone().repeat(batch_proc_size,1,1) )
 
if use_cuda:
  model1 = model1.cuda()
  model2 = model2.cuda()

loss_function = nn.NLLLoss(size_average=False)




#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model2.parameters()), lr=l_rate, momentum=0)
optimizer = adadp.ADADP(model2.parameters())











def train(epoch, model1, model2, T):

  model1.train()
  model2.train()
          
  for batch_idx, (data, target) in enumerate(model2.train_loader):

    if data.shape[0] != batch_size: 
      continue

    optimizer.zero_grad()
    loss_tot = 0

    data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
    data, target = data.cuda(), target.cuda()

    cum_grads = od()
    for i,p in enumerate(model2.parameters()):
      if p.requires_grad:
        cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]),requires_grad=False).cuda()
                  
    for i_batch in range(batch_size//batch_proc_size):
           
      data_proc = data[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size,:]
      target_proc = target[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size]
              
      output1 = model1(data_proc)
      output2 = model2(output1)

      loss = loss_function(output2,target_proc)
      loss_tot += loss.data

      loss.backward()
            
      if use_dp:
        px_expander.acc_scaled_grads(model=model2,C=grad_norm_max, cum_grads=cum_grads, use_cuda=use_cuda)
        optimizer.zero_grad()

    if use_dp:
      px_expander.add_noise_with_cum_grads(model=model2, C=grad_norm_max, sigma=noise_sigma, cum_grads=cum_grads, use_cuda=use_cuda)


    # step1 corresponds to the first part of ADADP (i.e. only one step of size half),
    # step2 to the second part (error estimate + step size adaptation)

    if batch_idx%2 is 0:
      optimizer.step1()
    else:
      optimizer.step2(tol)
            
    #optimizer.step()

    T += 1

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(model2.train_loader.dataset),
        100. * batch_idx / len(model2.train_loader), loss_tot.item()/batch_size))
            
  return T






def test(model1, model2, epoch):

  model1.eval()
  model2.eval()

  test_loss = 0
  correct = 0

  for data, target in model2.test_loader:
    if data.shape[0] != model2.batch_size:
      print('skipped last batch')
      continue

    data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
    data, target = data.cuda(), target.cuda()

    for i_batch in range(model2.batch_size//batch_proc_size):

      data_proc = data[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size,:]
      target_proc = target[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size]
      data_proc = data_proc.cuda()
      target_proc = target_proc.cuda()

      output1 = model1(data_proc)                   
      output2 = model2(output1)
            
      test_loss += F.nll_loss(output2, target_proc, size_average=False).item()

      pred = output2.data.max(1, keepdim=True)[1]
      correct += pred.eq(target_proc.data.view_as(pred)).cpu().sum()

  test_loss /= len(model2.test_loader.dataset)
  acc = correct.numpy() / len(model2.test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(model2.test_loader.dataset),100. * acc))

  return test_loss, acc








priv_pars = od()
priv_pars['T'], priv_pars['eps'],priv_pars['delta'], priv_pars['sigma'], priv_pars['q'] = 0, 0, delta, noise_sigma, sampling_ratio

accs = []
epsilons = []

for epoch in range(1,n_epochs+1):

  loss, acc = test(model1, model2, epoch)
  accs.append(acc)
  print('Current privacy pars: {}'.format(priv_pars))
  priv_pars['T'] = train(epoch, model1, model2, priv_pars['T'])

  if noise_sigma>0:

    update_privacy_pars(priv_pars)
    epsilons.append(priv_pars['eps'])



# Save the test accuracies
np.save('accs_' +str(run_id) + '_' + str(noise_sigma) + '_' + str(batch_size),accs)
  





