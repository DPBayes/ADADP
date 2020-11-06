



'''

A code for implementing the ADADP algorithm for neural networks,
described in

Koskela, A. and Honkela, A.,
Learning rate adaptation for differentially private stochastic gradient descent.
arXiv preprint arXiv:1809.03832. (2018)

The code is due to Antti Koskela (@koskeant)

'''





import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

class ADADP(Optimizer):

    def __init__(self, params, lr=1e-3):

        defaults = dict(lr=lr)

        self.p0 = None
        self.p1 = None
        self.lrs = lr
        self.accepted = 0
        self.failed = 0

        self.lrs_history = []

        super(ADADP, self).__init__(params, defaults)

    def step1(self):

        del self.p0
        self.p0 = []

        del self.p1
        self.p1 = []

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                dd = p.data.clone()
                self.p0.append(dd)

                self.p1.append(p.data  -  self.lrs*p.grad.data)
                p.data.add_(-0.5*self.lrs, p.grad.data)

    def step2(self, tol=1.0):

        for group in self.param_groups:

            err_e = 0.0

            for ijk,p in enumerate(group['params']):
                p.data.add_(-0.5*self.lrs, p.grad.data)
                err_e += (((self.p1[ijk] - p.data)**2/(torch.max(torch.ones(self.p1[ijk].size()),self.p1[ijk]**2))).norm(1))

            err_e = np.sqrt(float(err_e))

            self.lrs = float(self.lrs*min(max(np.sqrt(tol/err_e),0.9), 1.1))

            ## Accept the step only if err < tol.
            ## Can be sometimes neglected (more accepted steps)
            if err_e > 1.0*tol:
               for ijk,p in enumerate(group['params']):
                   p.data = self.p0[ijk]
            if err_e < tol:
               self.accepted += 1
            else :
               self.failed += 1

            self.lrs_history.append(self.lrs)
