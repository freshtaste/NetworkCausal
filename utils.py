import numpy as np


class Params(object):
    
    def __init__(self, N=30000, M=10000, K=1):
        self.N = N
        self.M = M
        self.K = K
        # parameters for the covariate
        self.normalX = False
        self.muX = np.arange(self.K) - (self.K-1)/2
        self.sig2X = np.ones(self.K)*0.2
        self.pX = 0.5
        # parameters for the treatment model
        self.betaZ =  np.linspace(-0.5, 0.5, self.K*2)
        self.epsZ = 1
        # parameters for outcome model
        self.betaXc =  np.linspace(0, 0.3, self.K*2)
        self.tau = 1
        self.gamma = 0.1
        self.alpha = 0
        self.sig2eps = 1
        self.fakeG = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
