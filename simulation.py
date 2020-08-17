import numpy as np
from utils import Params, sigmoid
from scipy.special import softmax


class Simulation(object):
    
    def __init__(self, params=None):
        if params == None:
            self.Params = Params()
        else:
            self.Params = params
        self.Net = None
        self.X = None
        self.Z = None
        self.G = None
        self.Xc = None
        self.Y = None
        self.prop_idv = None
        self.prop_neigh = None
        self.n = int(self.Params.N/self.Params.M)
        
    
    def get_data(self):
        self.__getNet(), self.__getX()
        self.__getZ()
        self.__getG()
        self.__getY()
        return self.Xc, self.Z, self.G, self.Y
    
    
    def __getNet(self):
        labels = list(np.arange(self.Params.M))*self.n
        self.Net = sorted(labels)
    
        
    def __getX(self):
        if self.Params.normalX:
            X = np.random.multivariate_normal(self.Params.muX, 
                            np.diag(self.Params.sig2X), (self.Params.M, self.n))
        else:
            X = np.random.choice([0,1],size=(self.Params.M, self.n, self.Params.K),
                             p=[1-self.Params.pX,self.Params.pX])
        Xc = np.zeros((self.Params.M, self.n, self.Params.K*2))
        Xcmean = np.mean(X, axis=1).reshape((self.Params.M, 1, self.Params.K))
        Xc[:,:,:self.Params.K] = X.reshape((self.Params.M, self.n, self.Params.K))
        Xc[:,:,self.Params.K:] = np.repeat(Xcmean, self.n, axis=1)
        self.X = X.reshape(self.Params.N, self.Params.K)
        self.Xc = Xc.reshape(self.Params.N, self.Params.K*2)
    
        
    def __getZ(self):
        self.Z = np.zeros(self.Params.N)
        self.prop_idv = sigmoid(self.Xc.dot(self.Params.betaZ))
        unif = np.random.uniform(0,1,self.Params.N)
        self.Z[unif < self.prop_idv] = 1

    
    def __getG(self):
        if self.Params.fakeG:
            params = 1 + np.arange(self.n*self.Params.K*2)/(self.n*self.Params.K*2)
            params = params.reshape((self.n, self.Params.K*2))
            U = self.Xc @ params.T
            self.G = np.argmax(U+np.random.gumbel(size=(self.Params.N,self.n)),axis=1)
            self.prop_neigh = softmax(U, axis=1)
        else:
            Z_stack = self.Z.reshape(self.Params.M, self.n)
            G_plus = np.repeat(np.sum(Z_stack, axis=1).reshape(self.Params.M, 1),
                               self.n, axis=1)
            G_stack = G_plus - Z_stack
            self.G = G_stack.reshape(self.Params.N)
        if len(set(self.G)) != self.n:
            raise RuntimeError("G is not fully generated.")
        
    
    def __getY(self):
        epsilon = np.random.normal(0, self.Params.sig2eps, self.Params.N)
        self.Y = (self.Params.alpha + 
             self.Params.tau*self.Z + 
             self.Xc.dot(self.Params.betaXc) + 
             self.Params.gamma * self.G * self.Z + 
             epsilon)
    
    
    def get_efficiency_bound(self):
        bound_thm = []
        for g in range(self.n):
            bound_thm.append(np.mean(self.Params.sig2eps/self.prop_neigh[:,g] * 
                        (1/self.prop_idv + 1/(1-self.prop_idv))))
        return np.array(bound_thm)