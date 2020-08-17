import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from simulation import Simulation
from utils import Params
from itertools import product


class Estimate(object):
    
    def __init__(self, sim):
        self.sim = sim
        self.Xc = sim.Xc
        self.Z = sim.Z
        self.G = sim.G
        self.Y = sim.Y
        self.prop_idv, self.prop_neigh = self.est_propensity()
        if sim.Params.fakeG:
            self.prop_idv_true, self.prop_neigh_true = sim.prop_idv, sim.prop_neigh
        else:
            self.prop_idv_true, self.prop_neigh_true = self.prop_idv, self.prop_neigh
        self.n = len(set(self.G))
        self.M = int(len(self.Z)/self.n)
        if sim.Params.normalX:
            self.sigXc = self.get_variance_linear()
        else:
            self.sigXc = self.get_variance_nonp()
        
        
    
    def est_propensity(self):
        idv = LogisticRegression(random_state=0,solver='newton-cg').fit(self.Xc, self.Z)
        prop_idv = idv.predict_proba(self.Xc)[:,1]
        neigh = LogisticRegression(random_state=0,solver='newton-cg',
                                   multi_class='multinomial').fit(self.Xc, self.G)
        prop_neigh = neigh.predict_proba(self.Xc)
        return prop_idv, prop_neigh
    

    def explore_sample_balance(self):
        n_sample = np.zeros((2, self.n))
        for z in range(2):
            for g in range(self.n):
                n_sample[z,g] = np.sum((self.G == g) & (self.Z == z))
        return n_sample
    

    def est(self):
        result = {'tau(g)': np.zeros(self.n), 'se': np.zeros(self.n),
                  'se est': np.zeros(self.n), 'se thm': np.zeros(self.n), 
                  'se influence': np.zeros(self.n), 'z': np.zeros(self.n), 
                  'z est': np.zeros(self.n), 'z thm': np.zeros(self.n), 
                  'z influence': np.zeros(self.n), 'n_sample': self.explore_sample_balance(),
                  'bound_est': np.zeros(self.n), 'bound_thm': np.zeros(self.n),
                  'bound_influence': np.zeros(self.n), 'bound_empirical': np.zeros(self.n)}
        for g in range(self.n):
            arr = (self.G == g)*((self.Z - self.prop_idv)*self.Y/
                                  (self.prop_neigh[:,g]*self.prop_idv*(1-self.prop_idv)))
            bound_est = np.mean(self.sigXc/self.prop_neigh[:,g] * 
                            (1/self.prop_idv + 1/(1-self.prop_idv)))
            bound_thm = np.mean(self.sim.Params.sig2eps/self.prop_neigh_true[:,g] * 
                        (1/self.prop_idv_true + 1/(1-self.prop_idv_true)))
            # compute the influence function and its variance
            beta1 = (self.sim.Params.alpha + self.sim.Params.tau + 
                     self.sim.Params.gamma*g + self.Xc.dot(self.sim.Params.betaXc))
            beta0 = self.sim.Params.alpha + self.Xc.dot(self.sim.Params.betaXc)
            influence = (self.G == g)*(self.Z*(self.Y-beta1)/self.prop_idv_true - 
                            (1-self.Z)*(self.Y-beta0)/(1-self.prop_idv_true))/self.prop_neigh_true[:,g]
            bound_influence = np.var(np.mean(influence.reshape((self.sim.Params.M, self.n)), axis=1)) * self.n
            #print(np.var())
            result['bound_influence'][g] = bound_influence
            result['bound_est'][g] = bound_est
            result['bound_thm'][g] = bound_thm
            result['bound_empirical'][g] = np.var(arr)
            result['tau(g)'][g] = np.mean(arr)
            result['se'][g] = np.sqrt(np.var(arr)/self.sim.Params.N)
            result['se est'][g] = np.sqrt(bound_est/self.sim.Params.N)
            result['se thm'][g] = np.sqrt(bound_thm/self.sim.Params.N)
            result['se influence'][g] = np.sqrt(bound_influence/self.sim.Params.N)
            result['z'][g] = (result['tau(g)'][g] - self.sim.Params.tau - g*self.sim.Params.gamma)/result['se'][g]
            result['z est'][g] = (result['tau(g)'][g] - self.sim.Params.tau - g*self.sim.Params.gamma)/result['se est'][g]
            result['z thm'][g] = (result['tau(g)'][g] - self.sim.Params.tau - g*self.sim.Params.gamma)/result['se thm'][g]
            result['z influence'][g] = (result['tau(g)'][g] - self.sim.Params.tau - g*self.sim.Params.gamma)/result['se influence'][g]
        return result
    
    
    def get_variance_linear(self):
        regressor = np.concatenate([self.Z.reshape(-1,1), self.Z.reshape(-1,1) * self.G.reshape(-1,1), self.Xc], axis=1)
        #model = LinearRegression().fit(regressor, (self.Y - np.mean(self.Y))**2)
        #sighat = model.predict(regressor)
        model = LinearRegression().fit(regressor, self.Y)
        yhat = model.predict(regressor)
        sighat = (self.Y - yhat)**2
        return sighat
    
    
    def get_variance_nonp(self):
        sighat = np.zeros(self.sim.Params.N)
        configs = list(product(set(self.Xc[:,0]), set(self.Xc[:,1]), set(self.Z), set(self.G)))
        for config in configs:
            x0, x1, z, g = config
            indexing = (self.Xc[:,0]==x0) & (self.Xc[:,1]==x1) & (self.Z==z) & (self.G==g)
            if np.sum(indexing) > 0:
                sighat[indexing] = np.var(self.Y[indexing])
        return sighat
    
    

if __name__ == '__main__':
    N=50000
    M=5000
    gamma = 5
    K=1
    fakeG=True 
    normalX=False
    p = Params(N,M,K)
    p.gamma = gamma
    p.betaXc = p.betaXc - p.betaXc
    p.normalX = normalX
    p.fakeG = fakeG
    s = Simulation(p)
    _ = s.get_data()
    e = Estimate(s)
    result = e.est()
    print(result)
    #p_bound = Params(30000, 10000,K)
    #p_bound.gamma = gamma
    #p_bound.normalX = normalX
    #p_bound.fakeG = True
    #s_bound = Simulation(p_bound)
    #_ = s_bound.get_data()
    #bound = s_bound.get_efficiency_bound()
    #print(bound, np.sqrt(bound/M), np.sqrt(bound/N))