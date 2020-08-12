#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class MC_Machine:
    
    def __init__(self, scale = 0.1, L = 20, epsilon = 0.1, num_simu = 1000, burn_in = 0.0):
        '''
        scale: scale parameter in Metropolis-Hasting Algorithm
        L: number of steps in leapfrog
        epsilon: the step size in leapfrog
        num_simu: number of simulated samples 
        '''
        from numpy.random import multivariate_normal as mvn
        from numpy.random import uniform as unif
        from numpy.random import normal
        import numpy as np

        self.np = np
        self.mvn = mvn
        self.unif = unif
        self.normal = normal
        self.L = L
        self.epsilon = epsilon
        self.num_simu = num_simu
        self.scale = scale
        self.burn_in = burn_in
        
    def HMC(self, U, grad_U, current_q, L = None, epsilon = None, num_simu = None, burn_in = None):
        '''
        q is the position variable that you want to sample,
        current_q is the initial point,
        p is the momentum variable.
        Currently mass matrix is assumed to be identity matrix.
        '''
        
        result = {}
        
        if not L:
            L = self.L 
        if not epsilon:
            epsilon = self.epsilon
        if not num_simu:
            num_simu = self.num_simu
        if not burn_in:
            burn_in = self.burn_in
        if isinstance(burn_in, float):
            start_index = int(burn_in * num_simu)
        elif isinstance(burn_in, int):
            start_index = burn_in
        else:
            raise ValueError('burn_in must be float or int')
            
        current_q = self.np.array(current_q)
        dim_q = current_q.shape[0]
        
        trajectory = self.np.zeros((num_simu,dim_q))
        current_q += 0.0
        p_initials = self.normal(0, 1, dim_q * num_simu).reshape(num_simu, dim_q)

        unif_numbers = self.unif(low = 0, high = 1, size = num_simu)
        count = 0.0
        
        for sim in range(num_simu):

            q = current_q + 0.0
            
            current_p = p_initials[sim,].reshape(dim_q, 1)
            p = current_p + 0.0
        
            p -= epsilon * grad_U(q) / 2 #half step update for momentum p
           
            for i in range(L-1):
                q += epsilon * p
                p -= epsilon * grad_U(q)
            q += epsilon * p
            p -= epsilon * grad_U(q) / 2
            
            current_U = U(current_q)
            candidate_U = U(q)
            current_K = self.np.sum(current_p ** 2) / 2
            candidate_K = self.np.sum(p ** 2) / 2

            tmp = current_U + current_K - candidate_U - candidate_K
            if self.np.log(self.unif(low = 0, high = 1)) < tmp:
                trajectory[sim,:] = q.flatten() + 0.0
                current_q = q + 0.0
                count += 1
            else:
                trajectory[sim,:] = current_q.flatten()
        
        result['trajectory'] = trajectory[start_index:,]
        result['accept_rate'] = count / num_simu
        return result
        
    def MCMC(self, U, current_x, num_simu = None, scale = None, burn_in = None):
        '''
        U is the negate of log likelihood function,
        current_x is the initial point.
        Here, the proposal function is chosen to be symmetric guassian distribution
        '''
        
        result = {}
        if not scale:
            scale = self.scale 
        if not num_simu:
            num_simu = self.num_simu
        if not burn_in:
            burn_in = self.burn_in
        if isinstance(burn_in, float):
            start_index = int(burn_in * num_simu)
        elif isinstance(burn_in, int):
            start_index = burn_in
        else:
            raise ValueError('burn_in must be float or int')
        current_x = self.np.array(current_x)
        current_x = current_x.reshape(-1, 1)
        dim_x = current_x.shape[0]
        
        trajectory = self.np.zeros((num_simu,dim_x))
        unif_numbers = self.unif(low = 0, high = 1, size = num_simu)
        count = 0.0
        
        for sim in range(num_simu):
            proposed_x = current_x.reshape(dim_x, 1) + scale * self.normal(0, 1, dim_x).reshape(dim_x, -1) 
            
            if self.np.log(unif_numbers[sim]) < U(current_x) - U(proposed_x):
                trajectory[sim,:] = proposed_x.flatten()
                current_x = proposed_x + 0.0
                count += 1
            else:
                trajectory[sim,:] = current_x.flatten()
        
        result['trajectory'] = trajectory[start_index:,]
        result['accept_rate'] = count / num_simu
        return result

