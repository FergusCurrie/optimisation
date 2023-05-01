'''
Iterative method to solve non-convex problems 

Willl use the example from slides. 

https://web.stanford.edu/class/ee364b/lectures/seq_slides.pdf

dot over variable = derivative of variable "dot notation" aka "newtonian notation"
'''

import numpy as np

class TwoLinkSystem:
    def __init__(self):
        self.m = np.array([1,5])
        self.l = np.array([1,1])
        self.N = 40
        self.T = 10
        self.theta_init = np.array([0, -2.9])
        self.theta_final = np.array([3,2.9])
        self.tau_max = 1.1
        self.alpha = 0.1
        self.beta_succ = 1.1
        self.beta_fail = 0.5
        self.rho1 = 0.90
        self.lamb = 2
        self.theta_dot = np.array([0,0])


    def M(self):
        M = np.zeros([[],[]])
        M[0,0] = np.sum(self.m) * self.l ** 2
        M[0,1] = self.m[1] * np.prod(self.l) * (np.prod(self.s) + np.prod(self.c))
        M[1,0] = self.m[1] * np.prod(self.l) * (np.prod(self.s) + np.prod(self.c))
        M[1,1] = self.m[1] * self.l ** 2
        return M

    def W(self):
        M = np.zeros([[],[]])
        M[0,1] = self.m[1] * np.prod(self.l) * (np.prod(self.s) + np.prod(self.c)) * self.theta_dot[1]
        M[1,0] = self.m[1] * np.prod(self.l) * (np.prod(self.s) + np.prod(self.c)) * self.theta_dot[0]
        return W 

    def J(self):
        h = self.T / self.N # time interval 
        self.tau[i]
        return h 

    def run(self):
        
        for i in range(T):
