'''
Iterative method to solve non-convex problems 

Willl use the example from slides. 

https://web.stanford.edu/class/ee364b/lectures/seq_slides.pdf

dot over variable = derivative of variable "dot notation" aka "newtonian notation"

TODO:
- add variations from notes
- gif of the control 
'''

import cvxpy as cvx 
import numpy as np


class TwoLinkSystem:
    def __init__(self):
        # Parameters 
        self.m = np.array([1,5])
        self.l = np.array([1,1])
        self.N = 40
        self.T = 10
        self.theta_init = np.array([0, -2.9])
        self.theta_final = np.array([3,2.9])
        self.tau_max = 1.1
        self.time_0()

        
    def time_0(self):
        self.theta = self.theta_init
        self.s = np.sin(self.theta)
        self.c = np.cos(self.theta)

    def partial_derivatives_theta(theta):
        pass

    def M(self):
        d_dtheta1, d_dtheta2 = self.partial_derivatives_theta(self.theta)
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
        return M

    def J(self):
        h = self.T / self.N # time interval 
        # self.tau[i]
        return h 

    def run(self):
        
        for i in range(self.T):
            pass
