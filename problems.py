'''
Script for generating test problems 
'''

import numpy as np
import cvxpy as cp

class Program():
    def __init__(self, f, constraints):
        self.f
        self.constraints = constraints
        pass
    
    def objective(self, x) -> np.float32:
        return self.f(x)
    
    def check_constraints(self) -> Bool:
        return 
    
    
class LinearConstraint():
    def __init__(self, a, b):
        self.a = a
        self.b = b 
        
    def check_constraint(self, x):
        if a @ x <= b:
            return True
        return False
    
class LinearConstraintTwoParameter():
    def __init__(self, a, b):
        self.a = a 
        
    def check_constraint(self, x1, x2):
        '''
        x1 : parameter 1 
        x2 : parameter 2 
        '''
        if a @ x1 <= x2:
            return True
        return False

class LinearObjective():
    def __init__(self, a):
        self.a = a
        
    def __call__(self, x):
        return self.a @ x 
    

def linear_program(m=100, n=20):
    A = np.random.normal(0,1,size=(m,n))
    b = np.random.uniform(0,1,size=(m))
    c = -A.T @ np.random.uniform(0,1,size=(m)) # so problem instance is bounded 
    return A,b,c

def quadratic_program(m=100, n=20):
    # to check 
    A = np.random.normal(0,1,size=(m,n))
    b = np.random.uniform(0,1,size=(m))
    Q = np.random.normal(0,1,size=(n,n))
    Q = Q.T @ Q
    c = -A.T @ np.random.uniform(0,1,size=(m)) # so problem instance is bounded 
    return Q,A,b,c

def piecewise_affine(m=100, n=20):
    A = np.random.normal(0,1,size=(m,n))
    b = np.random.uniform(0,1,size=(m))
    return A,b

def minimum_cardinality(m=100, n=30):
    # see page 22 - https://stanford.edu/class/ee364b/lectures/bb_slides.pdf
    # objective is 1^T z
    # variables are x and z, z_i in {0,1}
    # upper bounds from 
    A = np.random.normal(0,1,size=(m,n))
    b = np.random.uniform(0,1,size=(m))
    
    f = LinearObjective(a=np.ones(n))
    LinearConstraints(a, b) # 
    Program(f=f, constraints=constraints)
    