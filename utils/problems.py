"""
Script for generating test problems. 

"""

import numpy as np
import cvxpy as cp
from abc import ABC, abstractmethod  # abstract base class


class Program(ABC):
    @abstractmethod
    def objective(self, x):
        pass

    @abstractmethod
    def check_constraints(self, x):
        pass


class LinearProgram(Program):
    def __init__(self, m, n, p):
        self.m, self.n, self.p = m, n, p
        self.A, self.b, self.c, self.G, self.h = random_linear_program(
            self.m, self.n, self.p
        )

    def objective(self, x):
        return self.c.T @ x

    def check_constraints(self, x):
        # Check inequality violations
        if np.sum(self.A @ x > self.b) > 0:
            return False
        # CHeck equality violations
        if self.p > 0:
            if np.sum(self.G @ x != self.h) > 0:
                return False
        return True


class MixedIntegerQuadraticProgram(Program):
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.A, self.b = random_mixed_interger_quadratic_program(self.m, self.n)

    def objective(self, x):
        return np.linalg.norm(self.A @ x - self.b, ord=2) ** 2

    def check_constraints(self, x):
        return np.all(np.equal(np.mod(x, 1), 0))


class PiecewiseAffineProgram(Program):
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.A, self.b = random_piecewise_affine(self.m, self.n)

    def objective(self, x):
        return np.max(self.A @ x + self.b)

    def check_constraints(self, x):
        return True


def random_linear_program(m=100, n=20, p=5):
    """
    min c^T x
    s.t. Ax <= b
    """
    A = np.random.normal(0, 1, size=(m, n))
    b = np.random.uniform(0, 1, size=(m))
    c = -A.T @ np.random.uniform(0, 1, size=(m))  # so problem instance is bounded
    G = np.random.randn(p, n)
    h = np.random.randn(p)
    return A, b, c, G, h


def random_mixed_interger_quadratic_program(m=100, n=20):
    """
    differentiable, non-convex, mixed-integer quadratic program
    min ||Ax-b||_2^2
    s.t. x in Zn (integer)
    """
    A = np.random.rand(m, n)
    b = np.random.randn(m)
    return A, b


def random_piecewise_affine(m=100, n=20):
    """
    f(x) = max(a_1^T x + b_1, ... , a_m^T x + b_m)
    convex, non-differntiable piecewise affine function
    """
    A = np.random.normal(0, 1, size=(m, n))
    b = np.random.uniform(0, 1, size=(m))
    return A, b


def random_quadratic_program(m=100, n=20, p=5):
    """
    min x.T@Px + q.T@x + r
    s.t. Gx <= h
    s.t. Ax = b
    """
    np.random.seed(1)
    P = np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    r = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)
    return P, q, G, h, A, b, r


def random_second_order_cone_program():
    """
    min f.T@c
    s.t. ||A_i @ x + b_i||_2 <= c_i^T @ x + d_i, i=1,...,m
    s.t. Fx = g

    Convex, differentiable, inequality and equality constraints.
    see : https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/socp.ipynb#scrollTo=E2RDY1Yvt7AP
    """
    # Generate a random feasible SOCP.
    m = 3
    n = 10
    p = 5
    n_i = 5
    np.random.seed(2)
    f = np.random.randn(n)
    A = []
    b = []
    c = []
    d = []
    x0 = np.random.randn(n)
    for i in range(m):
        A.append(np.random.randn(n_i, n))
        b.append(np.random.randn(n_i))
        c.append(np.random.randn(n))
        d.append(np.linalg.norm(A[i] @ x0 + b, 2) - c[i].T @ x0)
    F = np.random.randn(p, n)
    g = F @ x0


def random_SDP():
    """
    Convex, differentiable
    min tr(CX)
    s.t. tr(A_i @ X) = b_i, i=1,...,p
    s.t. X >= 0

    Random semidefinite program.
    see : https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/sdp.ipynb#scrollTo=Gjexqe1wt7Kv
    """
    # Generate a random SDP.
    n = 3
    p = 3
    np.random.seed(1)
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())
    return C, A, b


# def minimum_cardinality(m=100, n=30):
#     # see page 22 - https://stanford.edu/class/ee364b/lectures/bb_slides.pdf
#     # objective is 1^T z
#     # variables are x and z, z_i in {0,1}
#     # upper bounds from
#     A = np.random.normal(0, 1, size=(m, n))
#     b = np.random.uniform(0, 1, size=(m))

#     f = LinearObjective(a=np.ones(n))
#     LinearConstraint(A, b)
#     Program(f=f, constraints=constraints)
