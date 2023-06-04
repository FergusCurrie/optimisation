import numpy as np
from scipy import optimize


def booth_function(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def sphere_function(x):
    return np.sum(x**2, axis=0)


def rosenbrock_function(x):
    s = 0
    for i in range(len(x) - 1):
        x1 = x[i]
        x2 = x[i + 1]
        s += 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2
    return s


def matyas_function(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


def problem1(x):
    """
    0.5x1^2 + 9/2x2^2
    """
    # d = np.array([0.5, 9 / 2])[np.newaxis][np.newaxis].T * np.ones_like(x)
    return np.tensordot(np.array([0.5, 9 / 2]), x**2, axes=([0], [0]))
