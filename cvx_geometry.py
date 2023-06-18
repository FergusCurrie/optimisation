"""
Start collecting some useful functions for graphing convex geometry
"""

import numpy as np
import matplotlib.pyplot as plt


def orthogonal_vec_2d(v):
    v1 = 1
    v2 = -(v[0] * v1) / v[1]
    return np.array([v1, v2])


def unit_vector(v):
    return v / np.linalg.norm(v, ord=2)


def plot_ellipsoid(ax, B, d):
    """
    Ellpsoid is {Bu + d | ||u||_2 <= 1}.
    Sample 1000 points from the unit sphere, transform by B, shift by d.
    """
    u = np.random.randn(2, 1000)
    u = u / np.linalg.norm(u, 2, axis=0)
    u_ = (B @ u) + np.array([d for _ in range(1000)]).T
    ax.scatter(*u_, s=1)


def plot_hyperplane(ax, a, b, offset=np.array([0, 0])):
    """
    doesn't offset the hyperplane from the origin

    plots hyperplane to normal vector (a) at a point (offset)
    """
    ax.arrow(
        *offset,
        *a,
        head_width=0.2,
        head_length=0.2,
        length_includes_head=True,
        color="red"
    )
    x0 = np.linspace(-5, 5, 10)
    x1 = (-a[0] * (x0 - offset[0])) / a[1] + offset[1]

    ax.plot(x0, x1)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_aspect("equal")
    ax.grid()


# d# ef x(P, f):


def unit_vector(v):
    return v / np.linalg.norm(v, ord=2)


def plot_hyperplane_x_proj(ax, a, x_proj):
    """
    doesn't offset the hyperplane from the origin
    """
    ax.arrow(
        *x_proj,
        *a,
        head_width=0.2,
        head_length=0.2,
        length_includes_head=True,
        color="red"
    )
    x0 = np.linspace(-5, 5, 10)
    x1 = (-a[0] * (x0 - x_proj[0])) / a[1] + x_proj[1]

    ax.plot(x0, x1)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_aspect("equal")
    ax.grid()


def plot_hyperplane_ab(ax, a, b):
    """
    doesn't offset the hyperplane from the origin
    """
    a_unit = unit_vector(a)
    x_proj = a_unit * b
    plot_hyperplane_x_proj(ax, a, x_proj)
