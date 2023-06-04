import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from jax.scipy.linalg import lu_solve
from jax import grad, jit, vmap, jacfwd, jacrev
from jax import random, hessian
import cvxpy
import matplotlib.pyplot as plt
from test_problem import get_test_function
import scipy

"""
Object for tracking experiment. 
Easy switch between functions (variable limits )
how does mesh grid work 

def hessian(fun):
    return jit(jacrev(jacfwd(fun)))
"""

# https://en.wikipedia.org/wiki/Test_functions_for_optimization


class OptimisationExperiment:
    def __init__(self, method, problem, search_method="btl") -> None:
        (
            self.func,
            self.contour,
            self.start_point,
            self.x_bounds,
            self.y_bounds,
        ) = get_test_function(problem)
        self.method = method
        self.search_method = search_method
        self.problem = problem

    def graph_function(self, ax):
        y = np.linspace(self.y_bounds[0], self.y_bounds[1], 500)
        x = np.linspace(self.x_bounds[0], self.x_bounds[1], 500)
        X, Y = np.meshgrid(x, y)
        Z = self.contour(X, Y)
        # LEVELS IS CRITICAL FOR A GOOD GRAPH
        contour = ax.contourf(
            X, Y, Z, cmap="viridis", levels=np.linspace(0, 200, 30), extend="max"
        )
        return contour

    def newton_method(self, x_start):
        """
        In newtons method the step size is just the inverse heesian
        """
        x_history = [x_start]
        x = x_start
        hf = hessian(self.func)
        gf = grad(self.func)
        for i in range(5):
            H = hf(x)  # hessian at x
            g = gf(x)  # grad at x
            # find inverse hessian for step
            H_inv = jax.numpy.linalg.inv(H)

            step = -1 * jnp.dot(H_inv, g)
            # line search
            t = 1  # line_search(x, step)

            # update
            x = x + t * step
            x_history.append(x.copy())
        return x_history

    def exact_line_search(self, x, step):
        t = np.linspace(0.0001, 100, 100)
        x_ = x + np.outer(t, step)
        assert np.shape(x_) == (100, 2)
        tmin = t[np.argmin(np.array([self.func(q) for q in x_]))]
        return tmin

    def backtracking_line_search(self, x, search_direction, alpha=0.1, beta=0.5):
        """
        Inexxact line search

        alpha in (0,0.5)
        beta in (0,1)
        """
        t = 1
        lhs = self.func(x + t * search_direction)
        rhs = self.func(x) + alpha * t * np.dot(grad(self.func)(x), search_direction)
        while lhs > rhs:
            lhs = self.func(x + t * search_direction)
            rhs = self.func(x) + alpha * t * np.dot(
                grad(self.func)(x), search_direction
            )
            t = beta * t
        return t

    def gradient_descent(self, x_start):
        x_history = [x_start]
        g = grad(self.func)
        for i in range(40):
            x_curr = x_history[-1]
            search_direction = -1 * g(x_curr)
            if self.search_method == "btl":
                step_size = self.backtracking_line_search(x_curr, search_direction)
            if self.search_method == "els":
                step_size = self.exact_line_search(x_curr, search_direction)
            x_next = x_curr + step_size * search_direction
            x_history.append(x_next.copy())
            # if np.linalg.norm(search_direction, ord=2) < 0.0001:
            #     print(f"break on {i}")
            #     break

        return x_history

    def plot_descent(self, x_history):
        fig, ax = plt.subplots(1, 2, figsize=(30, 15))
        contour = self.graph_function(ax[0])

        if 1:  # plot convergence
            for i, x in enumerate(x_history):
                # ax.arrow(x=x[1], y=x[0], dx=scaled_step[1], dy=scaled_step[0], color="red", head_width=0.3, head_length=0.3)
                if i == 0:
                    ax[0].scatter(x[1], x[0], color="blue")
                else:
                    ax[0].scatter(x[1], x[0], color="red")
            for i in range(0, len(x_history)):
                if i + 1 < len(x_history):
                    x2 = x_history[i + 1]
                else:
                    x2 = x_history[i]
                x1 = x_history[i]
                ax[0].plot([x1[1], x2[1]], [x1[0], x2[0]], color="red")

        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")
        best = x_history[-1]
        ax[0].set_title(f"optimal=[{best[0]:.2f},{best[1]:.2f}]")
        cbar = fig.colorbar(contour)

        ax[1].scatter(
            np.arange(len(x_history)),
            np.array([self.func(q) for q in x_history]),
            color="red",
        )
        ax[1].plot(
            np.arange(len(x_history)),
            np.array([self.func(q) for q in x_history]),
            color="red",
        )
        ax[0].set_xlabel("iterations")
        ax[0].set_ylabel("function value")
        fig.savefig(f"outputs/{self.method}_{self.search_method}_{self.problem}.png")

    def run(self, x_start):
        if self.method == "gradient_descent":
            x_history = self.gradient_descent(x_start)
        if self.method == "newton_method":
            x_history = self.gradient_descent(x_start)
        self.plot_descent(x_history)


# booth optimal = [1. 3.]?
# matyas optimal = [0. 0.]?
# rosenbrock optimal at [1., 1.]

for problem in ["rosenbrock", "matyas", "booth"]:
    for method in ["gradient_descent", "newton_method"]:
        for search_method in ["btl", "els"]:
            exp = OptimisationExperiment(method, problem, search_method)
            exp.run(np.array([0.0, -1.0]))
