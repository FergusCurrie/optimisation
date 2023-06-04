"""

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OptimisationExperiment:
    def __init__(self, f) -> None:
        self.history = []
        self.f = f

    def update(self, x):
        self.history.append(x)

    def plot_function_vs_iteration_nd(self, ax):
        ax.scatter(
            np.arange(len(self.history)),
            np.array([self.f(q) for q in self.history]),
            color="red",
        )
        ax.plot(
            np.arange(len(self.history)),
            np.array([self.f(q) for q in self.history]),
            color="red",
        )
        ax.set_xlabel("iterations")
        ax.set_ylabel("function value")

    def contour_two_d(self, x, y):
        return self.f(np.array([x, y]))

    def graph_convergence_two_d(self, ax):
        for i, x in enumerate(self.history):
            # ax.arrow(x=x[1], y=x[0], dx=scaled_step[1], dy=scaled_step[0], color="red", head_width=0.3, head_length=0.3)
            if i == 0:
                ax.scatter(x[1], x[0], color="blue")
            else:
                ax.scatter(x[1], x[0], color="red")
        for i in range(0, len(self.history)):
            if i + 1 < len(self.history):
                x2 = self.history[i + 1]
            else:
                x2 = self.history[i]
            x1 = self.history[i]
            ax.plot([x1[1], x2[1]], [x1[0], x2[0]], color="red")

    def graph_convergence_three_d(self, ax):
        for i, x in enumerate(self.history):
            # ax.arrow(x=x[1], y=x[0], dx=scaled_step[1], dy=scaled_step[0], color="red", head_width=0.3, head_length=0.3)
            y = self.f(x)
            if i == 0:
                ax.scatter(x[1], x[0], y, color="blue")
            else:
                ax.scatter(x[1], x[0], y, color="red")

        for i in range(0, len(self.history)):
            if i + 1 < len(self.history):
                x2 = self.history[i + 1]
            else:
                x2 = self.history[i]
            x1 = self.history[i]
            y1 = self.f(x1)
            y2 = self.f(x2)

            ax.plot3D([x1[1], x2[1]], [x1[0], x2[0]], [y1, y2], color="red", zorder=1)

    def graph_function_contour_two_d(self, ax):
        x1 = np.linspace(-10, 10, 500)
        x2 = np.linspace(-10, 10, 500)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.contour_two_d(X1, X2)
        contour = ax.contourf(
            X1, X2, Z, cmap="viridis", levels=np.linspace(0, 200, 30), extend="max"
        )
        return contour

    def plot_three_d(self, ax):
        # Create data for the plot
        x1 = np.linspace(-3, 5, 100)
        x2 = np.linspace(-3, 5, 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.array([X1, X2])
        Z = self.f(X)

        # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X1, X2, Z, cmap="viridis", zorder=-1)

        # Add level curves
        ax.contour(X1, X2, Z, levels=10, offset=ax.get_zlim()[0], cmap="coolwarm")

        # Set labels and title
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")

    def plot_descent_three_d(self, ax):
        self.plot_three_d(ax)
        # self.graph_convergence_three_d(ax)

    def plot_descent_two_d(self, ax):
        contour = self.graph_function_contour_two_d(ax)
        self.graph_convergence_two_d(ax)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        best = self.history[-1]
        ax.set_title(f"optimal=[{best[0]:.2f},{best[1]:.2f}]")
