import numpy as np
from scipy import optimize


# can this be written with a bidiagonal matrix?
# def rosenbrock_function(x):
#     bidiag = np.diag(np.ones(len(x))) + np.diag(np.ones(len(x) - 1), 1)
#     return np.sum(np.dot(x, bidiag) ** 2)

# print(rosenbrock_function(np.array([1, 2, 3, 4, 5])))


def booth_function(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def booth_contour(x, y):
    return booth_function(np.array([x, y]))


def sphere_function(x):
    return np.sum(x**2, axis=0)


def sphere_contour(x, y):
    s = sphere_function(np.array([x, y]))
    return s


def rosenbrock_function(x):
    s = 0
    for i in range(len(x) - 1):
        x1 = x[i]
        x2 = x[i + 1]
        s += 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2
    return s


def rosenbrock_contour(x, y):
    return rosenbrock_function(np.array([x, y]))


def matyas_function(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


def matyas_contour(x, y):
    return matyas_function(np.array([x, y]))


def get_test_function(name):
    if name == "booth":
        return booth_function, booth_contour, np.array([0, 6.5]), [-10,10],[-10,10]
    if name == "sphere":
        return sphere_function, sphere_contour
    if name == "rosenbrock":
        return rosenbrock_function, rosenbrock_contour, np.array([2.5, -1.5]), [-2,2],[-4,4]
    if name == "matyas":
        return matyas_function, matyas_contour, np.array([-10.0, -8.0]), [-10,10],[-10,10]


if __name__ == "__main__":
    # booth optimal - [1, 3]
    # rranges = (slice(-10, 10, 0.01), slice(-10, 10, 0.01))
    # resbrute = optimize.brute(booth_function, rranges)
    # print(resbrute)

    # rosenbrock optimal at [1, 1]
    # rranges = (slice(-2, 2, 0.01), slice(-1, 3, 0.01))
    # resbrute = optimize.brute(rosenbrock_function, rranges)
    # print(resbrute)

    # matyas optimal - [0, 0]
    rranges = (slice(-10, 10, 0.01), slice(-10, 10, 0.01))
    resbrute = optimize.brute(matyas_function, rranges)
    print(resbrute)