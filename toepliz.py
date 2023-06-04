import numpy as np
from scipy import linalg

x = np.array([1, 2, 3, 4, 5])
# x = np.array([0, 0, 1, 0, 0])

t = linalg.toeplitz(x)


# x = ...
bidiag = np.diag(np.ones(len(x))) + np.diag(np.ones(len(x) - 1), 1)
print(np.sum(np.dot(x, bidiag) ** 2))


s = 0
for i in range(len(x) - 1):
    x1 = x[i]
    x2 = x[i + 1]
    s += (x1 + x2) ** 2


print(s)
print(x[::2])  #
