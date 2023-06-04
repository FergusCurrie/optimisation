import numpy as np


# a = np.array([1, 2, 3])
# b = np.array([4, 5])
# outer_product = np.outer(a, b)


# result = np.outer(a, b)

# # print(result.shape)
# # print(np.outer(b, a).shape)
# print(a[:, np.newaxis].shape)
# print((a[:, np.newaxis] * b[np.newaxis, :]).shape)


def onehot_neighbours(neighbours: np.ndarray, num_field_samples: int) -> np.ndarray:
    z = np.eye(num_field_samples, dtype=np.uint8)[neighbours]
    print(z.shape)
    return np.sum(z, axis=1)


# distance : h*w, num field sampels
# neighbours : h*w, num neighbours
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
# print(f"x shape: {X.shape}")
# ohe = onehot_neighbours(X, 10)
# print(ohe)
# print(ohe.shape)
# print(f"ball park memory: {256 * 256 * 15 * 2000 * 8 * 1e-9} gB")


X1 = np.array([[0, 1, 0], [1, 0, 1]])
X2 = np.array([[1, 2, 3], [4, 5, 6]])

print(X2[X1].shape)


# Using copilot effectivley. 


# alt + [ / ] to cycle copilot suggestions 
# ctrl + enter 

# Write a function that takes in a numpy array and returns the square of the sum
# of the elements in the array. 
def square_of_sum(arr):
    return np.sum(arr) ** 2

