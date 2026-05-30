import numpy as np
from math import log


def pairwise_distance(X, Y):
    N = X.shape[0]
    M = Y.shape[0]

    distance_matrix = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            distance_matrix[i, j] = np.linalg.norm(X[j] - Y[i])
    return distance_matrix


def pd_expand(distance_matrix, k):
    N0 = int(np.sqrt(distance_matrix.shape[0]))
    N1 = k * N0
    L0, L1 = N0 ** 2, N1 ** 2
    index_map = np.kron(np.arange(L0).reshape([N0, N0]), np.ones([k, k], dtype='int32'))
    i = np.repeat(index_map.reshape([L1, 1]), L1, axis=1)
    j = np.repeat(index_map.reshape([1, L1]), L1, axis=0)
    return distance_matrix[i, j]


def gaussian_radial_basis(Y, beta=2.0):
    distance_matrix = pairwise_distance(Y, Y)
    return np.exp(-0.5 * np.power(distance_matrix / beta, 2))


def init_sigma2(X, Y):
    N = X.shape[0]
    M = Y.shape[0]
    t1 = M * np.trace(X @ X.T)
    t2 = N * np.trace(Y @ Y.T)
    t3 = 2.0 * np.sum(X @ Y.T)
    return (t1 + t2 - t3) / (M * N * 2.0)


def match(distance_matrix):
    row_indices = np.arange(distance_matrix.shape[0])
    nearest_indices = np.argmin(distance_matrix, axis=1)
    correspondence = np.array([row_indices, nearest_indices]).T
    nearest_distances = distance_matrix[row_indices, nearest_indices]
    mask = np.zeros_like(distance_matrix)
    mask[row_indices, nearest_indices] = 1
    masked = np.ma.masked_array(distance_matrix, mask)
    second_nearest_distances = np.amin(masked, axis=1)
    return correspondence, np.array(second_nearest_distances / nearest_distances)


def compute(X, Z, prior_matrix, sigma2, omega):
    N = X.shape[0]

    diff = X[None, :, :] - Z[:, None, :]
    gaussian_likelihood = np.exp(-(1 / (2 * sigma2)) * np.sum(diff ** 2, axis=2))

    weighted_likelihood = (1.0 - omega) * prior_matrix * gaussian_likelihood

    normalization = np.sum(weighted_likelihood, axis=0) + omega / N

    posterior_matrix = weighted_likelihood / normalization[None, :]

    sum = np.sum(posterior_matrix)
    D_X = np.diag(np.sum(posterior_matrix.T, axis=1))
    D_Y = np.diag(np.sum(posterior_matrix, axis=1))

    t1 = np.trace(X.T @ D_X @ X)
    t2 = np.trace(X.T @ posterior_matrix.T @ Z)
    t3 = np.trace(Z.T @ D_Y @ Z)
    trace_sum = t1 - 2.0 * t2 + t3
    Q = sum * log(sigma2) + trace_sum / (2.0 * sigma2)

    return posterior_matrix, D_Y, sum, trace_sum, Q
