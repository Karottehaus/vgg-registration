from src.vgg_extractor import VGGExtractor
import cv2
from lap import lapjv
import numpy as np
from utils.core import pairwise_distance, pd_expand, gaussian_radial_basis, init_sigma2, match, compute
from src.shape_context import ShapeContext
from settings import HEIGHT, WIDTH, SHAPE, MAX_ITR, TOLERANCE, FREQ, EPSILON, OMEGA, BETA, LAMBD


class ImageRegistration:
    def __init__(self):
        self.height = HEIGHT
        self.width = WIDTH
        self.shape = np.array(SHAPE)
        self.max_itr = MAX_ITR
        self.tolerance = TOLERANCE
        self.freq = FREQ
        self.epsilon = EPSILON
        self.omega = OMEGA
        self.beta = BETA
        self.lambd = LAMBD
        self.vgg_extractor = VGGExtractor()
        self.shape_context = ShapeContext()

    def register(self, IX, IY):

        tolerance = self.tolerance
        freq = self.freq
        epsilon = self.epsilon
        omega = self.omega
        beta = self.beta
        lambd = self.lambd

        # resize image
        Xscale = np.array(IX.shape[:2], dtype=float) / self.shape
        Yscale = np.array(IY.shape[:2], dtype=float) / self.shape
        IX = cv2.resize(IX, (self.width, self.height))
        IY = cv2.resize(IY, (self.width, self.height))

        # extract feature through CNN
        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0).astype(np.float32)

        D1, D2, D3 = self.vgg_extractor(cnn_input)
        D1, D2, D3 = D1.numpy(), D2.numpy(), D3.numpy()

        # flatten
        D1X, D1Y = np.reshape(D1[0], [-1, 256]), np.reshape(D1[1], [-1, 256])
        D2X, D2Y = np.reshape(D2[0], [-1, 512]), np.reshape(D2[1], [-1, 512])
        D3X, D3Y = np.reshape(D3[0], [-1, 512]), np.reshape(D3[1], [-1, 512])

        # normalization
        D1X, D1Y = D1X / np.std(D1X), D1Y / np.std(D1Y)
        D2X, D2Y = D2X / np.std(D2X), D2Y / np.std(D2Y)
        D3X, D3Y = D3X / np.std(D3X), D3Y / np.std(D3Y)

        del D1, D2, D3

        # compute feature space distance
        PD1 = pairwise_distance(D1X, D1Y)
        PD2 = pd_expand(pairwise_distance(D2X, D2Y), 2)
        PD3 = pd_expand(pairwise_distance(D3X, D3Y), 4)
        PD = 1.414 * PD1 + PD2 + PD3

        del D1X, D1Y, D2X, D2Y, D3X, D3Y, PD1, PD2, PD3

        grid = np.array([[i, j] for i in range(28) for j in range(28)], dtype='int32')

        X = np.array(grid, dtype='float32') * 8.0 + 4.0
        Y = np.array(grid, dtype='float32') * 8.0 + 4.0

        # normalize
        X = (X - 112.0) / 224.0
        Y = (Y - 112.0) / 224.0

        # prematch and select points
        init_correspondence, init_quality = match(PD)
        init_tau_max = np.max(init_quality)
        while np.where(init_quality >= init_tau_max)[0].shape[0] <= 128:
            init_tau_max -= 0.01

        init_high_quality_correspondence = init_correspondence[np.where(init_quality >= init_tau_max)]
        num_high_quality_points = init_high_quality_correspondence.shape[0]

        # select prematched feature points
        X, Y = X[init_high_quality_correspondence[:, 1]], Y[init_high_quality_correspondence[:, 0]]
        PD = PD[np.repeat(np.reshape(init_high_quality_correspondence[:, 1], [num_high_quality_points, 1]),
                          num_high_quality_points, axis=1),
        np.repeat(np.reshape(init_high_quality_correspondence[:, 0], [1, num_high_quality_points]),
                  num_high_quality_points,
                  axis=0)]

        N = X.shape[0]
        M = Y.shape[0]
        assert M == N

        # precalculation of feature match
        correspondence, quality = match(PD)

        # calculate theta_hat and delta
        tau_min = np.min(quality)
        tau_max = np.max(quality)
        while np.where(quality >= tau_max)[0].shape[0] <= 0.5 * num_high_quality_points:
            tau_max -= 0.01
        tau = tau_max
        delta = (tau_max - tau_min) / 10.0

        X_shape_context = self.shape_context.compute(X)

        # initialization
        Z = Y.copy()
        GRB = gaussian_radial_basis(Y, beta)
        W = np.zeros([M, 2])
        sigma2 = init_sigma2(X, Y)

        Q = 0
        dQ = float('Inf')
        itr = 1
        prior_matrix = None

        # registration process
        while itr < self.max_itr and abs(dQ) > tolerance and sigma2 > 1e-4:
            Z_old = Z.copy()
            Q_old = Q

            # for every k iterations
            if (itr - 1) % freq == 0:
                # compute C^{conv}_{\theta}
                high_quality_correspondence = correspondence[np.where(quality >= tau)]
                matched_feature_distances = PD[high_quality_correspondence[:, 0], high_quality_correspondence[:, 1]]

                max_matched_feature_distance = np.max(matched_feature_distances)
                if max_matched_feature_distance > 0:
                    matched_feature_distances = matched_feature_distances / max_matched_feature_distance

                L = np.ones([M, N])
                L[high_quality_correspondence[:, 0], high_quality_correspondence[:, 1]] = matched_feature_distances

                # compute C^{geo}_{\theta}
                Z_shape_context = self.shape_context.compute(Z)
                shape_context_cost = self.shape_context.cost(Z_shape_context, X_shape_context)

                # compute C
                L = L * shape_context_cost

                # linear assignment
                C = lapjv(L)[1]

                # prior probability matrix
                prior_matrix = np.ones_like(PD) * (1.0 - epsilon) / N
                prior_matrix[np.arange(C.shape[0]), C] = 1.0
                prior_matrix = prior_matrix / np.sum(prior_matrix, axis=1)

                tau = tau - delta
                if tau < tau_min:
                    tau = tau_min

            # compute minimization
            posterior_matrix, D_Y, sum, trace_sum, Q = compute(X, Z_old, prior_matrix, sigma2, omega)
            Q = Q + lambd / 2 * np.trace(W.T @ GRB @ W)

            # update variables
            t1 = GRB + lambd * sigma2 * np.linalg.inv(D_Y)
            t2 = np.linalg.inv(D_Y) @ posterior_matrix @ X - Y
            W = np.linalg.inv(t1) @ t2

            sigma2 = trace_sum / (2.0 * sum)
            omega = 1 - (sum / N)

            if omega > 0.99:
                omega = 0.99
            if omega < 0.01:
                omega = 0.01

            Z = Y + GRB @ W
            lambd = lambd * 0.95

            if lambd < 0.1:
                lambd = 0.1

            dQ = Q - Q_old
            itr = itr + 1

        print(f'[Registration] itr={itr:03d}, Q={Q:.5f}, tau={tau:.5f}')
        return ((X * 224.0) + 112.0) * Xscale, ((Y * 224.0) + 112.0) * Yscale, ((Z * 224.0) + 112.0) * Xscale
