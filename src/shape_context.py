import math
import numpy as np
from utils.core import pairwise_distance
from settings import NUM_BINS_R, NUM_BINS_THETA, R_INNER, R_OUTER


class ShapeContext:

    def __init__(self, num_bins_r=NUM_BINS_R, num_bins_theta=NUM_BINS_THETA, r_inner=R_INNER, r_outer=R_OUTER):
        self.num_bins_r = num_bins_r
        self.num_bins_theta = num_bins_theta
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.num_bins = num_bins_r * num_bins_theta

    def _get_theta(self, point_a, point_b):
        return math.atan2(
            point_b[1] - point_a[1],
            point_b[0] - point_a[0]
        )

    def _get_theta_matrix(self, points):
        num_points = len(points)
        theta_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(num_points):
                theta_matrix[i, j] = self._get_theta(points[i], points[j])

        return theta_matrix

    def compute(self, points):
        num_points = len(points)

        distance_matrix = pairwise_distance(points, points)
        mean_distance = distance_matrix.mean()
        normalized_distance = distance_matrix / mean_distance

        r_bin_edges = np.logspace(
            np.log10(self.r_inner),
            np.log10(self.r_outer),
            self.num_bins_r
        )

        r_bin_indices = np.zeros((num_points, num_points), dtype=int)

        for r_bin_edge in r_bin_edges:
            r_bin_indices += normalized_distance < r_bin_edge

        valid_distance_mask = r_bin_indices > 0

        theta_matrix = self._get_theta_matrix(points)

        positive_theta_matrix = theta_matrix + 2 * math.pi * (theta_matrix < 0)

        theta_bin_indices = (
                1 + np.floor(
            positive_theta_matrix / (2 * math.pi / self.num_bins_theta)
        )
        ).astype(int)

        shape_context = np.zeros((num_points, self.num_bins))

        for i in range(num_points):
            histogram = np.zeros((self.num_bins_r, self.num_bins_theta))

            for j in range(num_points):
                if valid_distance_mask[i, j]:
                    r_bin = r_bin_indices[i, j] - 1
                    theta_bin = theta_bin_indices[i, j] - 1

                    histogram[r_bin, theta_bin] += 1

            shape_context[i] = histogram.reshape(self.num_bins)

        return shape_context

    def _cost(self, histogram_a, histogram_b):
        cost = 0
        for bin_idx in range(self.num_bins):
            denominator = histogram_a[bin_idx] + histogram_b[bin_idx]

            if denominator:
                cost += (
                        (histogram_a[bin_idx] - histogram_b[bin_idx]) ** 2
                        / denominator
                )

        return cost * 0.5

    def cost(self, shape_context_a, shape_context_b):
        num_a, _ = shape_context_a.shape
        num_b, _ = shape_context_b.shape

        cost_matrix = np.zeros((num_a, num_b))

        for i in range(num_a):
            for j in range(num_b):
                cost_matrix[i, j] = self._cost(
                    shape_context_a[i] / num_a,
                    shape_context_b[j] / num_b
                )

        return cost_matrix
