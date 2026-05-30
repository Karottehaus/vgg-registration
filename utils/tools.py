import numpy as np
from scipy.interpolate import Rbf


def tps_warp(Y, Z, IY, out_shape):
    Y_height, Y_width = IY.shape[:2]
    Z_height, Z_width = out_shape[:2]
    # (Z_x, Z_y) -> Y_x
    backward_x = Rbf(Z[:, 0], Z[:, 1], Y[:, 0], function='thin-plate')
    # (Z_x, Z_y) -> Y_y
    backward_y = Rbf(Z[:, 0], Z[:, 1], Y[:, 1], function='thin-plate')

    Z_i, Z_j = np.mgrid[:Z_height, :Z_width]
    Z_i = Z_i.flatten()
    Z_j = Z_j.flatten()
    Y_i = np.int_(backward_x(Z_i, Z_j))
    Y_j = np.int_(backward_y(Z_i, Z_j))

    keep = np.logical_and(Y_i >= 0, Y_j >= 0)
    keep = np.logical_and(keep, Y_i < Y_height)
    keep = np.logical_and(keep, Y_j < Y_width)
    Y_i, Y_j, Z_i, Z_j = Y_i[keep], Y_j[keep], Z_i[keep], Z_j[keep]

    out_image = np.zeros(out_shape, dtype='uint8')
    out_image[Z_i, Z_j, :] = IY[Y_i, Y_j, :]

    return out_image


def checkerboard(I1, I2, num_tiles):
    assert I1.shape == I2.shape
    height, width, channels = I1.shape
    hi, wi = height // num_tiles, width // num_tiles
    outshape = (hi * num_tiles, wi * num_tiles, channels)

    out_image = np.zeros(outshape, dtype='uint8')
    for i in range(num_tiles):
        h = hi * i
        h1 = h + hi
        for j in range(num_tiles):
            w = wi * j
            w1 = w + wi
            if (i - j) % 2 == 0:
                out_image[h:h1, w:w1, :] = I1[h:h1, w:w1, :]
            else:
                out_image[h:h1, w:w1, :] = I2[h:h1, w:w1, :]

    return out_image
