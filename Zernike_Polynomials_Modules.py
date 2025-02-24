
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize

input_zernike = [(1, -1),
                 (1, 1),
                 (2, -2),
                 (2, 0),
                 (2, 2),
                 (3, -3),
                 (3, -1),
                 (3, 1),
                 (3, 3),
                 (4, -4),
                 (4, -2),
                 (4, 0),
                 (4, 2),
                 (4, 4),
                 (5, -5),
                 (5, -3),
                 (5, -1),
                 (5, 1),
                 (5, 3),
                 (5, 5),
                 (6, -6),
                 (6, -4),
                 (6, -2),
                 (6, 0),
                 (6, 2),
                 (6, 4),
                 (6, 6)]

def show_profile(profile, x_grid = None, y_grid = None):
    plt.ion()
    if x_grid is None or y_grid is None:
        size_y, size_x = profile.shape
        x_grid, y_grid = np.meshgrid(np.arange(size_x), np.arange(size_y))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, profile, cmap="viridis")
    plt.tight_layout()
    plt.show()
    plt.ioff()

def mean_gradient(image):
    """
    Calculate the average gradient of a 2D array
    :param image:
    :return: mean absolute gradient values
    """
    image[np.isnan(image)] = 0
    gradient = np.gradient(np.array(image))
    abs_grad = np.sqrt(gradient[0]**2 + gradient[1]**2)
    return np.mean(abs_grad)

def normalization(data):
    """
    Normalize the data into range [-1, 1]
    :param data:
    :return: normalized data
    """
    data = np.array(data)
    nan_index= np.isnan(data)
    data[nan_index] = 0.
    max_data = np.max(data)
    min_data = np.min(data)
    normalized_data = 2 * (data - min_data) / (max_data - min_data) - 1
    normalized_data[nan_index] = np.nan

    return normalized_data

def kronecker_delta(i, j):
    """
    Calculate the kronecker delta
    :param i: tablet
    :param j: response
    :return: kronecker delta
    """
    i = np.asarray(i)
    j = np.asarray(j)

    delta = np.where(i == j, 1, 0)
    return delta

def min_circle(coord, scale = 1):
    """
    Find the minimum circle (center and radius) that encircling all the input coordinates
    :param coord: a set of coordinates in (x, y) format
    :param scale: scale the radius
    :return: center position and the radius
    """
    def dist(cent, coords):
        return np.max(np.linalg.norm(coords - cent, axis=1))

    initial_center = np.mean(coord, axis=0)
    result = minimize(dist, initial_center, args=(coord,))
    center, radius = result.x, result.fun
    return center, radius*scale

def image_padding_for_circular_area(image, ref_circle, padding_scale = 0.2, cut = False):
    """
    Padding and clipping the image
    :param image: original image
    :param ref_circle: minimum circle for the centroids
    :param padding_scale: a scale with respect to the ref_circle's radius for padding
    :param cut: chose if ti cut the circular area out
    :return: processed image
    """
    inner_cut = []
    center = ref_circle[0]
    radius = ref_circle[1]
    r = int(np.ceil(radius))
    r_pad = int(r * padding_scale)
    before_shape = np.array(image.shape)
    if np.logical_and(1 < center[1] - r and center[1] + r < before_shape[1], 1 < center[0] - r and center[0] + r < before_shape[0]):
        padded_image = image
        print('No padding applied for this one.')
        top_cut, bot_cut = int(np.floor(center[0])) - r, int(np.floor(center[0])) + r
        lft_cut, rgt_cut = int(np.floor(center[1])) - r, int(np.floor(center[1])) + r
        inner_cut = padded_image[top_cut:bot_cut, lft_cut:rgt_cut]
        return inner_cut

    else:
        padded_image = np.zeros((before_shape[0] + 2 * r_pad, before_shape[1] + 2 * r_pad))
        top_inner, bot_inner = r_pad, r_pad + before_shape[0]
        lft_inner, rgt_inner = r_pad, r_pad + before_shape[1]
        padded_image[top_inner:bot_inner, lft_inner:rgt_inner] = image
        if cut is False:
            return padded_image
        if cut:
            top_cut, bot_cut = r_pad + int(np.floor(center[0])) - r, r_pad + int(np.floor(center[0])) + r
            lft_cut, rgt_cut = r_pad + int(np.floor(center[1])) - r, r_pad + int(np.floor(center[1])) + r
            inner_cut = padded_image[top_cut:bot_cut, lft_cut:rgt_cut]
        return inner_cut

class Zernike:
    def __init__(self):
        self.gradient_shift = None
        self.zern_profile = None
        self.x = None
        self.y = None
        self.fMat = {}

    def reset(self):
        self.gradient_shift = None
        self.zern_profile = None
        self.x = None
        self.y = None
        self.fMat = {}
        pass

    def calc_zern_cof(self, n, m, scaling = 1, N_fac = False):
        """
        Calculate the coefficients for a given order of zernike polynomial defined with (n, m)
        :param n: -
        :param m: -
        :param scaling: the amplitude of this zernike
        :param N_fac: enable if need to normalize the cof
        :return: a dictionary containing the coefficients for all the items in (x_power, y_power)
        """
        n = int(n)
        m = int(m)
        alpha = int((n - abs(m))/2)
        beta = int(np.ceil(abs(m)/2))
        gamma = int(np.ceil(abs(m)-1)/2)
        N_fac = np.sqrt(2*(n+1/(1 + kronecker_delta(0,m)))) if N_fac is True else 1
        self.fMat = {}
        for i in range(alpha + 1):
            for j in range(alpha - i + 1):
                if m >= 0:
                    for k in range(beta + 1):
                        A = (Zernike.fact(n - i))/(Zernike.fact(i)*Zernike.fact(int(0.5*(n + abs(m)) - i))*Zernike.fact(int(0.5*(n - abs(m)) - i)))
                        A *= (-1)**(i + k)
                        C_1 = A * Zernike.binorm(alpha - i, j) * Zernike.binorm(abs(m), 2*k)
                        xpow = 2 * (j + k)
                        ypow = n - 2 * (i + j + k)
                        if (xpow, ypow) not in self.fMat:
                            self.fMat[(xpow, ypow)] = 0.0
                        self.fMat[(xpow, ypow)] += C_1 * scaling * N_fac
                else:
                    for k in range(gamma + 1):
                        A = (Zernike.fact(n - i))/(Zernike.fact(i)*Zernike.fact(int(0.5*(n + abs(m)) - i))*Zernike.fact(int(0.5*(n - abs(m)) - i)))
                        A *= (-1) ** (i + k)
                        C_2 = A * Zernike.binorm(alpha - i, j) * Zernike.binorm(abs(m), 2 * k + 1)
                        xpow = 2 * (j + k)  + 1
                        ypow = n - 2 * (i + j + k)  - 1
                        if (xpow, ypow) not in self.fMat:
                            self.fMat[(xpow, ypow)] = 0.0
                        self.fMat[(xpow, ypow)] += C_2 * scaling * N_fac
        self.fMat = {k: v for k, v in self.fMat.items() if v != 0}
        return self.fMat

    def zern_from_cof(self, cof_dict, ref_padded_img, ref_circle, cut = True, normalize = False):
        """
        Calculate the height profile of a given zernike cof dict
        :param normalize: rescale the zernike profile into [-1, 1]
        :param cof_dict: -
        :param ref_padded_img: padded image as a reference for size
        :param ref_circle: reference minimum circle to define the unit circle
        :param cut: to set values outside the unit circle to NaN
        :return: height profile and Cartesian grid
        """

        radius = ref_circle[1]
        r = int(np.ceil(radius))
        [height, width] = ref_padded_img.shape
        sc_x = width/(2*r)
        sc_y = height/(2*r)
        x = np.linspace(-1*sc_x, 1*sc_x, width)
        y = np.linspace(-1*sc_y, 1*sc_y, height)
        X, Y = np.meshgrid(x, y)
        self.zern_profile = np.zeros_like(X, dtype = np.float64)
        for (x_pow, y_pow), factor in cof_dict.items():
            factor = np.float64(factor)
            self.zern_profile += factor * (X ** x_pow * Y ** y_pow)

        if cut:
            dist = X ** 2 + Y ** 2
            self.zern_profile[dist > 1] = None
        else:
            dist = X ** 2 + Y ** 2
            self.zern_profile[dist > 1] = 0.

        if normalize:
            self.zern_profile = normalization(self.zern_profile)

        return self.zern_profile, X, Y

    def zern_local_gradient(self, zern_profile, ref_coord, pitch = 150e-6, pixel_size = 4.8e-6):
        """
        Calculate the local gradient in the normalized Zernike profile, area defined by the reference coordinates and the window size
        :param pixel_size: camera's pixel size
        :param pitch: pitch of the lens-array
        :param zern_profile: zernike profile to be processed
        :param ref_coord: reference coordinates for calculating local gradients
        :return: local averaged gradient of the profile at ref_coord
        """
        window_size = np.ceil(pitch/(2*pixel_size))
        gradient_x, gradient_y = np.gradient(zern_profile)
        self.gradient_shift = []
        for j in ref_coord:
            top_w, bot_w = int(np.floor(j[0] - window_size)), int(np.floor(j[0] + window_size))
            lft_w, rgt_w = int(np.floor(j[1] - window_size)), int(np.floor(j[1] + window_size))
            avg_x = np.mean(gradient_x[top_w:bot_w, lft_w:rgt_w])
            avg_y = np.mean(gradient_y[top_w:bot_w, lft_w:rgt_w])
            self.gradient_shift.extend([avg_x, avg_y])
        # self.gradient_shift /= np.max(abs(np.array(self.gradient_shift)))
        if np.isnan(self.gradient_shift).any(): print("nan found in zernike coordinates! setting them to zero...")

        self.gradient_shift = np.nan_to_num(self.gradient_shift)

        return np.array(self.gradient_shift)

    @staticmethod
    def binorm(n, k):
        from sympy import binomial as comb
        return comb(n, k)

    @staticmethod
    def fact(x):
        from sympy import factorial
        return factorial(x)
