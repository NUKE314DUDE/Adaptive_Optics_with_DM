import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
from torch.onnx.symbolic_opset9 import hann_window
from Zernike_Polynomials_Modules import min_circle, normalization
from Coordinates_Finder_Modules import coord_diff, grid_nodes_refine, grid_from_proxi_center

def pass_filter(signal, sample_freq, cut_low = None, cut_high = None, order = 4):
    """
    Butter filter with different modes
    :param signal:
    :param sample_freq:
    :param cut_low:
    :param cut_high:
    :param order:
    :return: filtered signal
    """
    nq_freq = sample_freq * 0.5
    if cut_low is None and cut_high is not None:
        high = cut_high / nq_freq
        b, a = butter(order, high, btype='high', analog = False)
        filtered_signal = filtfilt(b, a, signal)
    elif cut_low is not None and cut_high is None:
        low = cut_low / nq_freq
        b, a = butter(order, low, btype='low', analog = False)
        filtered_signal = filtfilt(b, a, signal)
    elif cut_low is not None and cut_high is not None:
        low = cut_low / nq_freq; high = cut_high / nq_freq
        b, a = butter(order, [low, high], btype='band', analog = False)
        filtered_signal = filtfilt(b, a, signal)
    else:
        raise TypeError('No critical frequency entered!')
    return filtered_signal

def path_integration(px, py, start = (0, 0)):
    """
    Integrate along row then col from starting point
    :param px: gradient X
    :param py: gradient Y
    :param start: starting point
    :return: path integration result
    """
    M, N = px.shape
    path_x = np.arange(start[1], M)
    path_y = np.arange(start[0], N)
    ref_height = 0

    for i in path_x:
        ref_height += px[i, start[1]]

    for j in path_y:
        ref_height += py[path_x[-1], j]

    return ref_height, start

def frankot_chellappa(px, py):
    """
    Calculates the profile from gradient
    :param px: directional gradient along x
    :param py: directional gradient along y
    :return: reconstructed z profile, assumes the original profile has a mean height of 0
    """
    px = np.nan_to_num(np.array(px), nan = 0.0); py = np.nan_to_num(np.array(py), nan = 0.0)

    if px.shape != py.shape:
        raise TypeError('Directional gradients should have the save size!')

    # ref_height, start = path_integration(px, py) # path integration for baseline correction

    rows, cols = px.shape
    ft_px = np.fft.fft2(px)
    ft_py = np.fft.fft2(py)

    freq_x = 2j*np.pi*np.fft.fftfreq(rows).reshape(-1, 1)
    freq_y = 2j*np.pi*np.fft.fftfreq(cols)

    denom = freq_x**2 + freq_y**2
    denom[denom == 0] = 1e-12

    ft_z = (freq_x*ft_px + freq_y*ft_py) / denom

    # hann_window = np.outer(signal.windows.hann(M), signal.windows.hann(N))
    # z = np.fft.ifft2(ft_z * hann_window).real

    z = np.fft.ifft2(ft_z).real
    z -= np.mean(z)

    return z

def peaks(size_x, size_y, ran_x = 3, ran_y = 3, normalize = False):
    """
    Generates the peaks function with a certain shape
    :param size_x: size in x
    :param size_y: size in y
    :param ran_x: range x
    :param ran_y: range y
    :param normalize: to limit range in [-1, 1]
    :return: coordinates grid and Z profile
    """
    x = np.linspace(-ran_x, ran_x, size_x)
    y = np.linspace(-ran_y, ran_y, size_y)

    grid_x, grid_y = np.meshgrid(x, y)
    z = 3 * (1 - grid_x)**2 * np.exp(- grid_x**2 - (grid_y + 1)**2) - 10 * (grid_x / 5 - grid_x**3 - grid_y **5) * np.exp(-grid_x**2 - grid_y**2) - 1/3 * np.exp(-(grid_x + 1)**2 - grid_y**2)

    if normalize:
        # max_z = np.max(z)
        # min_z = np.min(z)
        # z = 2 * (z - min_z) / (max_z - min_z) - 1
        z = normalization(z)

    return z, grid_x, grid_y

def gaussian(x_space, amp = 1., mean = 0., std_dev = 1.):
    """
    Gaussian distribution
    :param x_space: x coordinates
    :param amp: amplitude
    :param mean:
    :param std_dev:
    :return:
    """
    y = amp*(1/(std_dev*np.sqrt(2*np.pi)))*np.exp(-(x_space - mean)**2/(2 * std_dev**2))

    return np.array(y)

def wavefront_reconstruction(padded_ref_coord, padded_current_coord,
                             focal=5.6e-3, pitch=150e-6,
                             pixel_size=4.8e-6,
                             lambda_correction=True):
    """
    Reconstruct the wavefront on WFS using centroid displacement
    :param padded_ref_coord:
    :param padded_current_coord:
    :param focal:
    :param pitch:
    :param pixel_size:
    :param lambda_correction:
    :return: reconstructed profile on DM plane
    """

    null_grid = grid_from_proxi_center(padded_ref_coord)
    _, rotation = grid_nodes_refine(null_grid, padded_ref_coord, get_rotation=True)

    ref_center = min_circle(padded_ref_coord)[0]
    rotated_ref_coord = (padded_ref_coord - ref_center) @ rotation + ref_center
    rotated_current_coord = (padded_current_coord - ref_center) @ rotation + ref_center
    ref_circle = min_circle(rotated_ref_coord, scale=1.6)
    radius_pixels = int(np.ceil(ref_circle[1]))

    x_space = np.arange(-radius_pixels, radius_pixels + pitch / pixel_size,
                        pitch / pixel_size) * pixel_size
    y_space = np.arange(-radius_pixels, radius_pixels + pitch / pixel_size,
                        pitch / pixel_size) * pixel_size
    x_grid, y_grid = np.meshgrid(x_space, y_space, indexing='ij')

    displacement_pixel = coord_diff(rotated_ref_coord, rotated_current_coord)
    displacement_meter = displacement_pixel * pixel_size

    ref_coord_phy = (rotated_ref_coord - ref_center) * pixel_size
    wavefront_slope = pitch * displacement_meter / focal

    if lambda_correction:
        wavefront_slope *= -1

    gradient_x = griddata(ref_coord_phy, wavefront_slope[:, 0], (x_grid, y_grid),
                          method='cubic', fill_value=0)
    gradient_y = griddata(ref_coord_phy, wavefront_slope[:, 1], (x_grid, y_grid),
                          method='cubic', fill_value=0)

    z_reconstruct = frankot_chellappa(gradient_x, gradient_y)

    return z_reconstruct, x_grid, y_grid

def show_wavefront(z, x, y,
                   unit = 'μm',
                   cmap = 'viridis',
                   elev = 30, azim = 45,
                   title = 'Reconstructed Wavefront'):

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111, projection = '3d')

    surf = ax.plot_surface(x, y, z,
                           cmap = cmap,
                           rstride = 1,
                           cstride = 1,
                           linewidth = 0,
                           antialiased = True)

    ax.set_xlabel(f'X [{unit}]', fontsize=12)
    ax.set_ylabel(f'Y [{unit}]', fontsize=12)
    ax.set_zlabel(f'Wavefront [{unit}]', fontsize=12)

    ax.view_init(elev=elev, azim=azim)
    cbar = fig.colorbar(surf, shrink=0.6)
    cbar.set_label(f'Height [{unit}]', rotation=270, labelpad=20)

    plt.title(title, fontsize = 14)
    plt.tight_layout()
    plt.show()