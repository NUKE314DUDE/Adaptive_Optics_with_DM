import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from torch.onnx.symbolic_opset9 import hann_window
from Zernike_Polynomials_Modules import min_circle
from Coordinates_Finder_Modules import coord_diff, closest_to_center, grid_nodes_refine, grid_from_proxi_center


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
        max_z = np.max(z)
        min_z = np.min(z)
        z = 2 * (z - min_z) / (max_z - min_z) - 1

    return z, grid_x, grid_y

def wavefront_gradient(padded_ref_coord, padded_current_coord, focal = 5.6e-3, pitch = 150e-6, pixel_size = 4.8e-6):

    current_displacement = coord_diff(padded_ref_coord, padded_current_coord) * pixel_size # matched_index marked the index from ref to current
    padded_ref_circle = min_circle(padded_ref_coord, scale = 1.25)
    null_grid = grid_from_proxi_center(padded_ref_coord)
    _, rotation = grid_nodes_refine(null_grid, padded_ref_coord, get_rotation = True)
    padded_ref_coord = ((padded_ref_coord - padded_ref_circle[0]) @ rotation) + padded_ref_circle[0]
    centered_ref_coord_space = (padded_ref_coord - padded_ref_circle[0]) * pixel_size
    space_x = np.arange(-padded_ref_circle[1] , padded_ref_circle[1]  + pitch/pixel_size, pitch/pixel_size) * pixel_size
    space_y = np.arange(-padded_ref_circle[1] , padded_ref_circle[1]  + pitch/pixel_size, pitch/pixel_size) * pixel_size
    grid_x, grid_y = np.meshgrid(space_x, space_y, indexing='ij')
    gradient_x = np.zeros_like(grid_x)
    gradient_y = np.zeros_like(grid_y)
    # checker = np.zeros_like(gradient_x)
    _, N = gradient_x.shape
    space_nodes = np.stack((grid_x, grid_y), axis = -1).reshape(-1, 2)
    fill_list = []; fill_index = []

    for pos in centered_ref_coord_space:
        dist = np.linalg.norm(space_nodes - pos, axis = 1)
        fill_list.append(np.argmin(dist))

    fill_list = sorted(fill_list)

    for idx in fill_list:
        fill_row = idx // N
        fill_col = idx % N
        fill_index.append([fill_row, fill_col])

    for order, operant in enumerate(fill_index):
        current_gradient = -current_displacement / focal # check units
        gradient_x[operant[0], operant[1]] = current_gradient[order, 0]
        gradient_y[operant[0], operant[1]] = current_gradient[order, 1]
        # checker[operant[0], operant[1]] = 1

    return gradient_x, gradient_y, grid_x, grid_y

def wavefront_gradient_test(padded_ref_coord, padded_current_coord,
                            focal = 5.6e-3, pitch = 150e-6, pixel_size = 4.8e-6):
    """

    :param padded_ref_coord:
    :param padded_current_coord:
    :param focal:
    :param pitch:
    :param pixel_size:
    :return:
    """
    # Calculate the displacement in meters
    current_displacement = coord_diff(padded_ref_coord, padded_current_coord) * pixel_size

    # Find the minimum circle enclosing the reference coordinates
    padded_ref_circle = min_circle(padded_ref_coord, scale=1.6)

    # Generate a grid based on the reference coordinates
    null_grid = grid_from_proxi_center(padded_ref_coord)
    _, rotation = grid_nodes_refine(null_grid, padded_ref_coord, get_rotation=True)
    padded_ref_coord = ((padded_ref_coord - padded_ref_circle[0]) @ rotation) + padded_ref_circle[0]

    # Convert coordinates to physical space (meters)
    centered_ref_coord_space = (padded_ref_coord - padded_ref_circle[0]) * pixel_size

    # Create a grid in physical space
    space_x = np.arange(-padded_ref_circle[1], padded_ref_circle[1] + pitch / pixel_size,
                        pitch / pixel_size) * pixel_size
    space_y = np.arange(-padded_ref_circle[1], padded_ref_circle[1] + pitch / pixel_size,
                        pitch / pixel_size) * pixel_size
    grid_x, grid_y = np.meshgrid(space_x, space_y, indexing='ij')

    # Initialize gradient fields
    gradient_x = np.zeros_like(grid_x)
    gradient_y = np.zeros_like(grid_y)

    # Calculate the gradient in physical space (radians)
    current_gradient = current_displacement / focal

    # Interpolate the gradient onto the grid
    points = centered_ref_coord_space
    values_x = -current_gradient[:, 0]
    values_y = -current_gradient[:, 1]

    gradient_x = griddata(points, values_x, (grid_x, grid_y), method='linear', fill_value=0)
    gradient_y = griddata(points, values_y, (grid_x, grid_y), method='linear', fill_value=0)

    return gradient_x, gradient_y, grid_x, grid_y