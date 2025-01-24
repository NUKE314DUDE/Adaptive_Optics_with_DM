import numpy as np
from scipy import signal
from Zernike_Polynomials_Modules import min_circle
from Coordinates_Finder_Modules import coord_diff

def frankot_chellappa(px, py):
    """
    Calculates the profile from gradient
    :param px: directional gradient along x
    :param py: directional gradient along y
    :return: reconstructed z profile
    """
    px = np.array(px)
    py = np.array(py)
    if px.shape != py.shape:
        raise TypeError('Directional gradients should have the save size!')
    M, N = px.shape
    freq_x = np.fft.fftfreq(M)
    freq_y = np.fft.fftfreq(N)
    [U, V] = np.meshgrid(freq_x, freq_y, indexing = 'ij')
    Px = np.fft.fft2(px)
    Py = np.fft.fft2(py)
    denom = (2 * np.pi * U)**2 + (2 * np.pi * V)**2
    denom[denom == 0] = 1e-10
    Fz = -1j * (2 * np.pi * U * Px + 2 * np.pi * V * Py) / denom
    Fz[0, 0] = 0
    # hann_window = np.outer(signal.windows.hann(M), signal.windows.hann(N))
    # Fz = Fz * np.fft.fftshift(hann_window)

    rZ = np.real(np.fft.ifft2(Fz))

    return rZ

def peaks(size_x, size_y, ran_x = 3, ran_y = 3, normalize = False):
    """
    Generates the peaks function with a certain size
    :param size_x:
    :param size_y:
    :param ran_x:
    :param ran_y:
    :param normalize: to limit range in [-1, 1]
    :return:
    """
    x = np.linspace(-ran_x, ran_x, size_x)
    y = np.linspace(-ran_y, ran_y, size_y)
    X, Y = np.meshgrid(x, y)
    Z = 3 * (1 - X)**2 * np.exp(-X**2 - (Y + 1)**2) - 10 * (X / 5 - X**3 - Y **5) * np.exp(-X**2 - Y**2) - 1/3 * np.exp(-(X + 1)**2 - Y**2)
    if normalize:
        max_Z = np.max(Z)
        min_Z = np.min(Z)
        Z = 2 * (Z - min_Z) / (max_Z - min_Z) - 1
    return X, Y, Z

def wavefront_reconstruction(padded_ref_coord, padded_current_coord, focal = 5.6e-3, pitch = 150e-6, pixel_size = 4.8e-6):
    dispacement = coord_diff(padded_ref_coord, )
    padded_ref_circle = min_circle(padded_ref_coord, scale = 1.1)
    space_x = np.arange(-padded_ref_circle[1] * pixel_size, padded_ref_circle[1] * pixel_size + pitch, pitch)
    space_y = np.arange(-padded_ref_circle[1] * pixel_size, padded_ref_circle[1] * pixel_size + pitch, pitch)
    grid_x, grid_y = np.meshgrid(space_x, space_y, indexing='ij')
