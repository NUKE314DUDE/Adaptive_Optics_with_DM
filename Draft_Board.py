import os
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
import numpy as np
import matplotlib.pyplot as plt
import time
from Misc_Tools import frankot_chellappa, peaks, wavefront_gradient, wavefront_gradient_test
from Zernike_Polynomials_Modules import show_profile, min_circle, image_padding_for_circular_area
from Coordinates_Finder_Modules import precise_coord, show_coord
from scipy import signal

ran = 0.25
voltage_to_coord = np.load(f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy") / ran
coord_to_voltage = np.linalg.pinv(voltage_to_coord)
zern_to_voltage = np.load(f"Data_Deposit/range_{ran}_zernike_to_voltage.npy")
reference_coord = np.load(f'Data_Deposit/range_{ran}_reference_coordinates.npy')
reference_circle = min_circle(reference_coord, scale = 1.25)
reference_frame = np.load(f'Data_Deposit/range_{ran}_probe_reference_image.npy')
padded_ref_img = image_padding_for_circular_area(reference_frame, reference_circle, cut=True)
padded_ref_coord = precise_coord(padded_ref_img)
padded_ref_circle = min_circle(padded_ref_coord, scale = 1.2)

zern_coff = np.zeros((27, 1))
zern_coff[7] = 1
zern_volt = np.einsum("ij, ik -> jk", zern_coff, zern_to_voltage).T
zern_disp = np.einsum("ij, ik -> jk", zern_volt, voltage_to_coord).reshape(-1, 2)
zern_gradient_x, zern_gradient_y, zern_grid_x, zern_grid_y = wavefront_gradient_test(padded_ref_coord, padded_ref_coord + zern_disp)

peaks_z, peaks_grid_x, peaks_grid_y = peaks(160, 160)
peaks_gradient_x, peaks_gradient_y = np.gradient(peaks_z)

re_zern = frankot_chellappa(zern_gradient_x, zern_gradient_y)
# re_zern = frankot_chellappa(-zern_gradient_x, zern_gradient_y)
# re_peaks = frankot_chellappa(peaks_gradient_x, peaks_gradient_y)
# show_profile(re_peaks, peaks_grid_x, peaks_grid_y)
# show_profile(peaks_z, peaks_grid_x, peaks_grid_y)
show_profile(re_zern, zern_grid_x, zern_grid_y)
# show_coord(padded_ref_img, padded_ref_coord + zern_disp)
print(f"P-V: {np.max(re_zern) - np.min(re_zern)}")
plt.show()