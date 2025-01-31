import os
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
import numpy as np
import matplotlib.pyplot as plt
import time
from Misc_Tools import frankot_chellappa, peaks, wavefront_reconstruction, show_wavefront
from Zernike_Polynomials_Modules import show_profile, min_circle, image_padding_for_circular_area
from Coordinates_Finder_Modules import precise_coord, show_coord

ran = 0.25
voltage_to_coord = np.load(f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy") / ran
coord_to_voltage = np.linalg.pinv(voltage_to_coord)
zern_to_voltage = np.load(f"Data_Deposit/range_{ran}_zernike_to_voltage.npy")
reference_coord = np.load(f'Data_Deposit/range_{ran}_reference_coordinates.npy')
reference_circle = min_circle(reference_coord, scale = 1.6)
reference_frame = np.load(f'Data_Deposit/range_{ran}_probe_reference_image.npy')
padded_ref_img = image_padding_for_circular_area(reference_frame, reference_circle, cut=True)
padded_ref_coord = precise_coord(padded_ref_img)
padded_ref_circle = min_circle(padded_ref_coord, scale = 1.1)

zern_coff = np.zeros((27, 1))
zern_coff[3] = 1
zern_volt = np.einsum("ij, ik -> jk", zern_coff, zern_to_voltage).T
zern_disp = np.einsum("ij, ik -> jk", zern_volt, voltage_to_coord).reshape(-1, 2)

zern_reconstruct, zern_grid_x, zern_grid_y = wavefront_reconstruction(padded_ref_coord, padded_ref_coord + zern_disp)
show_wavefront(zern_reconstruct, zern_grid_x, zern_grid_y)

print(f"P-V: {np.max(zern_reconstruct) - np.min(zern_reconstruct)}")