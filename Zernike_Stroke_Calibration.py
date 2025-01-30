import os
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
import numpy as np
import matplotlib.pyplot as plt
import time
from Misc_Tools import frankot_chellappa, peaks, wavefront_reconstruction, show_wavefront
from Zernike_Polynomials_Modules import show_profile, min_circle, image_padding_for_circular_area, Zernike, input_zernike
from Coordinates_Finder_Modules import precise_coord, show_coord

if __name__ == '__main__':

    ran = 0.25
    PIXEL_SIZE = 4.8e-6
    voltage_to_coord = np.load(f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy") / ran
    coord_to_voltage = np.linalg.pinv(voltage_to_coord)
    zern_to_voltage = np.load(f"Data_Deposit/range_{ran}_zernike_to_voltage.npy")
    reference_coord = np.load(f'Data_Deposit/range_{ran}_reference_coordinates.npy')
    reference_circle = min_circle(reference_coord, scale = 1.6)
    reference_frame = np.load(f'Data_Deposit/range_{ran}_probe_reference_image.npy')
    padded_ref_img = image_padding_for_circular_area(reference_frame, reference_circle, cut = True)
    padded_ref_coord = precise_coord(padded_ref_img)
    padded_ref_circle = min_circle(padded_ref_coord, scale = 1.1)
    F_C_reconstructed_profiles = []
    normalized_zernike_profiles = []
    F_C_reconstruction_stroke = []
    slope_measure_stroke = []

    zernike = Zernike()

    for i, j in enumerate(input_zernike):

        print(f'Zernike order: {i + 1}')

        # Reconstruct the zernike wavefront based on centroid displacement -> local gradient
        zern_coff_carrier = np.zeros((np.array(input_zernike).shape[0], 1))
        zern_coff_carrier[i] = 1
        F_C_current_zern_volt = np.einsum("ij, ik -> jk", zern_coff_carrier, zern_to_voltage).T
        F_C_current_zern_displacement = np.einsum("ij, ik -> jk", F_C_current_zern_volt, voltage_to_coord).reshape(-1, 2)
        current_zern_profile_fc, _, _ = wavefront_reconstruction(padded_ref_coord, padded_ref_coord + F_C_current_zern_displacement)
        F_C_reconstructed_profiles.append(current_zern_profile_fc)
        F_C_current_zern_stroke = np.max(current_zern_profile_fc) - np.min(current_zern_profile_fc)
        F_C_reconstruction_stroke.append(F_C_current_zern_stroke)

        # Calculate the zernike P-V based on voltage ratio between unit(μm) P-V and maximum voltage
        current_zern_coff = zernike.calc_zern_cof(j[0], j[1])
        current_normalized_zern_profile, x, y = zernike.zern_from_cof(current_zern_coff, padded_ref_img, padded_ref_circle, cut = False, normalize = True)
        current_normalized_zern_profile *= 0.5e-6
        normalized_zernike_profiles.append(current_normalized_zern_profile)
        current_zern_displacement = zernike.zern_local_gradient(current_normalized_zern_profile, padded_ref_coord)
        current_zern_volt = np.einsum("ij, ik -> jk", current_zern_displacement.reshape(2*len(padded_ref_coord), 1), coord_to_voltage)
        current_slope_stroke = 1/np.max(np.abs(current_zern_volt))
        slope_measure_stroke.append(current_slope_stroke)

        print(f"Stroke: slope: {current_slope_stroke}, F-C: {F_C_current_zern_stroke}")

