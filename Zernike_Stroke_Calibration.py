import os
import numpy as np
import matplotlib.pyplot as plt
from Misc_Tools import frankot_chellappa, wavefront_reconstruction, show_wavefront
from Zernike_Polynomials_Modules import show_profile, min_circle, image_padding_for_circular_area, Zernike, input_zernike
from Coordinates_Finder_Modules import precise_coord, show_coord
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

if __name__ == '__main__':

    RAN = 0.25
    PIXEL_SIZE = 4.8e-6
    FOCAL = 5.6e-3
    PITCH = 150e-6

    voltage_to_coord = np.load(f"Data_Deposit/range_{RAN}_probe_coordinates_diff.npy") / RAN
    # voltage_to_coord[voltage_to_coord < 1] = 0
    coord_to_voltage = np.linalg.pinv(voltage_to_coord)
    zern_to_voltage = np.load(f"Data_Deposit/range_{RAN}_zernike_to_voltage.npy")
    reference_coord = np.load(f'Data_Deposit/range_{RAN}_reference_coordinates.npy')
    reference_circle = min_circle(reference_coord, scale=1.6)
    reference_frame = np.load(f'Data_Deposit/range_{RAN}_probe_reference_image.npy')

    padded_ref_img = image_padding_for_circular_area(reference_frame, reference_circle, cut=True)
    padded_ref_coord = precise_coord(padded_ref_img)
    padded_ref_circle = min_circle(padded_ref_coord, scale = 1.11)

    zernike = Zernike()

    fc_strokes = []; slope_strokes = []; zernike_orders = []

    for i, (n, m) in enumerate(input_zernike):

        print(f"Processing Zernike: n = {n}, m = {m}")

        zern_amp_carrier = np.zeros((len(input_zernike), 1))
        zern_amp_carrier[i] = 1
        fc_current_zern_volt = np.einsum("ij, ik -> jk", zern_amp_carrier, zern_to_voltage).T
        fc_current_zern_displacement = np.einsum("ij, ik -> jk", fc_current_zern_volt, voltage_to_coord).reshape(-1, 2)

        fc_profile, _, _ = wavefront_reconstruction(padded_ref_coord, padded_ref_coord + fc_current_zern_displacement,
                                                    focal = FOCAL, pitch = PITCH, pixel_size = PIXEL_SIZE)

        fc_stroke = 1e6*(np.max(fc_profile) - np.min(fc_profile))
        fc_strokes.append(fc_stroke)

        current_zern_coff = zernike.calc_zern_cof(n, m)
        current_normalized_zern_profile, _, _ = zernike.zern_from_cof(current_zern_coff, padded_ref_img, padded_ref_circle,
                                                                      cut = True, normalize = True)

        current_normalized_zern_profile *= 0.5e-6

        current_zern_displacement = zernike.zern_local_gradient(
            current_normalized_zern_profile, padded_ref_coord, pitch = PITCH, pixel_size = PIXEL_SIZE)

        current_zern_displacement *= FOCAL / PIXEL_SIZE

        current_zern_volt = np.einsum("ij, ik -> jk", current_zern_displacement.reshape(-1 ,1), coord_to_voltage / PIXEL_SIZE)

        slope_stroke = 1 / np.max(np.abs(current_zern_volt))
        slope_strokes.append(slope_stroke)

        zernike_orders.append(f"Z({n}, {m})")

        # test_disp_slope = np.einsum("ij, ik -> jk", current_zern_volt.T, voltage_to_coord).reshape(-1, 2)
        # test_disp_fc = np.einsum("ij, ik -> jk", fc_current_zern_volt, voltage_to_coord).reshape(-1, 2)
        # show_coord(padded_ref_img, padded_ref_coord + 1e7*test_disp_slope, padded_ref_circle)
        # show_coord(padded_ref_img, padded_ref_coord + test_disp_fc, padded_ref_circle)
        # exit()

    plt.figure(figsize = (10, 6))

    x = np.arange(len(zernike_orders))
    width = 0.25

    plt.bar(x - width / 2, fc_strokes, width, label='F-C Algorithm', color='blue')
    plt.bar(x + width / 2, slope_strokes, width, label='Slope Method', color='orange')

    plt.xlabel('Zernike Mode')
    plt.ylabel('Stroke (μm)')
    plt.title('Comparison of Zernike Stroke Calculation Methods')
    plt.xticks(x, zernike_orders, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
