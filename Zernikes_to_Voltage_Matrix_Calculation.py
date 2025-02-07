import numpy as np
import matplotlib.pyplot as plt
import os
from Zernike_Polynomials_Modules import min_circle, Zernike, image_padding_for_circular_area, input_zernike
from Coordinates_Finder_Modules import precise_coord, precise_coord_fixed_ref, show_coord
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

if __name__ == '__main__':

    ran = 0.25
    voltage_to_displace = np.load(f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy")/ran
    displace_to_voltage = np.linalg.pinv(voltage_to_displace)
    reference_coord = np.load(f'Data_Deposit/range_{ran}_reference_coordinates.npy')
    reference_circle = min_circle(reference_coord, scale = 1.6)
    # captured_frames = np.load(f'Data_Deposit/range_{ran}_probe_captured_frames.npy')
    reference_frame = np.load(f'Data_Deposit/range_{ran}_probe_reference_image.npy')
    padded_ref_img = image_padding_for_circular_area(reference_frame, reference_circle, cut = True)
    padded_ref_coord = precise_coord(padded_ref_img)
    padded_ref_circle = min_circle(padded_ref_coord, scale = 1.11)

    Z = Zernike()
    Zern_cof = dict()
    Zern_coord_diff = []
    Zern_voltage = []
    amp = 1

    for i, j in enumerate(input_zernike):
        print(f'Now at No.{i}: with n = {j[0]}, m = {j[1]}.')
        current_cof = Z.calc_zern_cof(j[0], j[1])
        Zern_cof.update(current_cof)
        current_zern_profile, _, _ = Z.zern_from_cof(current_cof, padded_ref_img, padded_ref_circle)
        current_coord_diff = Z.zern_local_gradient(current_zern_profile, padded_ref_coord)
        # print(current_coord_diff)
        Zern_coord_diff.append(current_coord_diff)
        # show_coord(current_zern_profile, padded_ref_coord + 6e5*current_coord_diff.reshape(-1,2), padded_ref_circle)
        # plt.show()
        # input()
        init_voltage = np.sum(displace_to_voltage * current_coord_diff.reshape(2 * len(padded_ref_coord), 1), axis = 0, keepdims = True)
        voltage = amp * init_voltage/(np.max(np.abs(init_voltage)))
        Zern_voltage.append(voltage)
    Zern_voltage = np.array(Zern_voltage).reshape(i+1, len(voltage_to_displace))
    np.save(f"Data_Deposit/range_{ran}_zernike_to_voltage.npy", Zern_voltage)
    # print(Zern_voltage.shape)

    # For testing the zernike_to_voltage matrix's performance
    # test_displace = voltage_to_displace*Zern_voltage[-1].T
    # test_displace = np.sum(test_displace, axis = 0, keepdims = True)
    # show_coord(padded_ref_img, padded_ref_coord + test_displace.reshape(len(padded_ref_coord), 2), padded_ref_circle)
    # plt.show()

