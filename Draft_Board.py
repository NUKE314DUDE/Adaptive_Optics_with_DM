import os
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
import numpy as np
import matplotlib.pyplot as plt
from Coordinates_Finder_Modules import precise_coord, show_coord, img_preprocess, precise_coord_fixed_ref
from Zernike_Polynomials_Modules import min_circle, Zernike, image_padding_for_circular_area, show_profile, mean_gradient
import time
from DM_Control_Modules import AlPaoDM

if __name__  == "__main__":
    ran = 0.25
    voltage_to_coord = np.load(f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy")/ran
    coord_to_voltage = np.linalg.pinv(voltage_to_coord)
    reference_coord = np.load(f'Data_Deposit/range_{ran}_reference_coordinates.npy')
    reference_circle = min_circle(reference_coord, scale = 1.25)
    captured_frames = np.load(f'Data_Deposit/range_{ran}_probe_captured_frames.npy')
    reference_frame = np.load(f'Data_Deposit/range_{ran}_probe_reference_image.npy')
    padded_ref_img = image_padding_for_circular_area(reference_frame, reference_circle, cut = True)
    padded_ref_coord = precise_coord(padded_ref_img)
    padded_ref_circle = min_circle(padded_ref_coord, scale = 1.15)

    time.sleep(0.05)
    seq_length = 1000
    seq = np.zeros((27, seq_length))
    test_amp = 0.9*np.sin(np.linspace(0, 2 * np.pi, seq_length))
    seq[8, :] = test_amp
    DM = AlPaoDM()
    DM.send_zernike_patterns(seq, repeat=0)
    DM.stop_loop()

    # print(voltage_to_coord.shape)
    test_calc_coord = np.einsum('ij, ik -> kj', voltage_to_coord, DM.checker.reshape(57, 1))
    show_coord(padded_ref_img, padded_ref_coord + test_calc_coord.reshape(127,2), padded_ref_circle)
    plt.show()