import time
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from Coordinates_Finder_Modules import img_threshold, extrema_coordinates_gradient, \
    exclude_proxi_points, center_of_gravity_with_coord, show_coord, coord_diff, precise_coord, \
    show_coord_diff, precise_coord_fixed_ref, deformable_mirror_random_test_input
# from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from ids import camera
from ids import ids_peak, ids_peak_ipl_extension
from Lib64.asdk import DM
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

if __name__ == "__main__":
    # Initialization
    dm = DM("BAX758")

    def camera_snapshot(input_voltage=None, gap=0.05):

        image = None
        if input_voltage is None:
            for j in range(4):
                image = camera.get_frame()
        else:
            dm.Send(input_voltage)
            time.sleep(gap)
            for j in range(4):
                image = camera.get_frame()
        return image.astype(float)

    # def camera_snapshot(voltages = None, gap = 0.05, repeat = 4, timeout_ms = 1000):
    #     image = None
    #     if voltages is None:
    #         for count in range(repeat):
    #             buffer = camera.data_stream.WaitForFinishedBuffer(timeout_ms)
    #             image = np.copy(ids_peak_ipl_extension.BufferToImage(buffer).get_numpy())
    #             camera.data_stream.QueueBuffer(buffer)
    #     else:
    #         dm.Send(voltages)
    #         time.sleep(gap)
    #         for count in range(repeat):
    #             buffer = camera.data_stream.WaitForFinishedBuffer(timeout_ms)
    #             image = np.copy(ids_peak_ipl_extension.BufferToImage(buffer).get_numpy())
    #             camera.data_stream.QueueBuffer(buffer)
    #         return image

    camera = camera()
    camera.set_bit_depth(8)
    camera.set_full_chip()
    # cam.set_active_region(300,900,300,300)
    camera.set_exposure_ms(0.015)
    camera.set_gain(1.0)
    camera.start_acquisition()

    # thorsdk = TLCameraSDK()
    # camera = thorsdk.open_camera(thorsdk.discover_available_cameras()[0])
    # camera.exposure_time_us = 40
    # camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
    # camera.image_poll_timeout_ms = 1000  # 1 second polling timeout
    # camera.arm(2)

    ran = 0.25
    probed_transport_matrix = np.load(f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy")# Load saved probe results
    inverse_transport_matrix = np.linalg.pinv(probed_transport_matrix)# Get pseudo inverse matrix
    ref_coord = np.load(f"Data_Deposit/range_{ran}_reference_coordinates.npy")

    test_voltage_input = deformable_mirror_random_test_input(0.5, 1, degree_of_freedom=57).T
    initial_frame = camera_snapshot(test_voltage_input)
    initial_coord = precise_coord_fixed_ref(initial_frame, ref_coord)
    initial_diff = coord_diff(ref_coord, initial_coord).reshape(1, len(initial_coord)*2)
    show_coord_diff(initial_frame, ref_coord, initial_coord)
    initial_guess = np.sum(inverse_transport_matrix * initial_diff.T, axis = 0, keepdims = True)*ran
    initial_mse = np.mean(np.square(initial_diff))
    targeted_mse = 0.0001*initial_mse
    momentum = 0.2
    initial_new_voltage = test_voltage_input - (momentum * initial_guess).T
    vol_carrier = initial_new_voltage
    counter = 0
    # print(test_voltage_input)
    # print(initial_new_voltage)
    c_mse = 0

    # Close loop test
    while True:
        counter += 1
        current_frame = camera_snapshot(vol_carrier)
        current_coord = precise_coord_fixed_ref(current_frame, ref_coord)
        current_diff = coord_diff(ref_coord, current_coord).reshape(1, len(current_coord)*2)
        current_mse = np.mean(np.square(current_diff))
        jumper = abs(current_mse - c_mse)
        c_mse = current_mse
        current_guess = np.sum(inverse_transport_matrix * current_diff.T, axis = 0, keepdims = True)*ran
        vol_carrier = vol_carrier - (current_guess * momentum).T
        print(f"iteration no.{counter}, current mse is: {current_mse}")
        if current_mse <= targeted_mse or jumper <= 0.01 or counter > 16:
            print(f"loop ends in iteration no.{counter}, with a mse of: {current_mse}")
            break

    show_coord_diff(current_frame, ref_coord, current_coord)
    # camera.disarm()
    camera.stop_acquisition()
    camera.close()
    dm.Stop()
    plt.show()