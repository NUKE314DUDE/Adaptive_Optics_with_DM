import time
import matplotlib.pyplot as plt
import numpy as np
import os
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
from scipy.ndimage import gaussian_filter
from Coordinates_Finder_Modules import img_threshold, extrema_coordinates_gradient, \
    exclude_proxi_points, center_of_gravity_with_coord, show_coord, coord_diff, precise_coord, \
    show_coord_diff, precise_coord_fixed_ref, deformable_mirror_random_test_input
# from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from ids import camera, ids_peak, ids_peak_ipl_extension
from Lib64.asdk import DM

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


    # def camera_snapshot(voltage = None, gap = 0.1):
    #     """
    #     Signal the camera to capture a single frame
    #     Can input voltage
    #     :return: captured image (with addressed voltage if added)
    #     """
    #     if voltage is None:
    #         camera.issue_software_trigger()
    #         frame = None
    #         while frame is None:
    #             frame = camera.get_pending_frame_or_null()
    #         image = np.copy(frame.image_buffer)
    #     else:
    #         dm.Send(voltage)
    #         time.sleep(gap)
    #         camera.issue_software_trigger()
    #         frame = None
    #         while frame is None:
    #             frame = camera.get_pending_frame_or_null()
    #         image = np.copy(frame.image_buffer)
    #     return image

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
    probed_transport_matrix = np.load(
        f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy")  # Load saved probe results
    test_len = 50
    random_test_captured_frames_save_path = f'Data_Deposit/range_{ran}_random_test_captured_frames'
    random_test_voltage_input_save_path = f'Data_Deposit/range_{ran}_random_test_voltage_input'
    # random_test_reference_image_save_path = f'Data_Deposit/range_{ran}_random_test_reference_image'
    test_voltage_input = deformable_mirror_random_test_input(ran, test_len, degree_of_freedom=57)
    row, col = probed_transport_matrix.shape
    predicted_coordinates_change = []
    actual_coordinates_change = []
    random_test_captured_frames = []
    # ref_img = camera_snapshot(np.zeros(row))
    # ref_coord = precise_coord(ref_img)
    # np.save(random_test_reference_image_save_path, ref_img)
    ref_coord = np.load(f"Data_Deposit/range_{ran}_reference_coordinates.npy")

    # For testing using one frame

    # show_coord(ref_img, ref_coord)
    # test_img = camera_snapshot(test_voltage_input[0])
    # test_coord = precise_coord_fixed_ref(test_img, ref_coord)
    # show_coord(test_img, test_coord)
    # print(test_voltage_input)
    # test_cal_matrix = test_voltage_input[0].reshape(1, row)
    # test_cal_matrix = (1/ran)*test_cal_matrix
    # test_mul_matrix = probed_transport_matrix * test_cal_matrix.T
    # test_change_prediction = test_mul_matrix.sum(axis = 0, keepdims = True).reshape(1, int(col))
    # reshaped_test_change_prediction = test_change_prediction.reshape(int(col/2), 2)
    # show_coord(test_img, np.add(ref_coord, reshaped_test_change_prediction))

    # For looping test
    for i, j in enumerate(test_voltage_input):
        current_frame = camera_snapshot(j)
        random_test_captured_frames.append(current_frame)
        j = np.array(j).reshape(1, row)
        current_cal_matrix = (1/ran)*j
        current_mul_matrix = probed_transport_matrix * current_cal_matrix.T
        current_change_prediction = current_mul_matrix.sum(axis = 0, keepdims = True).reshape(1, int(col))
        predicted_coordinates_change.append(current_change_prediction)
        current_coordinates = precise_coord_fixed_ref(current_frame, ref_coord)
        current_actual_change = coord_diff(ref_coord, current_coordinates)
        actual_coordinates_change.append(np.array(current_actual_change).reshape(1, col))
        print(f"The random test is {(i+1)*100//test_len} percent complete")
    total_diff = np.subtract(actual_coordinates_change, predicted_coordinates_change)
    mse = (total_diff**2).sum()/(test_len * col)
    print(mse)
    np.save(random_test_voltage_input_save_path, test_voltage_input)
    np.save(random_test_captured_frames_save_path, random_test_captured_frames)

    # Check the latest coordinates
    show_coord(current_frame, np.add(ref_coord, np.array(current_change_prediction).reshape(int(col/2), 2)))
    show_coord(current_frame, np.add(ref_coord, current_actual_change))
    # actual_coordinates_change = np.array(actual_coordinates_change).reshape(row, col)
    # print(actual_coordinates_change - predicted_coordinates_change)
    plt.show()
    # camera.disarm()
    camera.stop_acquisition()
    camera.close()
    dm.Stop()