import time
import matplotlib.pyplot as plt
import numpy as np
import os
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
from Coordinates_Finder_Modules import img_threshold, extrema_coordinates_gradient, \
    exclude_proxi_points, center_of_gravity_with_coord, show_coord, coord_diff, precise_coord, \
    show_coord_diff, precise_coord_fixed_ref, deformable_mirror_single_probe_input
from Zernike_Polynomials_Modules import min_circle
# from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from Lib64.asdk import DM
from ids import camera
from ids import ids_peak, ids_peak_ipl_extension

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

    probe_range = 0.25
    ref_voltage, probe_voltage = deformable_mirror_single_probe_input(probe_range, degree_of_freedom=57)
    total_length = len(ref_voltage)
    reference_image_path = f"C:\Xiong_Jianxuan\Python_Projects\Adaptive_Optics\Data_Deposit\\range_{probe_range}_probe_reference_image.npy"
    captured_frames_path = f"C:\Xiong_Jianxuan\Python_Projects\Adaptive_Optics\Data_Deposit\\range_{probe_range}_probe_captured_frames.npy"
    coordinates_diff_save_path = f"C:\Xiong_Jianxuan\Python_Projects\Adaptive_Optics\Data_Deposit\\range_{probe_range}_probe_coordinates_diff.npy"
    ref_coord_save_path = f"C:\Xiong_Jianxuan\Python_Projects\Adaptive_Optics\Data_Deposit\\range_{probe_range}_reference_coordinates.npy"

    # Get the reference image & the reference coordinates
    dm.Send(ref_voltage)
    ref_img = camera_snapshot()
    np.save(reference_image_path, ref_img)
    ref_coord = precise_coord(ref_img)
    np.save(ref_coord_save_path, ref_coord)

    # Capture frames & calculate coordinates for each voltage
    captured_frames = []
    coordinates_diff = []
    current_frame = None
    current_coordinates = None
    for i, j in enumerate(probe_voltage):
        current_frame = camera_snapshot(j)
        captured_frames.append(current_frame)
        current_coordinates = precise_coord_fixed_ref(current_frame, ref_coord)
        current_coordinates_diff = coord_diff(ref_coord, current_coordinates)
        coordinates_diff.append(current_coordinates_diff)
        # show_coord(current_frame, ref_coord, min_circle(ref_coord))
        # show_coord_diff(ref_img, ref_coord, current_coordinates)
        # plt.show()
        # input()
        print(f'The probing is {(i+1)*100//total_length} percent complete')
    np.save(captured_frames_path, captured_frames)

    # Convert to desired format & save
    row, col = np.array(current_coordinates_diff).shape
    coordinates_diff = np.array(coordinates_diff).reshape(total_length, row*col)
    np.save(coordinates_diff_save_path, coordinates_diff)

    # Disengage the instrument & show the last result
    print(f"The saved probe matrix's shape is {coordinates_diff.shape}")
    camera.stop_acquisition()
    camera.close()
    # camera.disarm()
    dm.Stop()
    show_coord_diff(current_frame, ref_coord, current_coordinates)
    plt.show()