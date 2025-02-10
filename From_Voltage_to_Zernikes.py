import os
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
import time
import numpy as np
import matplotlib.pyplot as plt
from Coordinates_Finder_Modules import precise_coord, show_coord, extrema_coordinates_gradient, img_threshold, \
    img_preprocess, show_coord_diff, precise_coord_fixed_ref
from Zernike_Polynomials_Modules import min_circle, Zernike, image_padding_for_circular_area, show_profile
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

    Z = Zernike()

    ran = 0.25
    voltage_to_coord = np.load(f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy")/ran
    coord_to_voltage = np.linalg.pinv(voltage_to_coord)
    reference_coord = np.load(f'Data_Deposit/range_{ran}_reference_coordinates.npy')
    reference_circle = min_circle(reference_coord, scale = 1.1)
    # captured_frames = np.load(f'Data_Deposit/range_{ran}_probe_captured_frames.npy')
    reference_frame = np.load(f'Data_Deposit/range_{ran}_probe_reference_image.npy')
    padded_ref_img = image_padding_for_circular_area(reference_frame, reference_circle, cut = True)
    padded_ref_coord = precise_coord(padded_ref_img)
    padded_ref_circle = min_circle(padded_ref_coord, scale = 1.1)
    # show_coord(padded_ref_img, padded_ref_coord, padded_ref_circle)
    # plt.show(), input()

    # Set Zernikes
    Zern_cof = dict()
    Zern_coord_diff = []
    Zern_voltage = []
    amp = 0.5

    input_zernike = [(3, -3),
                 (3, -1),
                 (3, 1),
                 (3, 3),
                 (2, 0)]

    # Calculate Zernike voltages
    for i, j in reversed(list(enumerate(input_zernike))):
        current_cof = Z.calc_zern_cof(j[0], j[1])
        Zern_cof.update(current_cof)
        current_zern_profile, _, _ = Z.zern_from_cof(current_cof, padded_ref_img, padded_ref_circle)
        current_coord_diff = Z.zern_local_gradient(current_zern_profile, padded_ref_coord)
        Zern_coord_diff.append(current_coord_diff)
        init_voltage = np.sum(coord_to_voltage * current_coord_diff.reshape(2 * len(padded_ref_coord), 1), axis = 0, keepdims = True)
        voltage = init_voltage/(np.max(np.abs(init_voltage))).T
        Zern_voltage.append(amp * voltage)
    # Zern_coord_diff = np.sum(np.array(Zern_coord_diff), axis = 0, keepdims = True).T
    # Zern_voltage.append(np.sum(coord_to_voltage * Zern_coord_diff, axis = 0, keepdims = True))
    Zern_voltage = np.array(Zern_voltage)
    captured_zern_frames = []

    # Testing
    for v in Zern_voltage:
        current_frame = camera_snapshot(v.reshape(57))
        current_frame = image_padding_for_circular_area(current_frame, reference_circle)
        captured_zern_frames.append(current_frame)
        show_coord(current_frame, padded_ref_coord, padded_ref_circle)
        # show_coord_diff(current_frame, padded_ref_coord, precise_coord_fixed_ref(current_frame, padded_ref_coord))
    captured_zern_frames = np.array(captured_zern_frames)
    # np.save(f"Data_Deposit/amp_{amp}_zernike_generation_captured_frames.npy", captured_frames)
    Z.reset()
    camera.stop_acquisition()
    camera.close()
    # camera.disarm()
    dm.Stop()
    plt.show()
