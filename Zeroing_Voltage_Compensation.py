import os
import numpy as np
import matplotlib.pyplot as plt
import time
from Lib64.asdk import DM
from ids import camera, ids_peak_ipl_extension
from Coordinates_Finder_Modules import precise_coord, precise_coord_fixed_ref, average_distance, grid_from_proxi_center, \
    show_coord, show_coord_diff, coord_diff, grid_nodes_refine
from Zernike_Polynomials_Modules import min_circle, image_padding_for_circular_area
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

def dm_reset(dof = 57, t = 2):
    init_time = time.perf_counter()
    while time.perf_counter() - init_time < t:
        amp = np.exp(-1 / ((2 - time.perf_counter() + init_time) + 1e-6)) * np.cos(np.pi * (time.perf_counter() - init_time)/1e-4)
        time.sleep(0.0002)
        dm.Send(amp * np.ones((dof, 1)))

if __name__ == "__main__":

    # Initialization
    dm = DM("BAX758")
    # dm_reset()
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

    ran = 0.25
    voltage_to_coord = np.load(f"Data_Deposit/range_{ran}_probe_coordinates_diff.npy")/ran
    coord_to_voltage = np.linalg.pinv(voltage_to_coord)
    reference_coord = np.load(f'Data_Deposit/range_{ran}_reference_coordinates.npy')
    reference_circle = min_circle(reference_coord, scale = 1.05)
    # captured_frames = np.load(f'Data_Deposit/range_{ran}_probe_captured_frames.npy')
    reference_frame = np.load(f'Data_Deposit/range_{ran}_probe_reference_image.npy')
    padded_ref_img = image_padding_for_circular_area(reference_frame, reference_circle, cut = True)
    padded_ref_coord = precise_coord(padded_ref_img)
    padded_ref_circle = min_circle(padded_ref_coord, scale = 1.05)

    # Define and optimiza towards ideal grid
    grid_nodes = grid_from_proxi_center(padded_ref_coord, avg_mode = False)
    grid_nodes = grid_nodes_refine(grid_nodes, padded_ref_coord)
    show_coord(padded_ref_img, grid_nodes, padded_ref_circle)
    init_diff = coord_diff(grid_nodes, padded_ref_coord)
    init_error = np.sum(np.linalg.norm(init_diff, axis = 1))
    targeted_error = 0.001*init_error
    momentum = 0.25
    counter = 0
    init_guess = np.sum(coord_to_voltage * init_diff.reshape(2*len(grid_nodes), 1), axis = 0, keepdims = True)
    init_new_voltage = momentum * init_guess
    voltage_carrier = init_new_voltage
    error_carrier = 1000
    while True:
        counter += 1
        current_frame = image_padding_for_circular_area(camera_snapshot(voltage_carrier.T), reference_circle, cut = True)
        current_coord = precise_coord_fixed_ref(current_frame, padded_ref_coord)
        current_diff = coord_diff(grid_nodes, current_coord)
        current_error = np.sum(np.linalg.norm(current_diff, axis = 1))
        jumper = current_error / error_carrier
        print(jumper)
        error_carrier = current_error
        current_guess = np.sum(coord_to_voltage * current_diff.reshape(2*len(grid_nodes), 1), axis = 0, keepdims = True)
        voltage_carrier = voltage_carrier - (momentum * np.exp(-counter/100) * current_guess)
        print(f'iteration no.{counter}, current total error is {current_error}.')
        if current_error <= targeted_error or abs(1 - jumper) <= 4e-2 or counter >= 24:
            print(f'loop ends in iteration no.{counter}, with a total error of {current_error}.')
            break

    show_coord_diff(current_frame, grid_nodes, current_coord)
    print(np.max(np.abs(voltage_carrier)))
    # np.save(f'Data_Deposit/zeroing_compensation_voltage.npy', voltage_carrier)

    camera.stop_acquisition()
    camera.close()
    # camera.disarm()
    dm.Stop()
    plt.show()