import os
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Main_Camera_Control_Modules import MainCamera, MainCameraTrigger
from multiprocessing.sharedctypes import RawArray
from DM_Control_Modules import AlPaoDM, smoothed_sawtooth
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

CONFIG = {
    "DM" : {
        "zernike_order" : 3,
        "amp" : 0.95,
        "signal_freq" : 60,
        "sawtooth_params" : {
            "cut_freq_low" : 640,
            "cut_freq_high" : None,
            "fill" : 0.95
        }
    },
    "camera" : {
        "camera_size" : 2304,
        "subarray_mode" : 2.0,
        "subarray_size" : (1024, 1024),
        "sensor_mode" : 12.0,
        "exposure_time" : 1, # ms
        "trigger_source" : 2,
        "trigger_polarity" : 2,
        "trigger_delay" : 8e-6,
        "internal_line_interval" : 1*1e-6
    }
}

def dm_control_process():

    try:

        amp_modulation = smoothed_sawtooth(
            fill = CONFIG["DM"]["sawtooth_params"]["fill"],
            cut_freq_low = CONFIG["DM"]["sawtooth_params"]["cut_freq_low"],
            cut_freq_high = CONFIG["DM"]["sawtooth_params"]["cut_freq_high"],
            sig_freq = CONFIG["DM"]["signal_freq"]
        )

        dm_sequence = np.zeros((27, len(amp_modulation)))
        dm_sequence[CONFIG["dm"]["zernike_order"]] = CONFIG["dm"]["amp"] * amp_modulation

        dm = AlPaoDM()
        dm.send_zernike_patterns(dm_sequence, repeat = 0)
        print("DM sequence initiated...")

        while 1:
            time.sleep(1)

    except Exception as e:
        print(f"Error in DM sequence: {e}")

def camera_control_process(frame_queue: mp.Queue):

    try:

        cam = MainCamera()
        cam.camera_open()

        cam.set_single_parameter("subarray_mode", CONFIG["camera"]["subarray_mode"])
        cam.set_single_parameter("subarray_hsize", CONFIG["camera"]["subarray_size"][0])
        cam.set_single_parameter("subarray_vsize", CONFIG["camera"]["subarray_size"][1])
        cam.set_single_parameter("subarray_hpos", int((CONFIG["camera"]["camera_size"] / 2 - CONFIG["camera"]["subarray_size"][0] / 2)))
        cam.set_single_parameter("subarray_vpos", int((CONFIG["camera"]["camera_size"] / 2 - CONFIG["camera"]["subarray_size"][1] / 2)))
        cam.set_single_parameter("sensor_mode", CONFIG["camera"]["sensor_mode"])
        cam.set_single_parameter("exposure_time", CONFIG["camera"]["exposure_time"])
        cam.set_single_parameter("trigger_source", CONFIG["camera"]["trigger_source"])
        cam.set_single_parameter("trigger_polarity", CONFIG["camera"]["trigger_polarity"])
        cam.set_single_parameter("internal_line_interval", CONFIG["camera"]["internal_line_interval"])

        camera_trigger = MainCameraTrigger()
        camera_trigger.start_trigger(CONFIG["camera"]["trigger_delay"], CONFIG["DM"]["signal_freq"])

        cam.start_live()
        print("Camera initiated")

        plt.ion()
        fig, ax = plt.subplots()
        display = ax.imshow(np.zeros(CONFIG["camera"]["subarray_size"]), cmaps = 'gray')
        plt.show()

        while 1:
            t = time.perf_counter()
            frame = cam.get_last_live_frame()
            if frame is not None:
                frame_queue.put(frame)
                display.set_data(frame.astype('float'))
                fig.canvas.flush_events()
            print(f"Frame rate is: {1/(time.perf_counter() - t)}")
            time.sleep(0.001)

    except Exception as e:
        print(f"Error in camera live: {e}")

    finally:
        plt.close()
        plt.ioff()
        cam.stop_live()
        camera_trigger.stop_trigger()

if __name__ == '__main__':

    frame_queue = mp.Queue(maxsize = 10)

    try:

        dm_thread = mp.Process(target = dm_control_process)
        dm_thread.start()
        print("DM sequence running...")

        time.sleep(0.1)

        cam_thread = mp.Process(target = camera_control_process, args = (frame_queue, ))
        cam_thread.start()
        print("Camera live recording...")

        while 1:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        dm_thread.terminate()
        cam_thread.terminate()