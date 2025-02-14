import os
import time
from multiprocessing.sharedctypes import RawArray
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Main_Camera_Control_Modules import MainCamera, MainCameraTrigger
from DM_Control_Modules import AlPaoDM, smoothed_sawtooth
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

CONFIG = {
    "DM" : {
        "zernike_order" : 3,
        "amp" : 0.95,
        "signal_freq" : 120,
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
        "exposure_time" : 10.0, # ms
        "trigger_source" : 1,
        "trigger_polarity" : 2,
        "trigger_delay" : 8e-6,
        "internal_line_interval" : 1*1e-6
    }
}

def dm_control_process(stop_event):
    dm = AlPaoDM()
    try:

        amp_modulation = smoothed_sawtooth(
            fill = CONFIG["DM"]["sawtooth_params"]["fill"],
            cut_freq_low = CONFIG["DM"]["sawtooth_params"]["cut_freq_low"],
            cut_freq_high = CONFIG["DM"]["sawtooth_params"]["cut_freq_high"],
            sig_freq = CONFIG["DM"]["signal_freq"]
        )

        dm_sequence = np.zeros((27, len(amp_modulation)))

        dm_sequence[CONFIG["DM"]["zernike_order"]] = CONFIG["DM"]["amp"] * amp_modulation

        dm.send_zernike_patterns(dm_sequence, repeat = 0)
        print("DM sequence initiated...")

        while not stop_event.is_set():
            time.sleep(0.001)

    except Exception as e:
        print(f"Error in DM sequence: {e}")

    finally:
        dm.stop_loop()
        print("DM closed...")

def camera_control_process(img, stop_event):

    cam = MainCamera()
    camera_trigger = MainCameraTrigger()

    try:

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

        camera_trigger.start_trigger(CONFIG["camera"]["trigger_delay"], CONFIG["DM"]["signal_freq"])

        cam.start_live()
        print("Camera initiated...")

        plt.ion()
        fig, ax = plt.subplots()
        display = ax.imshow(np.zeros(CONFIG["camera"]["subarray_size"]), cmap = 'gray')
        fig.tight_layout()
        plt.show(block = False)

        while not stop_event.is_set():
            carrier = cam.get_last_live_frame()
            if carrier is not None:

                try:

                    img[:, :] = carrier[:, :]
                    display.set_data(img.astype("float"))
                    fig.canvas.flush_events()
                    time.sleep(0.001)

                except Exception as e:
                    print(f"Error in saving frame: {e}")

            time.sleep(0.001)

    except Exception as e:
        print(f"Error in camera live: {e}")

    finally:
        try:
            plt.close()
            plt.ioff()
            cam.stop_live()
            camera_trigger.stop_trigger()
            print("Camera closed...")
        except Exception as e:
            print(f"Error closing camera: {e}")

def keyboard_listener(stop_event):
    while not stop_event.is_set():
        try:
            time.sleep(0.001)
        except EOFError:
            break

if __name__ == '__main__':

    stopper = mp.Event()
    stopper_thread = mp.Process(target = keyboard_listener, args = (stopper,))
    stopper_thread.daemon = True
    stopper_thread.start()

    frame_raw = RawArray("H", CONFIG["camera"]["subarray_size"][0] * CONFIG["camera"]["subarray_size"][1])
    frame = np.frombuffer(frame_raw, dtype = "uint16").reshape(CONFIG["camera"]["subarray_size"])

    dm_thread = mp.Process(target = dm_control_process, args = (stopper, ))
    dm_thread.start()

    camera_thread = mp.Process(target = camera_control_process, args = (frame, stopper))
    camera_thread.start()

    try:
        while not stopper.is_set():
            time.sleep(0.01)
    except KeyboardInterrupt:
        stopper.set()
    finally:
        dm_thread.join()
        camera_thread.join()