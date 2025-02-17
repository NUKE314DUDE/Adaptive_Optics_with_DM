import os
import queue
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from matplotlib.animation import FuncAnimation
from Main_Camera_Control_Modules import MainCamera, MainCameraTrigger
from DM_Control_Modules import AlPaoDM, smoothed_sawtooth
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

CONFIG = {
    "DM" : {
        "zernike_order" : 3,
        "amp" : 0.8,
        "signal_freq" : 30,
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
        "exposure_time" : 0.01, # ms
        "trigger_source" : 2,
        "trigger_polarity" : 2,
        "trigger_delay" : 1e-6,
        "internal_line_interval" : 1*1e-8,
        "auto_range" : False,
        "max_frame_rate" : 1000
    }
}

class DMCameraSync:
    def __init__(self):
        self.DM = AlPaoDM()
        self.Camera = MainCamera()
        self.CameraTrigger = MainCameraTrigger()
        self.running = False
        self.frame_queue = queue.Queue(maxsize = 4)
        self.last_frame = None
        self.last_update_time = time.time()
        self.display_range = [0, 65535]

    def dm_control_thread(self, stop_event):
        try:

            amp_modulation = smoothed_sawtooth(
                fill=CONFIG["DM"]["sawtooth_params"]["fill"],
                cut_freq_low=CONFIG["DM"]["sawtooth_params"]["cut_freq_low"],
                cut_freq_high=CONFIG["DM"]["sawtooth_params"]["cut_freq_high"],
                sig_freq=CONFIG["DM"]["signal_freq"]
            )

            dm_sequence = np.zeros((27, len(amp_modulation)))
            dm_sequence[CONFIG["DM"]["zernike_order"]] = CONFIG["DM"]["amp"] * amp_modulation

            self.DM.send_zernike_patterns(dm_sequence, repeat=0)
            print("DM sequence initiated...")

            while not stop_event.is_set():
                time.sleep(0.001)

        except Exception as e:
            print(f"Error in DM thread: {str(e)}")

        finally:
            self.DM.stop_loop()
            print("DM closed...")


    def camera_control_thread(self, stop_event):
        try:
            self.Camera.camera_open()

            self.Camera.set_single_parameter("subarray_mode", CONFIG["camera"]["subarray_mode"])
            self.Camera.set_single_parameter("subarray_hsize", CONFIG["camera"]["subarray_size"][0])
            self.Camera.set_single_parameter("subarray_vsize", CONFIG["camera"]["subarray_size"][1])
            self.Camera.set_single_parameter("subarray_hpos",
                                     int((CONFIG["camera"]["camera_size"] / 2 - CONFIG["camera"]["subarray_size"][0] / 2)))
            self.Camera.set_single_parameter("subarray_vpos",
                                     int((CONFIG["camera"]["camera_size"] / 2 - CONFIG["camera"]["subarray_size"][1] / 2)))
            self.Camera.set_single_parameter("sensor_mode", CONFIG["camera"]["sensor_mode"])
            self.Camera.set_single_parameter("exposure_time",
                                             min(CONFIG["camera"]["exposure_time"], 1e3 * ((1/CONFIG["DM"]["signal_freq"]) - CONFIG["camera"]["trigger_delay"]
                                                 - 4*1e-7)))

            if CONFIG["camera"]["exposure_time"] * 1e-3 > ((1/CONFIG["DM"]["signal_freq"]) - CONFIG["camera"]["trigger_delay"]
                                                 - 4*1e-7): print("Camera is missing triggers, lower the exposure!")

            self.Camera.set_single_parameter("trigger_source", CONFIG["camera"]["trigger_source"])
            self.Camera.set_single_parameter("trigger_polarity", CONFIG["camera"]["trigger_polarity"])
            self.Camera.set_single_parameter("internal_line_interval", CONFIG["camera"]["internal_line_interval"])

            self.CameraTrigger.start_trigger(CONFIG["camera"]["trigger_delay"], CONFIG["DM"]["signal_freq"])

            self.Camera.start_live()
            print("Camera initiated...")

            frame_counter = 0
            start_time = time.time()

            while not stop_event.is_set():
                frame = self.Camera.get_last_live_frame()
                if frame is not None:
                    if CONFIG["camera"]["auto_range"]:

                        frame_min = np.min(frame); frame_max = np.max(frame)
                        self.display_range = [frame_min - 0.1 * (frame_max - frame_min),
                                              frame_max + 0.1 * (frame_max - frame_min)]

                    frame_counter += 1
                    if frame_counter % 10 == 0:

                        fps = frame_counter/(time.time() - start_time)
                        print(f"Current camera fps: {fps:.2f} FPS")
                        frame_counter = 0
                        start_time = time.time()

                    try:
                        # self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass

                time.sleep(1 / CONFIG["camera"]["max_frame_rate"])

        except Exception as e:
            print(f"Error in camera sampling: {str(e)}")

        finally:
            self.Camera.camera_close()
            self.CameraTrigger.stop_trigger()
            print("Camera closed...")

    def update_display(self, frame):
        current_time = time.time()
        # print(f"Refresh gap = {current_time - self.last_update_time:.3f} s")
        self.last_update_time = current_time
        if frame is not None:
            self.last_frame = frame
        if self.last_frame is not None:
            self.im.set_data(self.last_frame)
            self.im.set_clim(*self.display_range)
        return [self.im]

    def start(self):

        self.running = True
        stop_event = mp.Event()

        dm_thread = threading.Thread(target =self.dm_control_thread, args = (stop_event, ))
        camera_thread = threading.Thread(target = self.camera_control_thread, args = (stop_event, ))
        monitor_thread = threading.Thread(target = self._thread_monitor, args = (dm_thread, camera_thread))

        dm_thread.daemon = True; camera_thread.daemon = True; monitor_thread.daemon = True

        dm_thread.start()
        time.sleep(1)
        camera_thread.start()
        monitor_thread.start()

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            np.zeros((CONFIG["camera"]["subarray_size"])),
            cmap = "gray",
            vmin = self.display_range[0],
            vmax = self.display_range[1]
        )
        self.ax.set_title("Camera Feed")
        plt.tight_layout()

        self.ani = FuncAnimation(
            self.fig,
            self.update_display,
            frames = self._frame_generator,
            interval = 10, # ms
            blit = True,
            cache_frame_data = False
        )

        def on_key(event):
            if event.key == "x":
                print("Shutting down...")
                stop_event.set()
                self.running = False
                plt.close()
            return None

        self.fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event) or None)
        plt.show(block = False)

        while self.running:
            self.fig.canvas.flush_events()
            time.sleep(0.01)

        monitor_thread.join(); dm_thread.join(); camera_thread.join()
        print("Resource released...")

    def _frame_generator(self):
        print("Frame generator started...")
        while self.running:
            try:
                frame = self.frame_queue.get_nowait()
                # print(f"Frame grabbed with shape: {frame.shape}, data range: {np.min(frame)} to {np.max(frame)}")
                yield frame
            except queue.Empty:
                print("Frame Queue is empty.")
                yield None

    def _thread_monitor(self, dm_thread = None, camera_thread = None):
        while self.running:
            print(f"Thread status: DM[{dm_thread.is_alive()}], Camera[{camera_thread.is_alive()}]")
            time.sleep(1)

if __name__ == '__main__':
    controller = DMCameraSync()
    controller.start()