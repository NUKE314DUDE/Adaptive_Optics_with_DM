import time
import threading
from multiprocessing.sharedctypes import RawArray
import numpy as np
from hamamatsu.dcam import dcam, copy_frame, Stream
import matplotlib.pyplot as plt

## credit to Vipin Balan ##

def test_stop_live_thread(cam, stop_event):
    while not stop_event.is_set():
        key = input()
        if key == 'x':
            print('Stopping Camera live...')
            stop_event.set()
            cam.stop_live()
            cam.camera_close()
            break

def test_live_feed_thread(cam, img, stop_event):
    while not stop_event.is_set():
        try:
            carrier = cam.get_last_live_frame()
            if carrier is not None:
                img[:, :] = carrier[:]
        except Exception as e:
            print(f"Error test live feed thread: {e}")
            break

class mainCamera:
    def __init__(self):
        self.number_of_frames = None
        self.camera = None
        self.dcam = None
        self.stream = None
        self.tstream = None
        self.frame = None
        self.last_frame_idx = None
        self.frame_size = (2048, 2048)

    def camera_open(self):
        try:
            self.dcam = dcam.__enter__()
            self.camera = dcam[0].__enter__()
        except Exception as e:
            print(f"Error in camera open: {e}")

    def camera_close(self):
        try:
            if self.dcam:
                self.dcam.__exit__(None, None, None)
            if self.camera:
                self.camera.__exit__(None, None, None)
        except Exception as e:
            print(f"Error in camera close: {e}")

    def set_all_parameters(self,sensor_mode = 1.0,
                           exposure_time = 1e-3,
                           trigger_source = 1,
                           subarray_mode = 2,
                           h_size = 2304, v_size = 2304,
                           trigger_delay = 0,
                           trigger_polarity = 1,
                           internal_line_interval = 10e-6,
                           subarray_hpos = 0, subarray_vpos = 0,
                           readout_speed = 1, parameter = None, value = None):
        try:

            self.camera._buf_release()
            self.camera["sensor_mode"] = sensor_mode
            self.camera["readout_speed"] = readout_speed
            self.camera["exposure_time"] = exposure_time
            self.camera["trigger_source"] = trigger_source  #1
            self.camera["trigger_polarity"] = trigger_polarity  #1
            self.camera["subarray_mode"] = 1    #2
            self.camera["subarray_hsize"] = h_size
            self.camera["subarray_vsize"] = v_size
            self.camera["subarray_hpos"] = subarray_hpos    #512
            self.camera["subarray_vpos"] = subarray_vpos
            self.camera["subarray_mode"] = subarray_mode
            self.frame_size = (v_size, h_size)
            self.camera["trigger_delay"] = trigger_delay
            self.camera["internal_line_interval"] = internal_line_interval #9.74436090225564e-06*2
        except Exception as e:
            print(f"Error in set all parameters: {e}")

    def get_single_parameter(self, parameter_name):
        try:
            return self.camera[parameter_name].read()
        except Exception as e:
            print(f"Error in get single parameter: {e}")
            return None

    def get_single_parameter_limit(self, parameter_name):
        try:
            parameter = self.camera[parameter_name].read()
            return parameter["min_value"], parameter["max_value"]
        except Exception as e:
            print(f"Error in get single parameter limit: {e}")
            return None, None

    def set_single_parameter(self, parameter_name, parameter_value):
        try:
            self.camera[parameter_name] = parameter_value
            if parameter_name == "subarray_hsize":
                self.frame_size = (parameter_value, self.frame_size[1])
            if parameter_name == "subarray_vsize":
                self.frame_size = (self.frame_size[0], parameter_value)
        except Exception as e:
            print(f"Error in set single parameter: {e}")

    def prepare_acquisition(self, number_of_frames = 10):
        try:
            self.number_of_frames = number_of_frames
            self.frame = np.zeros((number_of_frames, self.frame_size[0], self.frame_size[1]), dtype = 'uint16')
            self.camera._buf_release()
            self.stream = Stream(self.camera, number_of_frames)
        except Exception as e:
            print(f"Error in prepare acquisition: {e}")

    def acquire_sequence(self):
        try:
            t = time.perf_counter()
            self.camera.start()
            for i, frame_buf in enumerate(self.stream):
                if i == 0: print(f"sequence started at {time.perf_counter() - t}")
                if self.camera["sensor_mode"].read() in [1, 12, 14, 16]:
                    self.frame[i] = copy_frame(frame_buf).reshape(self.frame_size)
            self.camera.stop() #check the "start" and “stop” handle in hamamatsu

            return self.frame
        except Exception as e:
            print(f"Error in acquire sequence: {e}")
            return None

    def start_live(self):
        try:
            time.sleep(1)
            self.camera._buf_release()
            self.stream = Stream(self.camera, 50)
            self.tstream = self.camera.transfer_stream()
            self.camera.start(live = True)
        except Exception as e:
            print(f"Error in start live: {e}")

    def get_last_live_frame(self):
        try:
            while self.stream.event_stream.__next__()==0:
                time.sleep(0.00001)

            current_frame_idx = next(self.tstream).nNewestFrameIndex
            while current_frame_idx == self.last_frame_idx:
                time.sleep(0.001)
                current_frame_idx = next(self.tstream).nNewestFrameIndex
            self.last_frame_idx = current_frame_idx
            if self.camera["sensor_mode"].read() in [1, 12, 14, 16]:
                return copy_frame(self.camera._lock_frame_index(current_frame_idx))
        except Exception as e:
            print(f"Error in get last live frame: {e}")

    def get_last_sequence_frame(self):
        try:
            while self.stream.event_stream.__next__()==0:
                time.sleep(0.00001)
            current_frame_idx = next(self.tstream).nNewestFrameIndex
            while current_frame_idx == self.last_frame_idx:
                time.sleep(0.001)
                current_frame_idx = next(self.tstream).nNewestFrameIndex
            if self.last_frame_idx is None:
                self.last_frame_idx = current_frame_idx
            else:
                self.last_frame_idx += 1
                if self.last_frame_idx >49:
                    self.last_frame_idx = 0
            if self.camera["sensor_mode"].read() in [1, 12, 14, 16]:
                return copy_frame(self.camera._lock_frame_index(current_frame_idx))
        except Exception as e:
            print(f"Error in get last sequence frame: {e}")

    def stop_live(self):
        try:
            self.camera.stop()
        except Exception as e:
            print(f"Error in stop live: {e}")

if __name__ == "__main__":
    CAM_SIZE = 2304
    IMG_SIZE = (1024, 480)
    image_16Raw = RawArray('H', IMG_SIZE[0] * IMG_SIZE[1])
    frame = np.frombuffer(image_16Raw, dtype='uint16').reshape(IMG_SIZE)

    main_cam = mainCamera()
    main_cam.camera_open()
    main_cam.set_single_parameter("subarray_mode", 2.0)
    main_cam.set_single_parameter("subarray_hsize", IMG_SIZE[0])
    main_cam.set_single_parameter("subarray_vsize", IMG_SIZE[1])
    main_cam.set_single_parameter("subarray_hpos", int((CAM_SIZE / 2 - IMG_SIZE[0] / 2)))
    main_cam.set_single_parameter("subarray_vpos", int((CAM_SIZE / 2 - IMG_SIZE[1] / 2)))
    main_cam.set_single_parameter("sensor_mode", 12.0)
    main_cam.set_single_parameter("exposure_time", 10.0)

    main_cam.start_live()

    test_img = main_cam.get_last_live_frame()
    if test_img is None: print('Check camera setting!')

    for par in main_cam.camera:
        if "enum_values" in main_cam.camera[par].keys():
            print(main_cam.camera[par]["uname"],main_cam.camera[par]['enum_values'],main_cam.camera[par]['default_value'])
        else:
            print(main_cam.camera[par]["uname"],main_cam.camera[par]['min_value'],main_cam.camera[par]['max_value'],main_cam.camera[par]['default_value'])


    stopper = threading.Event()

    stop_thread = threading.Thread(target=test_stop_live_thread, args = (main_cam, stopper))
    stop_thread.daemon = True
    stop_thread.start()

    live_feed = threading.Thread(target=test_live_feed_thread, args=(main_cam, frame, stopper))
    live_feed.daemon = True
    live_feed.start()

    plt.ion()
    fig, ax = plt.subplots()
    live_img = ax.imshow(frame)
    plt.show()
    try:
        while not stopper.is_set():
            live_img.set_data(frame.astype("float"))
            fig.canvas.flush_events()
            time.sleep(0.001)
    except KeyboardInterrupt:
        print('User interrupt...')
    finally:
        plt.close()
        stopper.set()
        stopper.clear()
        stop_thread.join()
        live_feed.join()
        plt.ioff()
