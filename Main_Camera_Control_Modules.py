import time
import numpy as np
from hamamatsu.dcam import dcam, copy_frame, Stream


class mainCamera:
    def __init__(self):
        self.camera = None
        self.dcam = None
        self.stream = None
        self.tstream = None
        self.frame = None
        self.latest_frame_idx = None
        self.frame_size = (2048, 2048)

    def camera_open(self):
        self.dcam = dcam.__enter__()
        self.camera = dcam[0].__enter__()

    def camera_close(self):
        self.dcam.__exit__(None, None, None)
        self.camera.__exit__(None, None, None)

    def set_all_parameters(self,sensor_mode = 1.0,
                           exposure_time = 2e-3,
                           trigger_source = 1,
                           subarray_mode = 2,
                           h_size = 2304, v_size = 2304,
                           trigger_delay = 0,
                           trigger_polarity = 1,
                           internal_line_interval = 1e-5,
                           subarray_hpos = 0, subarray_vpos = 0,
                           readout_speed = 1, parameter = None, value = None):
        self.camera._buf_release()
        self.camera["sensor_mode"] = sensor_mode
        self.camera["readout_speed"] = readout_speed
        self.camera["exposure_time"] = exposure_time
        self.camera["trigger_source"] = trigger_source  #1
        self.camera["trigger_polarity"] = trigger_polarity  # 1
        self.camera["subarray_mode"] = 1    #2
        self.camera["subarray_hsize"] = h_size
        self.camera["subarray_vsize"] = v_size
        self.camera["subarray_hpos"] = subarray_hpos    #512
        self.camera["subarray_vpos"] = subarray_vpos
        self.camera["subarray_mode"] = subarray_mode
        self.frame_size = (v_size, h_size)
        self.camera["trigger_delay"] = trigger_delay
        self.camera["internal_line_interval"] = internal_line_interval #9.74436090225564e-06*2

    def get_parameter(self, parameter_name):
        return self.camera[parameter_name].read()
