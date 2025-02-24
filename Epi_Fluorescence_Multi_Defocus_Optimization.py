import numpy as np
import matplotlib.pyplot as plt
from DM_Control_Modules import AlPaoDM
from Main_Camera_Control_Modules import MainCamera, MainCameraTrigger
from Zernike_Polynomials_Modules import mean_gradient, input_zernike

CONFIG = {
    "OPTIMIZATION" : {
        "defocus_steps" : 11,
        "zernike_steps" : 11,
        "defocus_amp" : 0.9,
        "zernike_amp" : 0.5,
        "interp" : "cubic",
        "normalize_frame_data" : False
    },
    "CAMERA" : {
        "camera_size" : 2304,
        "subarray_mode" : 2.0,
        "subarray_size" : (1024, 1024),
        "sensor_mode" : 12.0,
        "exposure_time" : 40 * 1e-6,
        "internal_line_interval" : 1 * 1e-6,
        "trigger_source" : 1
    },
    "VISUALIZATION" : {
        "refresh_interval" : 10,
        "colormap" : "viridis",
        "plot_metrics" : True
    }
}

class EpiFluorescenceOptimization:
    def __init__(self):
        self.camera = MainCamera()
        self.dm = AlPaoDM()
        self.defocus_amps = np.linspace(-CONFIG["OPTIMIZATION"]["defocus_amp"], CONFIG["OPTIMIZATION"]["defocus_amp"],
                                        CONFIG["OPTIMIZATION"]["defocus_steps"])
        self.zernike_amps = np.linspace(-CONFIG["OPTIMIZATION"]["zernike_amp"], CONFIG["OPTIMIZATION"]["zernike_amp"],
                                        CONFIG["OPTIMIZATION"]["zernike_steps"])
        self.target_zernikes = [z for z in input_zernike if z != (2, 0)]



if __name__ == '__main__':
    optimizer = EpiFluorescenceOptimization()