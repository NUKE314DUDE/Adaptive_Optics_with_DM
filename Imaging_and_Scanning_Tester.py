import os
from multiprocessing.sharedctypes import RawArray
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
import numpy as np
import threading
import matplotlib.pyplot as plt
from Main_Camera_Control_Modules import mainCamera, live_feed_thread
from DM_Control_Modules import AlPaoDM, smoothed_sawtooth

if __name__ == '__main__':
    CAM_SIZE = 2304
    IMG_SIZE = (CAM_SIZE, 480)
    live_frame_raw = RawArray('H', IMG_SIZE[0] * IMG_SIZE[1])
    live_frame = np.frombuffer(live_frame_raw, dtype='uint16').reshape(IMG_SIZE)
    DM = AlPaoDM()
    Cam = mainCamera()
    Cam.set_single_parameter("subarray_mode", 2)
    Cam.set_single_parameter("subarray_hsize", IMG_SIZE[0])
    Cam.set_single_parameter("subarray_vsize", IMG_SIZE[1])
    Cam.set_single_parameter("subarray_hpos", int((CAM_SIZE/2 - IMG_SIZE[0]/2)))
    Cam.set_single_parameter("subarray_vpos", int((CAM_SIZE/2 - IMG_SIZE[1]/2)))
    Cam.set_single_parameter("sensor_mode", 12.0)
    Cam.set_single_parameter("exposure_time", 10.0)
    amp_modulation = smoothed_sawtooth(cut_freq_low = 10000, sig_freq = 4)
    seq_length = len(amp_modulation)
    seq = np.zeros((27, seq_length))
    seq[2] = amp_modulation
    DM.send_zernike_patterns(seq, repeat = 0)
    Cam.camera_open()
    Cam.set_all_parameters()
    Cam.start_live()

    live_feed = threading.Thread(target = live_feed_thread, args = (Cam, live_frame))
    live_feed.daemon = True; live_feed.start()

    plt.ion()
    fig, ax = plt.figure(figsize = (10, 6.18))
    monitor = ax.imshow(live_frame, cmap='gray')

    while True:
        monitor.set_data(live_frame.astype('float'))
        fig.canvas.flush_events()


