import os
from multiprocessing.sharedctypes import RawArray
import numpy as np
import threading
import matplotlib.pyplot as plt
from Main_Camera_Control_Modules import mainCamera, test_live_feed_thread, test_stop_live_thread
from DM_Control_Modules import AlPaoDM, smoothed_sawtooth
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

if __name__ == '__main__':
    CAM_SIZE = 2304
    IMG_SIZE = (1024, 480)
    live_frame_raw = RawArray('H', IMG_SIZE[0] * IMG_SIZE[1])
    live_frame = np.frombuffer(live_frame_raw, dtype='uint16').reshape(IMG_SIZE)

    main_cam = mainCamera()
    main_cam.camera_open()
    main_cam.set_single_parameter("subarray_mode", 2)
    main_cam.set_single_parameter("subarray_hsize", IMG_SIZE[0])
    main_cam.set_single_parameter("subarray_vsize", IMG_SIZE[1])
    main_cam.set_single_parameter("subarray_hpos", int((CAM_SIZE / 2 - IMG_SIZE[0] / 2)))
    main_cam.set_single_parameter("subarray_vpos", int((CAM_SIZE / 2 - IMG_SIZE[1] / 2)))
    main_cam.set_single_parameter("sensor_mode", 12.0)
    main_cam.set_single_parameter("exposure_time", 10.0)

    DM = AlPaoDM()

    amp_modulation = smoothed_sawtooth(cut_freq_low = 10000, sig_freq = 200)
    seq_length = len(amp_modulation)
    seq = np.zeros((27, seq_length))
    seq[2] = amp_modulation
    DM.send_zernike_patterns(seq, repeat = 0)

    main_cam.start_live()

    stopper = threading.Event()

    live_feed = threading.Thread(target = test_live_feed_thread, args = (main_cam, live_frame, stopper))
    live_feed.daemon = True; live_feed.start()

    stop_thread = threading.Thread(target = test_stop_live_thread, args = (main_cam, stopper))
    stop_thread.daemon = True; stop_thread.start()

    fig = plt.figure(figsize = (10, 6.18))
    ax = fig.add_subplot(1,1,1)
    monitor = ax.imshow(live_frame, cmap='gray')
    plt.ion()
    plt.show()

    try:
        while not stopper.is_set():
            monitor.set_data(live_frame.astype('float'))
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("User interrupt")
    finally:
        plt.close()
        stopper.set()
        live_feed.join()
        stop_thread.join()
        DM.stop_loop()
        plt.ioff()
