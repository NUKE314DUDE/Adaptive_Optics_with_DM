import os
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from Main_Camera_Control_Modules import mainCamera, test_live_feed_thread, test_stop_live_thread, MainCameraTrigger
from multiprocessing.sharedctypes import RawArray
from DM_Control_Modules import AlPaoDM, smoothed_sawtooth
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

# class MainCameraTrigger:
#     def __init__(self):
#         self.NI_input_task = None
#         self.NI_output_task = None
#         self.NI_input = '/Dev1/pfi3'
#         self.NI_output = '/Dev1/ctr0'
#         self.min_delay = 25e-9
#         self.max_delay = 53.7
#
#     def start_trigger(self, delay, DM_freq):
#         try:
#             if self.NI_input_task or self.NI_output_task is not None:
#                 self.NI_input_task.stop()
#                 self.NI_input_task.close()
#                 self.NI_input_task = None
#                 self.NI_output_task.stop()
#                 self.NI_output_task.close()
#                 self.NI_output_task = None
#
#             delay = min(max(delay, self.min_delay), min(self.max_delay, 1 / DM_freq))
#
#             self.NI_input_task = nidaqmx.Task()
#
#             self.NI_input_task.di_channels.add_di_chan(
#                 self.NI_input,
#                 name_to_assign_to_lines = "DM Trigger Input"
#             )
#
#
#             self.NI_output_task = nidaqmx.Task()
#             self.NI_output_task.co_channels.add_co_pulse_chan_time(
#                 counter = self.NI_output,
#                 name_to_assign_to_channel = "MainCmera NI Trigger",
#                 units = TimeUnits.SECONDS,
#                 idle_state = Level.LOW,
#                 initial_delay = delay,
#                 low_time = 1e-7,
#                 high_time = 1e-7
#             )
#
#             self.NI_output_task.triggers.start_trigger.cfg_dig_edge_start_trig(
#                 trigger_source = self.NI_input,
#                 trigger_edge = Slope.RISING
#             )
#
#             self.NI_output_task.triggers.start_trigger.retriggerable = True
#
#             self.NI_input_task.start()
#             self.NI_output_task.start()
#
#         except DaqError as e:
#             print(f'NI-DAQmx Error: {e}')
#             self.stop_trigger()
#
#         except ValueError as e:
#             print(f'Parameter Error: {e}')
#         except Exception as e:
#             print(f'Unexpected Error: {e}')
#
#     def stop_trigger(self):
#         if self.NI_input_task or self.NI_output_task is not None:
#             self.NI_input_task.stop()
#             self.NI_input_task.close()
#             self.NI_input_task = None
#             self.NI_output_task.stop()
#             self.NI_output_task.close()
#             self.NI_output_task = None

if __name__ == '__main__':
    CAM_SIZE = 2304
    IMG_SIZE = (1024, 1024)
    live_frame_raw = RawArray('H', IMG_SIZE[0] * IMG_SIZE[1])
    live_frame = np.frombuffer(live_frame_raw, dtype='uint16').reshape(IMG_SIZE)

    main_cam = mainCamera()
    main_cam.camera_open()
    main_cam.set_single_parameter("subarray_mode", 2.0)
    main_cam.set_single_parameter("subarray_hsize", IMG_SIZE[0])
    main_cam.set_single_parameter("subarray_vsize", IMG_SIZE[1])
    main_cam.set_single_parameter("subarray_hpos", int((CAM_SIZE / 2 - IMG_SIZE[0] / 2)))
    main_cam.set_single_parameter("subarray_vpos", int((CAM_SIZE / 2 - IMG_SIZE[1] / 2)))
    main_cam.set_single_parameter("sensor_mode", 12.0)
    main_cam.set_single_parameter("exposure_time", 10.0)

    main_cam.set_single_parameter("trigger_polarity", 2.)
    main_cam.set_single_parameter("trigger_mode", 1.)
    main_cam.set_single_parameter("trigger_source", 2)
    main_cam.set_single_parameter("internal_line_interval", 1*1e-7)


    # DM = AlPaoDM()
    # AMP = 0.9
    SIG_FREQ = 100
    #
    # amp_modulation = smoothed_sawtooth(cut_freq_low = 640, sig_freq = SIG_FREQ)
    # seq_length = len(amp_modulation)
    # seq = np.zeros((27, seq_length))
    # seq[3] = AMP * amp_modulation
    # DM.send_zernike_patterns(seq, repeat = 0)

    main_cam.start_live()

    trigger = MainCameraTrigger()
    trigger.start_trigger(0.001, SIG_FREQ)


    stopper = threading.Event()

    live_feed = threading.Thread(target = test_live_feed_thread, args = (main_cam, live_frame, stopper))
    live_feed.daemon = True; live_feed.start()

    stop_thread = threading.Thread(target = test_stop_live_thread, args = (main_cam, stopper))
    stop_thread.daemon = True; stop_thread.start()

    fig = plt.figure(figsize = (10, 6.18))
    ax = fig.add_subplot(1,1,1)
    monitor = ax.imshow(live_frame)
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
        # stopper.clear()
        DM.stop_loop()
        plt.ioff()
