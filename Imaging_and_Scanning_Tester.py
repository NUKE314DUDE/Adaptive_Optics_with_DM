import os
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, FrequencyUnits, Level, Slope, TriggerType
from nidaqmx.errors import DaqError

from Main_Camera_Control_Modules import mainCamera, test_live_feed_thread, test_stop_live_thread
from multiprocessing.sharedctypes import RawArray
from DM_Control_Modules import AlPaoDM, smoothed_sawtooth
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

class MainCameraTrigger:
    def __init__(self):
        self.NI_task = None
        self.NI_counter_channel = 'Dev1/ctr0'
        self. NI_trigger_source = 'pfi3'
        self.NI_min_duty_cycle = 2e-6
        self.NI_max_duty_cycle = 0.999999

    def start_trigger(self, freq, cycle_fill):
        try:
            if self.NI_task is not None:
                self.NI_task.stop()
                self.NI_task.close()
                self.NI_task = None

            if freq <= 0:
                raise ValueError('Trigger frequency must be positive!')
            if cycle_fill < 0:
                raise ValueError('Phase must be non-negative!')

            ni_freq = 2.0 * freq
            cycle_fill_max = (1 - 2 * (1 - self.NI_max_duty_cycle)) / (ni_freq * 1e3)
            cycle_fill_min = (self.NI_min_duty_cycle / 2) / (ni_freq * 1e3)

            if cycle_fill < cycle_fill_min or cycle_fill > cycle_fill_max:
                raise ValueError(f'Cycle ratio should be between {cycle_fill_min} and {cycle_fill_max}')

            ni_duty_cycle = 1.0 - (cycle_fill * 1e-3 * ni_freq)
            ni_duty_cycle = max(self.NI_min_duty_cycle, min(ni_duty_cycle, self.NI_max_duty_cycle))

            self.NI_task = nidaqmx.Task()
            self.NI_task.co_channels.add_co_pulse_chan_freq(
                counter = self.NI_counter_channel,
                name_to_assign_to_channel = "MainCamera NI trigger",
                units = FrequencyUnits.HZ,
                idle_state = Level.LOW,
                initial_delay = 0.0,
                freq = ni_freq,
                duty_cycle = ni_duty_cycle
            )

            self.NI_task.timing.cfg_implicit_timing(
                sample_mode = AcquisitionType.FINITE,
                samps_per_chan = 1
            )

            self.NI_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source = self.NI_trigger_source,
                trigger_edge = Slope.RISING
            )

            self.NI_task.triggers.start_trigger.retriggerable = True
            self.NI_task.start()

        except DaqError as e:
            print(f'NI-DAQmx Error: {e}')
            if self.NI_task is not None:
                self.NI_task.close()
                self.NI_task = None

        except ValueError as e:
            print(f'Parameter Error: {e}')
        except Exception as e:
            print(f'Unexpected Error: {e}')

    def stop_trigger(self):
        if self.NI_task is not None:
            self.NI_task.stop()
            self.NI_task.close()
            self.NI_task = None

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
    main_cam.set_single_parameter("exposure_time", 10.0) # Units in ms

    main_cam.set_single_parameter("sensor_mode", 12.0)
    main_cam.set_single_parameter("readout_direction", 1)
    main_cam.set_single_parameter("trigger_polarity", 2)
    main_cam.set_single_parameter("trigger_source", 1)
    main_cam.set_single_parameter("internal_line_interval", 10*1e-6)


    DM = AlPaoDM()
    AMP = 0.7
    SIG_FREQ = 200

    amp_modulation = smoothed_sawtooth(cut_freq_low = 10000, sig_freq = SIG_FREQ)
    seq_length = len(amp_modulation)
    seq = np.zeros((27, seq_length))
    seq[3] = AMP * amp_modulation
    DM.send_zernike_patterns(seq, repeat = 0)

    trigger = MainCameraTrigger()
    trigger.start_trigger(SIG_FREQ, 0.8)

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
        stopper.clear()
        live_feed.join()
        stop_thread.join()
        DM.stop_loop()
        plt.ioff()
