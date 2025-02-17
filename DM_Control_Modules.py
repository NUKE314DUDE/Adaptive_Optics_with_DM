import time
import numpy as np
import ctypes
import os.path
import multiprocessing as mp
from Misc_Tools import pass_filter, normalization
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

RAN = 0.25
DOF = 57
SN = "BAX758"
ZERN_to_VOLT_MATRIX_PATH = f"Data_Deposit/range_{RAN}_zernike_to_voltage.npy"

def smoothed_sawtooth(fill = 0.95, cut_freq_low = None, cut_freq_high = None, sig_freq = None, dm_sample_duration = None):
    """
    Generate smoothed (or original) sawtooth signal
    :param fill: ratio of the rising edge
    :param cut_freq_low:
    :param cut_freq_high:
    :param sig_freq:
    :param dm_sample_duration:
    :return: normalized and smoothed sawtooth signal
    """
    if sig_freq is None: sig_freq = 640
    if dm_sample_duration is None: dm_sample_duration = 6.67e-6# 6.67e-6
    sig_length = int(np.ceil((1 / sig_freq) / dm_sample_duration))
    l1 = int(np.ceil(fill * sig_length)); l2 = sig_length - l1
    l1 = l1 + 1 + l1%2; l2 = l2 + 1 + l2%2
    # defocus_seq = np.concatenate((np.linspace(-1.0, 1.0, l1),
    #                              np.linspace(1.0, -1.0, l2)))
    t1 = np.bartlett(l1)[: l1 // 2 + 1]
    t2 = np.bartlett(l2)[l2 // 2 + 1:]
    defocus_seq = np.concatenate((t1, t2))
    padding_size = len(defocus_seq) // 2
    defocus_seq = np.pad(defocus_seq, padding_size, mode = 'wrap')
    if cut_freq_low is None and cut_freq_high is None:
        return normalization(defocus_seq[padding_size:-padding_size])
    else:
        smoothed_defocus_seq = pass_filter(defocus_seq, 1 / dm_sample_duration, cut_freq_low, cut_freq_high)
        return normalization(smoothed_defocus_seq[padding_size:-padding_size])

def alpao_loop_single(sequence_raw, stop_raw):

    lib = ctypes.cdll.LoadLibrary('Lib64/ASDK.dll')
    lib.asdkInit.restype = ctypes.POINTER(ctypes.c_void_p)
    asdk_dm = lib.asdkInit(SN.encode("utf-8"))
    lib.asdkSet(asdk_dm, "daqFreq".encode("utf-8"), ctypes.c_double(20*1e6))
    lib.asdkSet(asdk_dm, "SyncMode".encode("utf-8"), ctypes.c_double(1.0))

    stop = np.frombuffer(stop_raw, dtype="uint32")
    sequence = np.frombuffer(sequence_raw, dtype="float64")
    if np.any(sequence > 1):
        print('Voltages contain element(s) larger than 1!')
    elif np.nan in sequence:
        print('Voltages contain nan element(s)')

    c=0

    while stop[0]==0:
        output=lib.asdkSendPattern(asdk_dm,
                                   sequence.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_uint32(1),
                                   ctypes.c_uint32(1))
        if output!=0:
            lib.asdkPrintLastError()

        c+=1

    lib.asdkRelease(asdk_dm)

def alpao_loop_sequence(sequence_raw, length_raw, repeats_raw, stop_raw, trigger_in = 0):

    length = np.frombuffer(length_raw, dtype="uint32")
    # print(length.size)
    repeats = np.frombuffer(repeats_raw, dtype="uint32")
    stop = np.frombuffer(stop_raw, dtype="uint32")
    sequence = np.frombuffer(sequence_raw, dtype="float64").reshape((length[0], DOF))
    if np.any(sequence > 1):
        print('Voltages have elements larger than 1!')
    elif np.nan in sequence:
        print('Voltages contain nan element(s)')
    lib = ctypes.cdll.LoadLibrary('Lib64/ASDK.dll')
    lib.asdkInit.restype = ctypes.POINTER(ctypes.c_void_p)
    asdk_dm = lib.asdkInit(SN.encode("utf-8"))
    lib.asdkSet(asdk_dm, "daqFreq".encode("utf-8"), ctypes.c_double(20*1e6))
    lib.asdkSet(asdk_dm, "SyncMode".encode("utf-8"), ctypes.c_double(1.0))
    lib.asdkSet(asdk_dm, "Timeout".encode("utf-8"), ctypes.c_double(60.0))
    lib.asdkSet(asdk_dm, "TriggerIn".encode("utf-8"), ctypes.c_double(trigger_in))

    c=0

    while stop[0]==0 and (c<repeats[0] or repeats==0):
        output=lib.asdkSendPattern(asdk_dm,
                                   sequence.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_uint32(length[0]),
                                   ctypes.c_uint32(1))
        if output!=0:
            lib.asdkPrintLastError()

        c+=1

    lib.asdkRelease(asdk_dm)

class AlPaoDM:
    def __init__(self):
        self.process = None
        self.patterns_raw = None
        self.length_raw = None
        self.repeat_raw = None
        self.stop_raw = None
        self.patterns = None
        self.length = None
        self.stop = None
        self.repeat = None
        self.AlPao_process = None
        self.checker = None
        if os.path.isfile("Data_Deposit/zeroing_compensation_voltage.npy"):
            self.zero_compensation_voltage = np.array(np.load("Data_Deposit/zeroing_compensation_voltage.npy")).T
        else:
            self.zero_compensation_voltage = np.zeros(DOF)
        self.zern_to_volt = np.load(ZERN_to_VOLT_MATRIX_PATH)

    def send_zernike_patterns(self, zernike_orders, repeat, trigger = 0):
        """
        Based on input zernike amps, calculate zernike voltages.
        :param zernike_orders: col amps for each zernike (up to 27)
        :param repeat:
        :param trigger:
        :return:
        """
        voltages = np.einsum('ij, ik -> jk', self.zern_to_volt, zernike_orders)
        self.send_voltage_patterns(voltages, repeat, trigger)

    def send_voltage_patterns(self, voltages, repeat, trigger = 0):
        voltages += self.zero_compensation_voltage
        # self.checker = voltages[:, 100]
        # self.patterns_raw = mp.RawArray("d", voltages.shape[0]*DOF*np.dtype("float64").itemsize)
        self.patterns_raw = mp.RawArray("d", DOF * voltages.shape[1])
        self.length_raw = mp.RawArray("I", 1)
        self.repeat_raw = mp.RawArray("I", 1)
        self.stop_raw = mp.RawArray("I", 1)
        self.patterns = np.frombuffer(self.patterns_raw).reshape((voltages.shape[1], DOF))
        self.patterns[:, :] = np.array(voltages).T
        self.length = np.frombuffer(self.length_raw, dtype = 'uint32')
        self.stop = np.frombuffer(self.stop_raw, dtype = 'uint32')
        self.repeat = np.frombuffer(self.repeat_raw, dtype = 'uint32')
        self.length[0] = voltages.shape[1]
        self.repeat[0] = repeat
        self.stop[0] = 0
        self.process = mp.Process(target = alpao_loop_sequence,
                                  args = (self.patterns_raw, self.length_raw, self.repeat_raw, self.stop_raw, trigger))
        self.process.start()

    def update_pattern_voltages(self, voltages):
        self.patterns[:,:] = np.array(voltages + self.zero_compensation_voltage).T

    def update_zernike_patterns(self, zernike_orders):
        voltages = np.einsum('ij, ik -> jk', self.zern_to_volt, zernike_orders)
        self.patterns[:, :] = np.array(voltages).T

    def start_direct_control(self):
        self.patterns_raw = mp.RawArray('d', DOF)
        self.stop_raw = mp.RawArray('I', 1)
        self.patterns = np.frombuffer(self.patterns_raw)
        self.stop = np.frombuffer(self.stop_raw, dtype = 'uint32')
        self.stop[0] = 0
        self.process = mp.Process(target = alpao_loop_single,
                                  args = (self.patterns_raw, self.stop_raw))
        self.process.start()

    def send_direct_voltage(self, voltages):
        self.patterns = np.array(voltages + self.zero_compensation_voltage).T

    def send_direct_zernike(self, zernike_orders):
        voltages = np.einsum('ij, ik -> jk', self.zern_to_volt, zernike_orders)
        self.patterns[:, :] = np.array(voltages + self.zero_compensation_voltage).T

    def stop_loop(self):
        if self.stop is not None:
            self.stop[0] = 1

# Dynamic zernike patterns for testing
if __name__ == '__main__':
    time.sleep(0.05)
    seq_length = 1000
    seq = np.zeros((27, seq_length))
    test_amp_1 = np.sin(np.linspace(0, 2*np.pi, seq_length))
    test_amp_2 = np.cos(np.linspace(0, 2*np.pi, seq_length))
    seq[0, :] = test_amp_1
    seq[1, :] = test_amp_2
    DM = AlPaoDM()
    DM.send_zernike_patterns(seq, repeat = 0)
    time.sleep(4)
    input()
    DM.stop_loop()