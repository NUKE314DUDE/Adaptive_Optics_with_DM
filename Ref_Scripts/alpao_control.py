import numpy
import ctypes
import time
from os.path import isfile
from multiprocessing import RawArray,Process

def alpao_loop_single(sequence_raw, stop_raw):
    lib = ctypes.cdll.LoadLibrary('Lib64/ASDK.dll')
    lib.asdkInit.restype = ctypes.POINTER(ctypes.c_void_p)
    asdk_dm = lib.asdkInit("BOL131".encode("utf-8"))
    lib.asdkSet(asdk_dm, "daqFreq".encode("utf-8"), ctypes.c_double(10000000.0))
    lib.asdkSet(asdk_dm, "SyncMode".encode("utf-8"), ctypes.c_double(1.0))


    stop = numpy.frombuffer(stop_raw, dtype="uint32")
    sequence = numpy.frombuffer(sequence_raw, dtype="float64")
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


def alpao_loop_sequence(sequence_raw, len_raw, repeats_raw, stop_raw, trigger_in=0):

    len = numpy.frombuffer(len_raw, dtype="uint32")
    repeats = numpy.frombuffer(repeats_raw, dtype="uint32")
    stop = numpy.frombuffer(stop_raw, dtype="uint32")
    sequence = numpy.frombuffer(sequence_raw, dtype="float64").reshape((len[0], 69))

    lib = ctypes.cdll.LoadLibrary('Lib64/ASDK.dll')
    lib.asdkInit.restype = ctypes.POINTER(ctypes.c_void_p)
    asdk_dm = lib.asdkInit("BOL131".encode("utf-8"))
    lib.asdkSet(asdk_dm, "daqFreq".encode("utf-8"), ctypes.c_double(20000000.0))
    # lib.asdkSet(asdk_dm, "NbSteps".encode("utf-8"), ctypes.c_double(5.0))
    lib.asdkSet(asdk_dm, "SyncMode".encode("utf-8"), ctypes.c_double(1.0))
    lib.asdkSet(asdk_dm, "Timeout".encode("utf-8"), ctypes.c_double(60.0))
    lib.asdkSet(asdk_dm, "TriggerIn".encode("utf-8"), ctypes.c_double(trigger_in))


    c=0


    while stop[0]==0 and (c<repeats[0] or repeats==0):
        output=lib.asdkSendPattern(asdk_dm,
                                   sequence.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_uint32(len[0]),
                                   ctypes.c_uint32(1))
        if output!=0:
            lib.asdkPrintLastError()


        c+=1


    lib.asdkRelease(asdk_dm)



class AlpaoDM:
    def __init__(self):
        self.patterns_raw=None
        self.len_raw=None
        self.repeats_raw = None
        self.stop_raw=None
        self.patterns=None
        self.len = None
        self.stop = None
        self.repeats= None
        self.alpao_process=None
        if isfile("flat_voltages.npy"):
            self.flat_voltages=numpy.load("flat_voltages.npy")
        else:
            self.flat_voltages=numpy.zeros(69)
        self.zer_mat = numpy.load("zernike_matrix.npy")

    def send_pattern_zernikes(self, zernikes, repeats, trigger=0):
        voltages = numpy.einsum("ij,kj->ik", zernikes, self.zer_mat)
        self.send_pattern_voltages(voltages, repeats,trigger)

    def send_pattern_voltages(self, voltages, repeats, trigger=0):
        voltages=voltages+self.flat_voltages[None, :]
        self.patterns_raw=RawArray("c", voltages.shape[0]*69*numpy.dtype("float64").itemsize)
        self.len_raw=RawArray("c", numpy.dtype("uint32").itemsize)
        self.repeats_raw = RawArray("c", numpy.dtype("uint32").itemsize)
        self.stop_raw=RawArray("c", numpy.dtype("uint32").itemsize)
        self.patterns=numpy.frombuffer(self.patterns_raw).reshape((voltages.shape[0],69))
        self.patterns[:,:]=voltages
        self.len = numpy.frombuffer(self.len_raw, dtype="uint32")
        self.stop = numpy.frombuffer(self.stop_raw, dtype="uint32")
        self.repeats = numpy.frombuffer(self.repeats_raw, dtype="uint32")
        self.len[0] = voltages.shape[0]
        self.repeats[0] = repeats
        self.stop[0] = 0

        self.process = Process(target=alpao_loop_sequence, args=(self.patterns_raw, self.len_raw, self.repeats_raw,self.stop_raw, trigger))
        self.process.start()

    def update_pattern_voltages(self,voltages):
        self.patterns[:, :] = voltages+self.flat_voltages[None, :]

    def update_pattern_zernikes(self,zernikes):
        voltages = numpy.einsum("ij,kj->ik", zernikes, self.zer_mat)+self.flat_voltages[None, :]
        self.patterns[:, :] = voltages

    def start_direct_control(self):
        self.patterns_raw=RawArray("c", 69*numpy.dtype("float64").itemsize)
        self.stop_raw=RawArray("c", numpy.dtype("uint32").itemsize)
        self.patterns=numpy.frombuffer(self.patterns_raw)
        self.stop = numpy.frombuffer(self.stop_raw, dtype="uint32")
        self.stop[0] = 0

        self.process = Process(target=alpao_loop_single,
                               args=(self.patterns_raw, self.stop_raw))
        self.process.start()

    def send_direct_voltages(self, voltages):
        self.patterns[:]=voltages+self.flat_voltages

    def send_direct_zernikes(self, zernikes):
        voltages=numpy.dot(self.zer_mat, zernikes)+self.flat_voltages
        self.patterns[:]=voltages

    def stop_loop(self):
        if self.stop is not None:
            self.stop[0] = 1




if __name__ == "__main__":
    # from PyDAQmx import *
    import time
    delay = 0.1
    #
    # t_ctr_o=Task()
    # t_ctr_o.CreateCOPulseChanFreq("Dev1/ctr0", None, DAQmx_Val_Hz, DAQmx_Val_Low, delay, 100.0, 0.5)
    # t_ctr_o.CfgImplicitTiming(DAQmx_Val_FiniteSamps,1)
    # t_ctr_o.CfgDigEdgeStartTrig("pfi3",DAQmx_Val_Rising)
    # t_ctr_o.SetStartTrigRetriggerable(True)
    #
    # t_ctr_o.StartTask()

    values_tip = numpy.sin(numpy.linspace(0, 2 * numpy.pi, 30000))
    values_tilt = numpy.cos(numpy.linspace(0, 2 * numpy.pi, 30000))

    sequence=numpy.zeros((30000,27))
    sequence[:,0]=values_tip
    sequence[:, 1] = values_tilt

    dm=AlpaoDM()
    dm.send_pattern_zernikes(sequence,0)

    time.sleep(5)

    input()

    dm.stop_loop()

    # dm.send_pattern_zernikes(values[:,None]*zer_coeffs[None,:],0)
    #
    # time.sleep(5)
    #
    # dm.stop_loop()




