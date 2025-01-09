import sys
import json
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QMessageBox,\
    QTableWidgetItem, QDialog, QVBoxLayout,QHBoxLayout, QDialogButtonBox ,QDoubleSpinBox,QFormLayout,QGridLayout,QPushButton, QAction
from PyQt5.QtCore import QThread,pyqtSignal,pyqtSlot,Qt
from PyQt5.QtGui import QImage,QPixmap
import numpy
import time
from multiprocessing import RawArray,Process
from Vipin_hamamatsu_library import camera
from datetime import datetime
from os.path import isfile, splitext
from PyDAQmx import *
import h5py
from alpao_control import AlpaoDM
from pipython import GCSDevice, pitools
from scipy import signal

SCAN_DIRECTION = 1
MIRROR_SAMPLE_DURATION=6.67e-6
IMAGESIZE = (2304,480)
temp_folder = "C:\\oblique lightsheet images\\temporary\\"

def get_parameters(file_name,frequency,vsize):
    with open(file_name, "r") as infile:
        data = json.load(infile)
    return data["vsize_"+str(int(vsize)).zfill(3)]["frequency_"+str(int(frequency))]


def findminmax(image16Raw, minMaxRaw, lutRaw, changedRaw, autoRaw, stoppedRaw):
    image16=numpy.frombuffer(image16Raw, dtype="uint16")
    minMax=numpy.frombuffer(minMaxRaw, dtype="uint16")
    changed=numpy.frombuffer(changedRaw, dtype="uint8")
    stopped = numpy.frombuffer(stoppedRaw, dtype="uint8")
    lut=numpy.frombuffer(lutRaw, dtype="uint8")
    auto=numpy.frombuffer(autoRaw, dtype="uint8")
    newlut=numpy.empty(2**16, dtype="uint8")
    while stopped[0]==0:
        if changed==1:
            if auto==1:
                minMaxLocal=numpy.asarray([numpy.percentile(image16,5),numpy.percentile(image16,100)]).astype("uint16")
            else:
                minMaxLocal=numpy.copy(minMax)
            if minMaxLocal[0]>0:
                newlut[:minMaxLocal[0]]= numpy.zeros(minMaxLocal[0],dtype="uint8")
            if minMaxLocal[1]<2**16-1:
                newlut[minMaxLocal[1]:]=numpy.ones(int(2**16-float(minMaxLocal[1])),dtype="uint8")*255
            newlut[minMaxLocal[0]:minMaxLocal[1]]=numpy.linspace(0,254,minMaxLocal[1]-minMaxLocal[0]).astype("uint8")
            numpy.copyto(lut,newlut)
            numpy.copyto(changed,numpy.asarray([0]).astype("uint8"))
        else:
            time.sleep(0.001)
    stopped[0]=2


def convert16to8(image16Raw, lutRaw, image8Raw, changedRaw, stoppedRaw):
    lut=numpy.frombuffer(lutRaw, dtype="uint8")
    image16=numpy.frombuffer(image16Raw, dtype="uint16")
    image8=numpy.frombuffer(image8Raw, dtype="uint8")
    changed=numpy.frombuffer(changedRaw, dtype="uint8")
    stopped = numpy.frombuffer(stoppedRaw, dtype="uint8")
    while stopped[0]==0:
        if changed==1:
            numpy.copyto(image8,lut[image16])
            numpy.copyto(changed,numpy.asarray([0]).astype("uint8"))
        else:
            time.sleep(0.001)
    while stopped[0]!=2:
        time.sleep(0.001)
    stopped[0] = 3


def live_camera_thread(image16Raw, parametersRaw, camerastopRaw, newframeRaw):
    # parameters: 0-exposure, 1-line interval, 2-hsize, 3-vsize, 4-trigger
    cam = camera()
    cam.camera_open()
    image16 = numpy.frombuffer(image16Raw, dtype="uint16")
    stop = numpy.frombuffer(camerastopRaw, dtype="uint8")
    stop[0]=0
    newframe = numpy.frombuffer(newframeRaw, dtype="uint8")
    live_parameters = numpy.frombuffer(parametersRaw, dtype="float")
    last_parameters = numpy.copy(live_parameters)
    cam.set_one_parameter("subarray_hpos", 0)
    cam.set_one_parameter("subarray_vpos", 0)
    cam.set_all_parameters(sensor_mode=12.0,
                           hsize=int(live_parameters[2]),
                           vsize=int(live_parameters[3]),
                           trigger_polarity=2,
                           subarray_hpos=int((2304-live_parameters[2])/2),
                           subarray_vpos=int((2304-live_parameters[3])/2))
    cam.set_one_parameter("internal_line_interval", live_parameters[1]*1e-06)
    cam.set_one_parameter("exposure_time", live_parameters[0])
    cam.set_one_parameter("readout_direction", SCAN_DIRECTION)
    cam.set_one_parameter("trigger_source", int(live_parameters[4]))
    
    cam.start_live()
    t=time.perf_counter()

    while stop[0]==0:
        image16[:] = cam.get_last_live_frame().flatten()
        newframe[0]=1
        if live_parameters[0]!=last_parameters[0]:
            cam.set_one_parameter("exposure_time", live_parameters[0])
            last_parameters[0]=live_parameters[0]
        if live_parameters[1]!=last_parameters[1]:
            cam.set_one_parameter("internal_line_interval", live_parameters[1]*1e-06)
            last_parameters[1]=live_parameters[1]
        if live_parameters[4]!=last_parameters[4]:
            cam.set_one_parameter("trigger_source", int(live_parameters[4]))
            last_parameters[4]=live_parameters[4]
    cam.stop_live()
    cam.close()
    stop[0]=3


def sequence_camera_thread(filename, sync_parameters, image16Raw, parametersRaw, camerastopRaw,
                           newframeRaw, scanparametersRaw, timelapseparametersRaw, progressRaw):
    # parameters: 0-exposure, 1-line interval, 2-hsize, 3-vsize, 4-trigger
    # scan parameters: 0-active, 1-start position, 2-stop position, 3-step size
    # scan axes: 0-z, 1-x, 2-y
    # timelapse parameters: 0-active, 1-interval, 2-time points
    # sync parameters: 0-frequency, 1-phase
    stage = GCSDevice('C-413.2GA')
    stage.ConnectUSB(serialnum="119024343")
    pitools.startup(stage, stages=["V-524.1AA",], refmodes=["FRF", ])
    stage.VEL(stage.axes[0], 250.0)


    cam = camera()
    cam.camera_open()
    progress = numpy.frombuffer(progressRaw, dtype="float")
    image16 = numpy.frombuffer(image16Raw, dtype="uint16")
    stop = numpy.frombuffer(camerastopRaw, dtype="uint8")
    scan_parameters = numpy.frombuffer(scanparametersRaw, dtype="float")
    timelapse_parameters = numpy.frombuffer(timelapseparametersRaw, dtype="float")


    stop[0]=0
    newframe = numpy.frombuffer(newframeRaw, dtype="uint8")
    live_parameters = numpy.frombuffer(parametersRaw, dtype="float")
    last_parameters = numpy.copy(live_parameters)
    cam.set_all_parameters(sensor_mode=12.0,
                           hsize=int(live_parameters[2]),
                           vsize=int(live_parameters[3]),
                           trigger_polarity=2,
                           subarray_hpos=int((2304-live_parameters[2])/2),
                           subarray_vpos=int((2304-live_parameters[3])/2))
    cam.set_one_parameter("internal_line_interval", live_parameters[1]*1e-06)
    cam.set_one_parameter("exposure_time", live_parameters[0])
    cam.set_one_parameter("readout_direction", SCAN_DIRECTION)
    cam.set_one_parameter("trigger_source", 1)

    if int(timelapse_parameters[0]) == 0:
        time_points=1
    else:
        time_points=int(timelapse_parameters[2])

    if int(scan_parameters[0]) == 0:
        y_size = 1
    else:
        y_size = int(numpy.abs((scan_parameters[2]-scan_parameters[1]))/scan_parameters[3])+1

    print(y_size)
    total_images = time_points*y_size
    dataset_shape = (time_points,y_size,int(live_parameters[3]),int(live_parameters[2]))

    while isfile(filename):
        if splitext(filename)[0][-5]=="_" and splitext(filename)[0][-4:].isdigit():
            last_digit=int(splitext(filename)[0][-4:])
            filename=splitext(filename)[0][:-4]+str(last_digit+1).zfill(4)+".hdf5"
        else:
            filename = splitext(filename)[0] +"_"+ str(0).zfill(4) + ".hdf5"
    f = h5py.File(filename, "w", libver='latest' )
    dset = f.create_dataset("image", dataset_shape, dtype="uint16")
    timestamps=f.create_dataset("timestamps",time_points, dtype="float")
    exposure=f.create_dataset("exposure",1, dtype="float")
    exposure[0]=live_parameters[0]
    frequency=f.create_dataset("frequency",1, dtype="float")
    frequency[0]=sync_parameters[0]
    y_positions = f.create_dataset("y_positions", y_size, dtype="float")
    if  scan_parameters[2]>= scan_parameters[1]:
        y_positions[:] = numpy.linspace(scan_parameters[1], scan_parameters[2], y_size)
    else:
        y_positions[:] = numpy.linspace(scan_parameters[2], scan_parameters[1], y_size)



    cam.set_one_parameter("trigger_source", int(live_parameters[4]))


    start_time=time.perf_counter()

    c=0

    if int(scan_parameters[0]) != 0:
        pitools.moveandwait(stage,stage.axes[0],(y_positions[0]-2*scan_parameters[3])/1000.0)
        stage_speed=scan_parameters[3]*frequency[0]/1000.0
        stage.CTO(4, 2, 1)
        stage.CTO(4, 3, 0)
        stage.CTO(4, 8, 0.0)
        stage.CTO(4, 9, 0.0)
        # stage.CTO(4, 8, y_positions[0]/1000.0)
        # stage.CTO(4, 9, y_positions[-1]/1000.0)
        stage.CTO(4, 1, scan_parameters[3]/1000.0)
        stage.TRO(4, 1)
        stage.VEL(stage.axes[0],stage_speed)
    f.swmr_mode = True


    cam.start_live()


    for t in range(time_points):
        while (time.perf_counter()-start_time)<t*timelapse_parameters[1]:
            if stop[0] == 1:
                break
            time.sleep(0.001)

            timestamps[t]=time.perf_counter()-start_time
        stage.MOV(stage.axes[0],(y_positions[-1]+2*scan_parameters[3])/1000.0)


        for y in range(y_size):
            image=cam.get_last_sequence_frame()
            dset[t,y,:,:]=image.reshape((dataset_shape[2],dataset_shape[3]))
            image16[:]=image.flatten()
            c+=1
            progress[0]=c/total_images
            newframe[0]=1
            if live_parameters[0]!=last_parameters[0]:
                cam.set_one_parameter("exposure_time", live_parameters[0])
                last_parameters[0]=live_parameters[0]
                exposure[0]=live_parameters[0]
            if live_parameters[1]!=last_parameters[1]:
                cam.set_one_parameter("internal_line_interval", live_parameters[1]*1e-06)
                last_parameters[1]=live_parameters[1]
            if live_parameters[4]!=last_parameters[4]:
                cam.set_one_parameter("trigger_source", int(live_parameters[4]))
                last_parameters[4]=live_parameters[4]
            if stop[0]==1:
                break
        f.flush()
        if stop[0] == 1:
            break
        if t!=time_points-1:
            stage.VEL(stage.axes[0], 250.0)
            pitools.moveandwait(stage,stage.axes[0],(y_positions[0]-2*scan_parameters[3])/1000.0)
            stage.VEL(stage.axes[0], stage_speed)

    stage.CloseConnection()

    progress[0] = 0
    cam.stop_live()
    cam.close()
    stop[0]=3


class CameraThread(QThread):
    newImage = pyqtSignal()

    def __init__(self, interface):
        super().__init__()
        self.newframe = numpy.frombuffer(interface.newframeRaw, dtype="uint8")

        self.stop = False

    def run(self):

        while not self.stop:
            if self.newframe[0]==1:
                self.newImage.emit()
                self.newframe[0] = 0
            else:
                time.sleep(0.005)


class SequenceThread(QThread):
    sequenceFinished = pyqtSignal()
    updateProgress =pyqtSignal()

    def __init__(self, interface):
        super().__init__()
        self.interface = interface
#
    def run(self):
        while self.interface.camerastop[0]!=3:
            self.updateProgress.emit()
            time.sleep(0.005)
        self.sequenceFinished.emit()


class ImageUpdateThread(QThread):
    changeImage = pyqtSignal(QImage)

    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.imagesize = self.interface.imagesize
        self.convertToQtFormat = QImage(numpy.zeros(self.imagesize, dtype="uint8").data,
                                        self.imagesize[0],
                                        self.imagesize[1],
                                        QImage.Format_Grayscale8)
        self.stop = False

    def run(self):
        while not self.stop:
            t = time.perf_counter()
            self.convertToQtFormat = QImage(self.interface.image8.data,
                                            self.imagesize[0],
                                            self.imagesize[1],
                                            QImage.Format_Grayscale8)
            p = self.convertToQtFormat.scaled(self.interface.image_label.size(),
                                              Qt.KeepAspectRatio)
            self.changeImage.emit(p)
            t = time.perf_counter() - t
            time.sleep(numpy.amax(numpy.asarray([0.016 - t, 0.001])))
            

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("interface.ui",self)

        # --------------------------------------------------------------------------
        # Setting up folders and widget
        self.imagesize = IMAGESIZE
        now = datetime.now()
        self.date_string = now.strftime("%Y_%m_%d")
        self.image_save_c = 0
        while isfile(temp_folder+self.date_string+"_"+str(self.image_save_c).zfill(4)+".hdf5"):
            self.image_save_c += 1

        self.lineEdit_savefile.setText(temp_folder+self.date_string+"_"+str(self.image_save_c).zfill(4)+".hdf5")
        self.pushButton_browse.clicked.connect(self.change_savefile_name)

        # --------------------------------------------------------------------------
        self.pushButton_current_pos_apply.clicked.connect(self.move_stage)
        self.pushButton_acquire_sequence.clicked.connect(self.start_acquire)
        self.sequence_thread = None
        # --------------------------------------------------------------------------
        #  variables needed for the camera
        self.parametersRaw= RawArray("c", 5*numpy.dtype("float").itemsize)
        self.camerastopRaw= RawArray("c", numpy.dtype("uint8").itemsize)
        self.newframeRaw= RawArray("c", numpy.dtype("uint8").itemsize)
        self.scanparametersRaw=RawArray("c", 4*numpy.dtype("float").itemsize)
        self.timelapseparametersRaw=RawArray("c", 3*numpy.dtype("float").itemsize)
        self.progressRaw=RawArray("c", numpy.dtype("float").itemsize)
        self.progress=numpy.frombuffer(self.progressRaw,dtype="float")
        self.scanparameters=numpy.frombuffer(self.scanparametersRaw, dtype="float")
        self.timelapseparameters=numpy.frombuffer(self.timelapseparametersRaw, dtype="float")
        self.parameters=numpy.frombuffer(self.parametersRaw, dtype="float")
        self.parameters[:]=[self.doubleSpinBox_exposure.value()*10**(-3),self.doubleSpinBox_interval.value(),IMAGESIZE[0],IMAGESIZE[1],1]
        self.camerastop = numpy.frombuffer(self.camerastopRaw, dtype="uint8")
        self.camerastop[0]=0
        self.newframe = numpy.frombuffer(self.newframeRaw, dtype="uint8")
        self.newframe[0]=0

        # --------------------------------------------------------------------------
        # variables needed for live autoscale and visualization
        self.stoppedRaw=RawArray("c",numpy.dtype("uint8").itemsize)
        self.minMaxRaw=RawArray("c",2*numpy.dtype("uint16").itemsize)
        self.lutRaw=RawArray("c",int(2**16)*numpy.dtype("uint8").itemsize)
        self.imageChangedRaw=RawArray("c",numpy.dtype("uint8").itemsize)
        self.autoRaw=RawArray("c",numpy.dtype("uint8").itemsize)
        self.minMaxChangedRaw=RawArray("c",numpy.dtype("uint8").itemsize)
        self.imageChanged=numpy.frombuffer(self.imageChangedRaw, dtype="uint8")
        self.auto=numpy.frombuffer(self.autoRaw, dtype="uint8")
        self.minMaxChanged=numpy.frombuffer(self.minMaxChangedRaw, dtype="uint8")
        numpy.copyto(self.imageChanged,numpy.asarray([1]).astype("uint8"))
        numpy.copyto(self.auto,numpy.asarray([1]).astype("uint8"))
        numpy.copyto(self.minMaxChanged,numpy.asarray([1]).astype("uint8"))
        self.minMax=numpy.frombuffer(self.minMaxRaw, dtype="uint16")
        numpy.copyto(self.minMax,numpy.asarray([0,2**16-1]).astype("uint16"))
        self.processes_stopped=numpy.frombuffer(self.stoppedRaw,dtype="uint8")
        self.processes_stopped[0]=0
        self.lut=numpy.frombuffer(self.lutRaw, dtype="uint8")
        self.darkvalue_verticalSlider.valueChanged.connect(self.scaleParameters)
        self.brightvalue_verticalSlider.valueChanged.connect(self.scaleParameters)
        self.autoscale_checkBox.clicked.connect(self.scaleParameters)


        self.image16Raw=None
        self.image8Raw=None
        self.image16=None
        self.image8=None
        self.minmaxp=None
        self.convert16to8p=None
        self.livecamp=None
        self.stage = None

        self.start_live_threads()

        self.iuth = ImageUpdateThread(self)
        self.iuth.changeImage.connect(self.setImage)
        self.iuth.start()
        self.cth = CameraThread(self)
        self.cth.newImage.connect(self.newImage)
        self.cth.start()

        # --------------------------------------------------------------------------
        # setting up the optotune lens

        self.max_current = 1.0
        self.max_amplitude = 1.0
        self.max_interval = 100.0
        self.max_delay = 100.0
        self.dm = AlpaoDM()




        self.DEV = "Dev1"
        self.NI_task = None
        self.switch_to_static()
        self.changefrequency()


        # ----------------------------------------------------------------------------------------------
        # Connecting buttons to functions for the camera controls

        self.pushButton_abort_sequence.clicked.connect(self.abort_sequence)
        self.pushButton_set_current_y_as_start.clicked.connect(self.set_current_y_as_start)
        self.pushButton_set_current_y_as_stop.clicked.connect(self.set_current_y_as_stop)
        # self.doubleSpinBox_exposure.setValue(self.camera.get_one_parameter("exposure_time") * 1000)
        # self.doubleSpinBox_interval.setValue(self.camera.get_one_parameter("internal_line_interval") * 1e6)

        self.doubleSpinBox_exposure.valueChanged.connect(self.change_camera_exposure)

        self.horizontalSlider_interval.valueChanged.connect(self.changeinterval_slider)
        self.doubleSpinBox_interval.valueChanged.connect(self.changeinterval_spinbox)

        self.spinBox_interval_digit.valueChanged.connect(self.change_interval_digit)

        # ----------------------------------------------------------------------------------------------
        # Connecting buttons to functions for the lens controls

        self.horizontalSlider_current.valueChanged.connect(self.changecurrent_slider)
        self.doubleSpinBox_current.valueChanged.connect(self.changecurrent_spinbox)

        self.horizontalSlider_amplitude.valueChanged.connect(self.changeamplitude_slider)
        self.doubleSpinBox_amplitude.valueChanged.connect(self.changeamplitude_spinbox)

        self.horizontalSlider_offset.valueChanged.connect(self.changeoffset_slider)
        self.doubleSpinBox_offset.valueChanged.connect(self.changeoffset_spinbox)
        self.spinBox_offset_digit.valueChanged.connect(self.change_offset_digit)

        self.horizontalSlider_phase.valueChanged.connect(self.changephase_slider)
        self.doubleSpinBox_phase.valueChanged.connect(self.changephase_spinbox)
        self.spinBox_phase_digit.valueChanged.connect(self.change_phase_digit)

        self.pushButton_frequency_apply.clicked.connect(self.changefrequency)

        self.radioButton_static.pressed.connect(self.switch_to_static)
        self.radioButton_sawtooth.pressed.connect(self.switch_to_sawtooth)

        self.horizontalSlider_interval.valueChanged.connect(self.changeinterval_slider)
        self.doubleSpinBox_interval.valueChanged.connect(self.changeinterval_spinbox)

        self.pushButton_preset_apply.clicked.connect(self.apply_preset)

        self.pushButton_apply_sensorsize.clicked.connect(self.change_sensor_size)
        self.pushButton_preset_save.clicked.connect(self.save_preset)

    def set_trigger_tasks(self):
        try:
            t = time.perf_counter()
            if self.NI_task is not None:
                self.NI_task.StopTask()
            freq=self.doubleSpinBox_frequency.value()
            ni_freq=freq*2.0
            ni_duty_cycle=1.0-(self.doubleSpinBox_phase.value()*0.001*ni_freq)
            if ni_duty_cycle>0.99999:
                ni_duty_cycle = 0.99999
            if ni_duty_cycle<0.000002:
                ni_duty_cycle = 0.000002

            self.NI_task=Task()
            self.NI_task.CreateCOPulseChanFreq("Dev1/ctr0", None, DAQmx_Val_Hz,
                                               DAQmx_Val_Low,
                                               0.0,
                                               ni_freq,
                                               ni_duty_cycle)
            self.NI_task.CfgImplicitTiming(DAQmx_Val_FiniteSamps,1)
            self.NI_task.CfgDigEdgeStartTrig("pfi3",DAQmx_Val_Rising)
            self.NI_task.SetStartTrigRetriggerable(True)
            self.NI_task.StartTask()
        except:
            print("bad trigger values!")

    def smooth_sawtooth(self):
        freq = self.doubleSpinBox_frequency.value()
        n_samples=int((1/freq)/MIRROR_SAMPLE_DURATION)
        cutoff_freq=self.doubleSpinBox_cutoff_freq.value()

        defocus_sequence = numpy.concatenate((numpy.linspace(-1.0, 1.0, int(n_samples*0.95)),
                                              numpy.linspace(1.0, -1.0, n_samples-int(n_samples*0.95))))
        # defocus_sequence = numpy.linspace(-1.0, 1.0, n_samples)
        b, a = signal.butter(3, cutoff_freq, 'low', fs=1 / MIRROR_SAMPLE_DURATION)
        defocus_sequence_filtered = signal.lfilter(b, a, numpy.tile(defocus_sequence,10))
        defocus_sequence_filtered=defocus_sequence_filtered/numpy.amax(defocus_sequence_filtered)
        defocus_sequence_filtered=defocus_sequence_filtered[n_samples:-n_samples]
        max_index=numpy.argmax(defocus_sequence_filtered[:3*n_samples])
        defocus_sequence_filtered=defocus_sequence_filtered[max_index:max_index+n_samples]
        # if SCAN_DIRECTION == 1:
        #     amplitudes_1 = numpy.linspace(-1.0, 1.0, nsamples - int(nsamples *(self.doubleSpinBox_ret_time.value()/100.0)))
        #     amplitudes_2 = numpy.linspace(1.0, -1.0, int(nsamples *(self.doubleSpinBox_ret_time.value()/100.0)))
        # else:
        #     amplitudes_1 = numpy.linspace(1.0, -1.0, nsamples - int(nsamples *(self.doubleSpinBox_ret_time.value()/100.0)))
        #     amplitudes_2 = numpy.linspace(-1.0, 1.0, int(nsamples *(self.doubleSpinBox_ret_time.value()/100.0)))
        #
        #
        # amplitudes = numpy.concatenate((amplitudes_2, amplitudes_1))
        # window_half = int(nsamples *(self.doubleSpinBox_ret_time.value()/200.0))
        # amplitudes_padded = numpy.zeros(amplitudes.shape[0] + 2 * window_half)
        # amplitudes_padded[:window_half] = amplitudes[-window_half:]
        # amplitudes_padded[-window_half:] = amplitudes[:window_half]
        # amplitudes_padded[window_half:-window_half] = amplitudes
        #
        # amplitudes = numpy.correlate(amplitudes_padded, numpy.ones(2 * window_half) / (2 * window_half), mode="same")[
        #              window_half:-window_half]
        # amplitudes=amplitudes/numpy.amax(numpy.abs(amplitudes))
        # amplitudes = numpy.roll(amplitudes, -numpy.argmax(amplitudes))
        return self.doubleSpinBox_amplitude.value()*defocus_sequence_filtered+self.doubleSpinBox_offset.value()

    def start_mirror_defocus_sequence(self):
        freq = self.doubleSpinBox_frequency.value()
        n_samples=int((1/freq)/MIRROR_SAMPLE_DURATION)

        zernikes = numpy.zeros((n_samples,27))
        zernikes[:,3] = self.smooth_sawtooth()
        if self.aberration_mode_comboBox.currentText()=="Spherical compensation":
            if self.objective_comboBox.currentText()=="20X":
                exps=numpy.load("polyfit_correction_20X.npy")
                for i in range(exps.shape[0]):
                    zernikes[:,int(exps[i,0])]=numpy.polyval(exps[i,1:],zernikes[:,3])
            elif self.objective_comboBox.currentText()=="10X":
                exps=numpy.load("polyfit_correction_10X.npy")
                for i in range(exps.shape[0]):
                    zernikes[:,int(exps[i,0])]=numpy.polyval(exps[i,1:],zernikes[:,3])

        self.dm.stop_loop()
        self.dm.send_pattern_zernikes(zernikes, 0)

    def set_mirror_defocus_sequence(self):
        freq=self.doubleSpinBox_frequency.value()
        n_samples=int((1/freq)/MIRROR_SAMPLE_DURATION)
        print(n_samples)

        zernikes = numpy.zeros((n_samples,27))
        zernikes[:,3] = self.smooth_sawtooth()

        self.dm.update_pattern_zernikes(zernikes)

    def start_mirror_defocus_static(self):
        zernikes = numpy.zeros(27)

        zernikes[3] = self.doubleSpinBox_current.value() + self.doubleSpinBox_offset.value()
        self.dm.stop_loop()

        self.dm.start_direct_control()
        self.dm.send_direct_zernikes(zernikes)

    def set_mirror_defocus_static(self):
        zernikes = numpy.zeros(27)

        zernikes[3] = self.doubleSpinBox_current.value() + self.doubleSpinBox_offset.value()

        self.dm.send_direct_zernikes(zernikes)

    def set_current_y_as_start(self):
        current_position = self.doubleSpinBox_galvo_voltage.value()
        self.doubleSpinBox_y_start_pos.setValue(current_position)

    def set_current_y_as_stop(self):
        current_position = self.doubleSpinBox_galvo_voltage.value()
        self.doubleSpinBox_y_stop_pos.setValue(current_position)

    def reset_threads_to_live(self):
        self.start_mirror_defocus_sequence()
        self.processes_stopped[0]=1
        while self.processes_stopped[0]!=3:
            time.sleep(0.001)
        self.processes_stopped[0] = 0
        self.camerastop[0] = 0
        self.changefrequency()
        self.start_live_threads()

    def start_live_threads(self):
        self.image16Raw=RawArray("c",self.imagesize[0]*self.imagesize[1]*numpy.dtype("uint16").itemsize)
        self.image8Raw=RawArray("c",self.imagesize[0]*self.imagesize[1]*numpy.dtype("uint8").itemsize)
        self.image16=numpy.frombuffer(self.image16Raw, dtype="uint16").reshape(self.imagesize)
        self.image8=numpy.frombuffer(self.image8Raw, dtype="uint8").reshape(self.imagesize)
        self.minmaxp=Process(target=findminmax,args=(self.image16Raw,self.minMaxRaw, self.lutRaw, self.minMaxChangedRaw, self.autoRaw, self.stoppedRaw))
        self.convert16to8p=Process(target=convert16to8,args=(self.image16Raw, self.lutRaw, self.image8Raw,self.imageChangedRaw,self.stoppedRaw))
        self.livecamp=Process(target=live_camera_thread,args=(self.image16Raw, self.parametersRaw, self.camerastopRaw, self.newframeRaw))
        self.livecamp.daemon=True
        self.minmaxp.daemon=True
        self.convert16to8p.daemon=True
        self.stage=GCSDevice('C-413.2GA')
        self.stage.ConnectUSB(serialnum="119024343")
        pitools.startup(self.stage, stages=["V-524.1AA",], refmodes=["FRF", ])
        self.stage.VEL(self.stage.axes[0], 250.0)
        self.doubleSpinBox_current_pos.setValue(self.stage.qPOS()["1"]*1000)
        self.livecamp.start()
        self.minmaxp.start()
        self.convert16to8p.start()

    def start_sequence_threads(self):
        self.stage.CloseConnection()

        self.image16Raw=RawArray("c",self.imagesize[0]*self.imagesize[1]*numpy.dtype("uint16").itemsize)
        self.image8Raw=RawArray("c",self.imagesize[0]*self.imagesize[1]*numpy.dtype("uint8").itemsize)
        self.image16=numpy.frombuffer(self.image16Raw, dtype="uint16").reshape(self.imagesize)
        self.image8=numpy.frombuffer(self.image8Raw, dtype="uint8").reshape(self.imagesize)
        self.minmaxp=Process(target=findminmax,args=(self.image16Raw,self.minMaxRaw, self.lutRaw, self.minMaxChangedRaw, self.autoRaw, self.stoppedRaw))
        self.convert16to8p=Process(target=convert16to8,args=(self.image16Raw, self.lutRaw, self.image8Raw,self.imageChangedRaw,self.stoppedRaw))

        if self.scan_y_checkBox.isChecked():
            self.scanparameters[:] = [1,
                                      self.doubleSpinBox_y_start_pos.value(),
                                      self.doubleSpinBox_y_stop_pos.value(),
                                      self.doubleSpinBox_y_step_size.value()]
        else:
            self.scanparameters[:]= [0, 0, 0, 0]

        if self.scan_t_checkBox.isChecked():
            self.timelapseparameters[:]=[1.0,self.scan_t_interval_doubleSpinBox.value(),self.scan_t_timepoints_spinBox.value()]
        else:
            self.timelapseparameters[:]=[0,0,0]

        sync_parameters=numpy.asarray([self.doubleSpinBox_frequency.value(),self.doubleSpinBox_phase.value()])
        self.livecamp=Process(target=sequence_camera_thread, args=(self.lineEdit_savefile.text(),
                                                                   sync_parameters,
                                                                   self.image16Raw,
                                                                   self.parametersRaw,
                                                                   self.camerastopRaw,
                                                                   self.newframeRaw,
                                                                   self.scanparametersRaw,
                                                                   self.timelapseparametersRaw,
                                                                   self.progressRaw))
        self.livecamp.daemon=True
        self.minmaxp.daemon=True
        self.convert16to8p.daemon=True
        self.livecamp.start()
        self.minmaxp.start()
        self.convert16to8p.start()
        self.sequence_thread = SequenceThread(self)
        self.sequence_thread.sequenceFinished.connect(self.reset_threads_to_live)
        self.sequence_thread.updateProgress.connect(self.update_progress_bar)
        self.sequence_thread.start()

    def update_progress_bar(self):
        self.progressBar.setValue(int(self.progress[0] * 1000))

    def stop_autoscale_threads(self):
        self.processes_stopped[0]=1
        self.camerastop[0]=1
        while self.processes_stopped[0]!=3 or self.camerastop[0]!=3:
            time.sleep(0.001)
        self.processes_stopped[0] = 0
        self.camerastop[0] = 0

    def abort_sequence(self):
        self.camerastop[0]=1

    def change_sensor_size(self):
        self.stage.CloseConnection()
        self.cth.stop=True
        self.iuth.stop=True
        self.stop_autoscale_threads()
        while self.cth.isRunning() or self.iuth.isRunning():
            time.sleep(0.001)
        if self.comboBox_h_sensorsize.currentText()=="Full width":
            self.imagesize=(2304,int(self.comboBox_v_sensorsize.currentText()))
        elif self.comboBox_h_sensorsize.currentText()=="Square":
            self.imagesize=(int(self.comboBox_v_sensorsize.currentText()),int(self.comboBox_v_sensorsize.currentText()))
        self.parameters[2]=self.imagesize[0]
        self.parameters[3] = self.imagesize[1]
        self.changeinterval_spinbox()
        self.change_camera_exposure()

        self.start_live_threads()
        self.iuth = ImageUpdateThread(self)
        self.iuth.changeImage.connect(self.setImage)
        self.iuth.start()
        self.cth=CameraThread(self)
        self.cth.newImage.connect(self.newImage)
        self.cth.start()
        if self.radioButton_static.isChecked():
            self.switch_to_static()
        else:
            self.switch_to_sawtooth()

    def move_stage(self):
        self.stage.MOV(self.stage.axes[0], self.doubleSpinBox_current_pos.value()/1000.0)

    def start_acquire(self):
        if not self.radioButton_sawtooth.isChecked():
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Sawtooth is off!")
            msg.exec_()
        else:
            self.stop_autoscale_threads()
            self.start_sequence_threads()

    def change_camera_exposure(self):
        self.parameters[0]=self.doubleSpinBox_exposure.value()*10**(-3)

    def switch_to_static(self):
        self.parameters[4]=1
        self.start_mirror_defocus_static()

    def switch_to_sawtooth(self):
        self.parameters[4] = 2
        self.start_mirror_defocus_sequence()

    def changeinterval_slider(self):
        current_value = self.doubleSpinBox_interval.value()
        minimum = numpy.floor(current_value/10**-self.spinBox_interval_digit.value())*10**-self.spinBox_interval_digit.value()
        value = (minimum+self.horizontalSlider_interval.value()/1000*10**-self.spinBox_interval_digit.value())
        if value > minimum+0.99*10**-self.spinBox_interval_digit.value():
            value=minimum+0.99*10**-self.spinBox_interval_digit.value()
        self.parameters[1] = value
        self.doubleSpinBox_interval.blockSignals(True)
        self.doubleSpinBox_interval.setValue(minimum+self.horizontalSlider_interval.value()/1000*10**-self.spinBox_interval_digit.value())
        self.doubleSpinBox_interval.blockSignals(False)

    def changeinterval_spinbox(self):
        self.parameters[1]=self.doubleSpinBox_interval.value()
        current_value = self.doubleSpinBox_interval.value()
        minimum = numpy.floor(current_value/10**-self.spinBox_interval_digit.value())*10**-self.spinBox_interval_digit.value()
        self.horizontalSlider_interval.blockSignals(True)
        self.horizontalSlider_interval.setValue(int((current_value-minimum)/10**-self.spinBox_interval_digit.value()*1000))
        self.horizontalSlider_interval.blockSignals(False)

    def change_interval_digit(self):
        self.doubleSpinBox_interval.setSingleStep(10**(-self.spinBox_interval_digit.value()))

    def change_offset_digit(self):
        self.doubleSpinBox_offset.setSingleStep(10**(-self.spinBox_offset_digit.value()-2))

    def changecurrent_slider(self):
        if self.radioButton_static.isChecked():
            self.set_mirror_defocus_static()
        self.doubleSpinBox_current.blockSignals(True)
        self.doubleSpinBox_current.setValue(self.horizontalSlider_current.value()/1000*self.max_current)
        self.doubleSpinBox_current.blockSignals(False)

    def changecurrent_spinbox(self):
        if self.radioButton_static.isChecked():
            self.set_mirror_defocus_static()
        self.horizontalSlider_current.blockSignals(True)
        self.horizontalSlider_current.setValue(int(self.doubleSpinBox_current.value()*1000/self.max_current))
        self.horizontalSlider_current.blockSignals(False)

    def changeamplitude_slider(self):
        if self.radioButton_sawtooth.isChecked():
            self.set_mirror_defocus_sequence()
        self.doubleSpinBox_amplitude.blockSignals(True)
        self.doubleSpinBox_amplitude.setValue(self.horizontalSlider_amplitude.value() / 1000 * self.max_amplitude)
        self.doubleSpinBox_amplitude.blockSignals(False)

    def changeamplitude_spinbox(self):
        if self.radioButton_sawtooth.isChecked():
            self.set_mirror_defocus_sequence()
        self.horizontalSlider_amplitude.blockSignals(True)
        self.horizontalSlider_amplitude.setValue(int(self.doubleSpinBox_amplitude.value()*1000/self.max_amplitude))
        self.horizontalSlider_amplitude.blockSignals(False)

    def changeoffset_slider(self):
        if self.radioButton_sawtooth.isChecked():
            self.set_mirror_defocus_sequence()
        self.doubleSpinBox_offset.blockSignals(True)
        self.doubleSpinBox_offset.setValue(self.horizontalSlider_offset.value()/1000*self.max_current)
        self.doubleSpinBox_offset.blockSignals(False)

    def changeoffset_spinbox(self):
        if self.radioButton_sawtooth.isChecked():
            self.set_mirror_defocus_sequence()
        self.horizontalSlider_offset.blockSignals(True)
        self.horizontalSlider_offset.setValue(int(self.doubleSpinBox_offset.value()*1000/self.max_current))
        self.horizontalSlider_offset.blockSignals(False)

    def changephase_slider(self):
        current_value=self.doubleSpinBox_phase.value()
        minimum = numpy.floor(
            current_value / 10 ** -self.spinBox_phase_digit.value()) * 10 ** -self.spinBox_phase_digit.value()
        value = (minimum + self.horizontalSlider_phase.value() / 1000 * 10 ** -self.spinBox_phase_digit.value())
        if value > minimum + 0.99 * 10 ** -self.spinBox_phase_digit.value():
            value = minimum + 0.99 * 10 ** -self.spinBox_phase_digit.value()
        self.set_trigger_tasks()
        self.doubleSpinBox_phase.blockSignals(True)
        self.doubleSpinBox_phase.setValue(
            minimum + self.horizontalSlider_phase.value() / 1000 * 10 ** -self.spinBox_phase_digit.value())
        self.doubleSpinBox_phase.blockSignals(False)


        # self.doubleSpinBox_phase.blockSignals(True)
        # self.doubleSpinBox_phase.setValue(float(self.horizontalSlider_phase.value())*1.0/1000)
        # self.doubleSpinBox_phase.blockSignals(False)
        # self.sig_gen.SetPhase(self.doubleSpinBox_phase.value()*360.0)

    def changephase_spinbox(self):
        self.set_trigger_tasks()
        current_value=self.doubleSpinBox_phase.value()
        minimum = numpy.floor(current_value / 10 ** -self.spinBox_phase_digit.value()) \
                  * 10 ** -self.spinBox_phase_digit.value()
        self.horizontalSlider_phase.blockSignals(True)
        self.horizontalSlider_phase.setValue(int((current_value-minimum)/10**-self.spinBox_phase_digit.value()*1000))
        self.horizontalSlider_phase.blockSignals(False)

    def change_phase_digit(self):
        self.doubleSpinBox_phase.setSingleStep(10**(-self.spinBox_phase_digit.value()))

    def apply_preset(self):
        new_frequency=[int(s) for s in self.comboBox_preset_selection.currentText().split() if s.isdigit()][0]
        if self.objective_comboBox.currentText()=="20X":
            parameters=get_parameters("settings_20X.json",new_frequency,int(self.comboBox_v_sensorsize.currentText()))
        if self.objective_comboBox.currentText()=="10X":
            parameters=get_parameters("settings_10X.json",new_frequency,int(self.comboBox_v_sensorsize.currentText()))
        if parameters is not None:
            # self.doubleSpinBox_ret_time.setValue(parameters["Return time"])
            self.doubleSpinBox_frequency.setValue(new_frequency)
            self.changefrequency()
            self.doubleSpinBox_exposure.setValue(parameters["Exposure"])
            self.doubleSpinBox_interval.setValue(parameters["Line interval"])
            self.doubleSpinBox_amplitude.setValue(parameters["Amplitude"])
            self.doubleSpinBox_phase.setValue(parameters["Phase"])
            self.changefrequency()

    def save_preset(self):
        if self.objective_comboBox.currentText()=="20X":
            with open("settings_20X.json", "r") as infile:
                data = json.load(infile)
        if self.objective_comboBox.currentText()=="10X":
            with open("settings_10X.json", "r") as infile:
                data = json.load(infile)
        frequency=self.doubleSpinBox_frequency.value()
        vsize=self.comboBox_v_sensorsize.currentText()

        data["vsize_"+str(int(vsize)).zfill(3)]["frequency_"+str(int(frequency))] = {"Line interval": self.doubleSpinBox_interval.value(),
                                                                              "Exposure": self.doubleSpinBox_exposure.value(),
                                                                              "Amplitude": self.doubleSpinBox_amplitude.value(),
                                                                              "Phase": self.doubleSpinBox_phase.value(),
                                                                              "Return time": 10}
        # "Return time": self.doubleSpinBox_ret_time.value()
        if self.objective_comboBox.currentText()=="20X":
            with open("settings_20X.json", "w") as outfile:
                outfile.write(json.dumps(data, indent=4, sort_keys=True))
        if self.objective_comboBox.currentText()=="10X":
            with open("settings_10X.json", "w") as outfile:
                outfile.write(json.dumps(data, indent=4, sort_keys=True))



    def changefrequency(self):
        self.dm.stop_loop()
        self.set_trigger_tasks()

        if self.radioButton_sawtooth.isChecked():
            self.start_mirror_defocus_sequence()
        else:
            self.start_mirror_defocus_static()

    def scaleParameters(self):
        if self.autoscale_checkBox.isChecked():
            numpy.copyto(self.auto,numpy.asarray([1]).astype("uint8"))
        else:
            numpy.copyto(self.auto,numpy.asarray([0]).astype("uint8"))
            minval=self.darkvalue_verticalSlider.value()/1024*2**16
            maxval=self.brightvalue_verticalSlider.value()/1024*2**16-1
            numpy.copyto(self.minMax,numpy.asarray([minval,maxval]).astype("uint16"))
        numpy.copyto(self.minMaxChanged,numpy.asarray([1]).astype("uint8"))

    def newImage(self):
        numpy.copyto(self.imageChanged,numpy.asarray([1]).astype("uint8"))
        if self.autoscale_checkBox.isChecked():
            numpy.copyto(self.minMaxChanged,numpy.asarray([1]).astype("uint8"))

    def change_savefile_name(self):
        name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "Tif images (*.hdf5)")
        if name:
            self.lineEdit_savefile.setText(name)
        else:
            self.lineEdit_savefile.setText(
                temp_folder + self.date_string + "_" + str(self.image_save_c).zfill(4) + ".hdf5")
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self,event):
        self.iuth.stop=True
        self.cth.stop=True
        self.stage.CloseConnection()
        self.stop_autoscale_threads()
        self.NI_task.ClearTask()
        self.dm.stop_loop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
