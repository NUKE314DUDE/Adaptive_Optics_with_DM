from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import warnings
import numpy


class camera:
    def __init__(self, cam_num=0):
        ids_peak.Library.Initialize()
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()
        self.device = device_manager.Devices()[cam_num].OpenDevice(ids_peak.DeviceAccessType_Control)


        # Nodemap for accessing GenICam nodes
        self.remote_nodemap = self.device.RemoteDevice().NodeMaps()[0]
        self.data_stream = self.device.DataStreams()[0].OpenDataStream()

    def get_size(self):
        return(self.remote_nodemap.FindNode("SensorWidth").Value(),self.remote_nodemap.FindNode("SensorHeight").Value())

    def set_node_value(self,name,value):
        if value<self.remote_nodemap.FindNode(name).Minimum():
            self.remote_nodemap.FindNode(name).SetValue(
                self.remote_nodemap.FindNode(name).Minimum())
        elif value>self.remote_nodemap.FindNode(name).Maximum():
            self.remote_nodemap.FindNode(name).SetValue(
                self.remote_nodemap.FindNode(name).Maximum())
        else:
            self.remote_nodemap.FindNode(name).SetValue(value)

    def set_full_chip(self):
        self.remote_nodemap.FindNode("OffsetX").SetValue(0)
        self.remote_nodemap.FindNode("OffsetY").SetValue(0)
        self.remote_nodemap.FindNode("Width").SetValue(
            self.remote_nodemap.FindNode("Width").Maximum())
        self.remote_nodemap.FindNode("Height").SetValue(
            self.remote_nodemap.FindNode("Height").Maximum())

    def set_active_region(self,x,y,w,h):
        self.remote_nodemap.FindNode("OffsetX").SetValue(0)
        self.remote_nodemap.FindNode("OffsetY").SetValue(0)

        self.set_node_value("Width",w)
        self.set_node_value("Height", h)
        self.set_node_value("OffsetX",x)
        self.set_node_value("OffsetY", y)

    def set_exposure_ms(self,value,framerate=None):
        value=value*1000
        self.set_node_value("ExposureTime",value)

        if framerate is None:
            self.remote_nodemap.FindNode("AcquisitionFrameRate").SetValue(
                self.remote_nodemap.FindNode("AcquisitionFrameRate").Maximum())
        else:
            self.set_node_value("AcquisitionFrameRate",framerate)

    def set_gain(self,value):
        self.set_node_value("Gain",value)

    def set_bit_depth(self,value):
        if value==8:
            self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry(self.remote_nodemap.FindNode("PixelFormat").Entries()[0])
        elif value==10:
            self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry(self.remote_nodemap.FindNode("PixelFormat").Entries()[3])
        elif value==12:
            self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry(self.remote_nodemap.FindNode("PixelFormat").Entries()[4])
        else:
            warnings.warn("Invalid bit depth")

    def start_acquisition(self, buffersize=0):
        payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
        if buffersize<self.data_stream.NumBuffersAnnouncedMinRequired():
            buffer_count_max = self.data_stream.NumBuffersAnnouncedMinRequired()
        else:
            buffer_count_max = buffersize

        # Allocate buffers and add them to the pool
        for buffer_count in range(buffer_count_max):
            buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
            self.data_stream.QueueBuffer(buffer)

        self.data_stream.StartAcquisition()
        self.remote_nodemap.FindNode("AcquisitionStart").Execute()
        self.remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()

    def stop_acquisition(self):
        self.remote_nodemap.FindNode("AcquisitionStop").Execute()
        self.remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()

        self.data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for buffer in self.data_stream.AnnouncedBuffers():
            self.data_stream.RevokeBuffer(buffer)

    def get_frame(self, timeout_ms=1000):
        buffer = self.data_stream.WaitForFinishedBuffer(timeout_ms)
        img = numpy.copy(ids_peak_ipl_extension.BufferToImage(buffer).get_numpy())
        self.data_stream.QueueBuffer(buffer)

        return img


    def close(self):
        ids_peak.Library.Close()

if __name__=="__main__":
    import time
    cam=camera()
    cam.set_bit_depth(8)
    cam.set_full_chip()
    # cam.set_active_region(300,900,300,300)

    cam.set_exposure_ms(1.0)
    cam.set_gain(10.0)


    cam.start_acquisition()
    t = time.perf_counter()
    for i in range(100):
        img=cam.get_frame()
        print(img.shape,time.perf_counter()-t)
        t = time.perf_counter()
    cam.stop_acquisition()

    cam.close()
