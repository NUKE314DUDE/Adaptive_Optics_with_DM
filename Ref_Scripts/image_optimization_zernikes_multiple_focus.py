from Vipin_hamamatsu_library import camera
import alpao_control
import numpy
import time
import tifffile
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cosine
from scipy.signal import correlate2d


def image_metric(image):
    mean=numpy.mean(image)
    var=numpy.var(image)
    return var/mean


if __name__ == "__main__":
    dm = alpao_control.AlpaoDM()
    dm.start_direct_control()
    dm.send_direct_voltages(numpy.zeros(57))

    cam = camera()
    cam.camera_open()
    cam.set_one_parameter("subarray_mode",2)
    cam.set_one_parameter("subarray_hsize",1024)
    cam.set_one_parameter("subarray_vsize",1024)
    cam.set_one_parameter("subarray_hpos",int(2304/2-512))
    cam.set_one_parameter("subarray_vpos",int(2304/2-512))
    cam.set_one_parameter("sensor_mode", 12.0)
    cam.set_one_parameter("exposure_time", 0.005)

    cam.start_live()
    print("camera opened")

    image=cam.get_last_live_frame()
    plt.ion()

    num=15

    def_values=numpy.linspace(-0.8,0.8,11)
    order=numpy.argsort(numpy.abs(def_values))
    def_values=def_values[order]
    for  i in range(int(11/2)):
        def_values[1+2*i:1+2*i+2]=numpy.sort(def_values[1+2*i:1+2*i+2])


    for d,defocus in enumerate(def_values):
        amp_range = 0.5
        if d==0:
            good_zerns = numpy.zeros(20)
        elif d==1:
            center_zerns=numpy.copy(good_zerns)
        else:
            good_zerns=center_zerns#-(good_zerns-center_zerns)
        good_zerns[3] = defocus
        dm.send_direct_zernikes(good_zerns)
        if d==0:
            n_iter=2
        else:
            n_iter=2

        fig, ax = plt.subplots()
        fig_num=fig.number
        axim = ax.imshow(image)

        while 1:
            if plt.fignum_exists(fig_num):
                image = cam.get_last_live_frame()
                print(image_metric(image),numpy.amax(image))
                axim.set_data(image)
                fig.canvas.flush_events()
            else:
                break

        image = cam.get_last_live_frame()
        tifffile.imwrite("uncorrected_def_"+str(numpy.round(defocus,2)).zfill(4)+".tif",image)

        for j in range(n_iter):
            for z in range(2,20):
                metrics=[]
                if z!=3:
                    for i,amp in enumerate(numpy.linspace(-amp_range/(2**j),amp_range/(2**j),num)):
                        zerns = numpy.copy(good_zerns)
                        zerns[z]+=amp
                        dm.send_direct_zernikes(zerns)
                        time.sleep(0.1)
                        image = cam.get_last_live_frame()
                        # if amp==0:
                        #     plt.imshow(image)
                        #     plt.show()
                        metrics.append(image_metric(image))
                        if i == int(num / 2):
                            print(d, z, image_metric(image))



                    spl=CubicSpline(numpy.linspace(-amp_range/(2**j),amp_range/(2**j),num),savgol_filter(metrics,7,5))
                    interp_curve=spl(numpy.linspace(-amp_range/(2**j),amp_range/(2**j),100*num))
                    good_zerns[z] +=numpy.linspace(-amp_range/(2**j),amp_range/(2**j),100*num)[numpy.argmax(interp_curve)]



        dm.send_direct_zernikes(good_zerns)
        time.sleep(0.1)
        image = cam.get_last_live_frame()
        tifffile.imwrite("corrected_def_"+str(numpy.round(defocus,2)).zfill(4)+".tif",image)

        print(numpy.dot(dm.zer_mat, zerns)+dm.flat_voltages)

        numpy.save("opt_zernikes_"+str(defocus)+".npy", good_zerns)
        plt.imshow(image)
        plt.show()
    dm.stop_loop()

