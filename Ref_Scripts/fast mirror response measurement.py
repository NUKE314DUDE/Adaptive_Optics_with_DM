from alpao_control import AlpaoDM
from ids import camera as idscamera
from zernike_generator import zernike
import numpy
import time
import matplotlib.pyplot as plt
import tifffile

PITCH = 31.25
THRESHOLD_1=15
THRESHOLD_2=7
POKE=0.1
CENT_THRESHOLD=0.5
MIRROR_SAMPLE_DURATION=6.67e-6

def relax_mirror(dm):
    t_start=time.perf_counter()
    while time.perf_counter()-t_start<0.5:
        amp=numpy.exp(-(time.perf_counter()-t_start)/0.2)*numpy.cos((time.perf_counter()-t_start)/0.033*2*numpy.pi)
        dm.send_direct_voltages(numpy.ones(69)*amp)


def find_centroid(image,x,y):
    half_size=int(PITCH/2)

    square=(image[int(y)-half_size:int(y)+half_size,int(x)-half_size:int(x)+half_size]).astype(float)
    square=square-THRESHOLD_2
    square[square<0]=0
    x_c=numpy.arange(-int(PITCH/2),+int(PITCH/2))
    x_c,y_c=numpy.meshgrid(x_c,x_c)

    cent_x = numpy.sum(x_c * square) / numpy.sum(square)
    cent_y = numpy.sum(y_c * square) / numpy.sum(square)

    return cent_y,cent_x


def find_centroids(image,centers_x,centers_y):
    centroids_x = []
    centroids_y = []

    for i in range(centers_x.shape[0]):
        cy, cx = find_centroid(image, centers_x[i], centers_y[i])
        centroids_x.append(cx + centers_x[i])
        centroids_y.append(cy + centers_y[i])

    return numpy.asarray(centroids_x+centroids_y)

if __name__=="__main__":
    dm = AlpaoDM()
    dm.start_direct_control()
    relax_mirror(dm)


    time.sleep(0.1)

    camera=idscamera()
    camera.set_exposure_ms(0.04)
    camera.set_bit_depth(8)
    camera.set_full_chip()
    # cam.set_active_region(300,900,300,300)
    camera.set_gain(1.0)
    camera.start_acquisition(4)

    frame=None
    while frame is None:
        frame=camera.get_frame()
        frame = camera.get_frame()
    reference_frame = frame.astype(float)

    plt.imshow(reference_frame)
    plt.show()

    for i in range(4):
        frame = camera.get_frame()
        frame = camera.get_frame()
        reference_frame += numpy.copy(frame)
    reference_frame=reference_frame/5

    image=reference_frame.astype(float)
    image=image-THRESHOLD_1
    image[image<0]=0

    spots_x=[]
    spots_y=[]

    while numpy.any(image>0):
        spot = numpy.unravel_index(numpy.argmax(image),image.shape)
        spots_y.append(spot[0])
        spots_x.append(spot[1])
        image[spot[0]-int(1.3*PITCH/2):spot[0]+int(1.3*PITCH/2),spot[1]-int(1.3*PITCH/2):spot[1]+int(1.3*PITCH/2)] = 0
        if image[spot]!=0:
            print(spot[0]-int(1.3*PITCH/2),spot[0]+int(1.3*PITCH/2),spot[1]-int(1.3*PITCH/2),spot[1]+int(1.3*PITCH/2))
            plt.imshow(image)
            plt.show()

    cent_num=len(spots_x)
    spots_x = numpy.asarray(spots_x)
    spots_y = numpy.asarray(spots_y)
    center_x = numpy.mean(spots_x)
    center_y = numpy.mean(spots_y)
    radius=numpy.amax(numpy.sqrt((spots_x-center_x)**2+(spots_y-center_y)**2))+PITCH/2

    camera_coords_x,camera_coords_y=numpy.meshgrid(numpy.arange(image.shape[1]),numpy.arange(image.shape[0]))
    camera_coords_x=(camera_coords_x-center_x)/radius
    camera_coords_y=(camera_coords_y-center_y)/radius


    circ_points=1000
    angle=numpy.linspace(0,2*numpy.pi,circ_points)

    circle_x=center_x+radius*numpy.cos(angle)
    circle_y=center_y+radius*numpy.sin(angle)

    plt.imshow(numpy.sqrt(camera_coords_x**2+camera_coords_y**2))
    plt.plot(circle_x,circle_y)
    plt.show()

    plt.imshow(reference_frame)
    plt.scatter(spots_x,spots_y)
    plt.scatter(center_x,center_y)
    plt.plot(circle_x,circle_y)
    plt.show()

    reference_centroids=find_centroids(reference_frame,spots_x,spots_y)

    zer_gen=zernike(6)
    zer_to_cent_matrix=numpy.zeros((reference_centroids.shape[0],zer_gen.zernike_num-1))

    zernike_gradients_x=numpy.zeros((zer_gen.zernike_num-1,reference_frame.shape[0],reference_frame.shape[1]))
    zernike_gradients_y=numpy.zeros((zer_gen.zernike_num-1,reference_frame.shape[0],reference_frame.shape[1]))


    for z in range(zer_gen.zernike_num - 1):
        print(z+1, "/", zer_gen.zernike_num - 1)
        y_exp, x_exp = numpy.nonzero(zer_gen.zernike_matrices[z + 1, :, :])
        zern_phase = numpy.zeros(reference_frame.shape)
        for j in range(y_exp.shape[0]):
            zern_phase += zer_gen.zernike_matrices[z + 1, y_exp[j], x_exp[j]] * camera_coords_x ** x_exp[j] * camera_coords_y ** y_exp[j]
        zernike_gradients_y[z],zernike_gradients_x[z] = numpy.gradient(zern_phase)


    for i in range(cent_num):
        print(i+1, "/", cent_num)
        half_size=int(PITCH/2)

        for z in range(zer_gen.zernike_num - 1):
            zer_to_cent_matrix[i+cent_num, z] = numpy.mean(
                zernike_gradients_y[z][int(spots_y[i]) - half_size:int(spots_y[i]) + half_size,
                int(spots_x[i]) - half_size:int(spots_x[i]) + half_size])
            zer_to_cent_matrix[i, z] = numpy.mean(
                zernike_gradients_x[z][int(spots_y[i]) - half_size:int(spots_y[i]) + half_size,
                int(spots_x[i]) - half_size:int(spots_x[i]) + half_size])


    acts_to_cent_matrix=numpy.zeros((reference_centroids.shape[0],69))

    for i in range(69):
        print(i + 1, "/", 69)
        # relax_mirror(dm)
        inps = numpy.zeros(69)
        inps[i]=POKE
        dm.send_direct_voltages(inps)
        time.sleep(0.1)
        for j in range(5):
            image = camera.get_frame().astype(float)
        centroids_p=find_centroids(image, spots_x, spots_y)-reference_centroids

        # plt.imshow(image)
        # plt.scatter(spots_x, spots_y, c=numpy.sqrt(centroids_p[:cent_num]**2+centroids_p[cent_num:]**2))
        # plt.scatter(center_x, center_y)
        # plt.plot(circle_x, circle_y)
        # plt.show()

        inps[i]=-POKE
        dm.send_direct_voltages(inps)
        time.sleep(0.1)
        for j in range(5):
            image = camera.get_frame().astype(float)
        centroids_n=find_centroids(image ,spots_x, spots_y)-reference_centroids

        # plt.imshow(image)
        # plt.scatter(spots_x, spots_y, c=numpy.sqrt(centroids_n[:cent_num]**2+centroids_n[cent_num:]**2))
        # plt.scatter(center_x, center_y)
        # plt.plot(circle_x, circle_y)
        # plt.show()

        centroids=(centroids_p-centroids_n)/2
        centroids[numpy.abs(centroids)<CENT_THRESHOLD]=0
        acts_to_cent_matrix[:, i] = centroids

        # plt.imshow(image)
        # plt.scatter(spots_x, spots_y, c=numpy.sqrt(centroids[:cent_num]**2+centroids[cent_num:]**2))
        # plt.scatter(center_x, center_y)
        # plt.plot(circle_x, circle_y)
        # plt.show()

        best_cent = numpy.argmax(centroids[:cent_num]**2)
        camera.stop_acquisition()
        print(spots_x[best_cent] - int(PITCH / 2), spots_y[best_cent] - int(PITCH / 2), int(PITCH), int(PITCH))
        camera.set_active_region(0, int(spots_y[best_cent]-PITCH/2), camera.get_size()[0], int(PITCH))
        camera.set_exposure_ms(0.04)

        dm.stop_loop()
        for amp in numpy.linspace(0.1,0.1,1):
            print(amp)
            wave_time_ms = 500
            n_steps = int(wave_time_ms * 0.001 / MIRROR_SAMPLE_DURATION / 2)
            voltage_sequence=numpy.zeros((n_steps,69))
            voltage_sequence[:int(n_steps/2),i]=-amp
            voltage_sequence[int(n_steps / 2):, i] = amp

            images=numpy.zeros((2000, int(PITCH),camera.get_size()[0]),dtype="uint8")


            dm.send_pattern_voltages(voltage_sequence, 0)
            time.sleep(3)
            camera.start_acquisition(10)
            c=0
            t=time.perf_counter()
            for j in range(2000):
                images[j]=camera.get_frame()
            print(time.perf_counter()-t)


            camera.stop_acquisition()
            dm.stop_loop()

            tifffile.imwrite("test_act_" + str(i).zfill(2) +"_amp_"+str(numpy.round(amp,1))+ ".tif", images)
        camera.set_full_chip()

        dm.start_direct_control()
        dm.send_direct_voltages(numpy.zeros(69))
        time.sleep(2.0)
        camera.start_acquisition()

        for j in range(5):
            image = camera.get_frame().astype(float)


    dm.stop_loop()

