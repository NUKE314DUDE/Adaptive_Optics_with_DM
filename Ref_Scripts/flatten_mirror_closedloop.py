from Lib64.asdk import DM
import os
os.add_dll_directory(os.getcwd())
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from zernike_generator import zernike
from scipy.signal import find_peaks
import numpy
import time
import matplotlib.pyplot as plt

PITCH = 43.48
THRESHOLD_1=15
THRESHOLD_2=9
POKE=0.2
CENT_THRESHOLD=0.5

def relax_mirror(dm):
    t_start=time.perf_counter()
    while time.perf_counter()-t_start<0.5:
        amp=numpy.exp(-(time.perf_counter()-t_start)/0.2)*numpy.cos((time.perf_counter()-t_start)/0.033*2*numpy.pi)
        dm.Send(numpy.ones(69)*amp)


def find_centroid(image,x,y):
    half_size=int(PITCH/2)
    square=(image[int(y)-half_size:int(y)+half_size,int(x)-half_size:int(x)+half_size]).astype(float)
    square=square-THRESHOLD_2
    square[square<0]=0
    x_c = numpy.arange(-int(PITCH / 2), +int(PITCH / 2))
    y_c = numpy.arange(-int(PITCH / 2), +int(PITCH / 2))
    x_c,y_c=numpy.meshgrid(x_c,y_c)

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


dm = DM("BOL131")
relax_mirror(dm)

time.sleep(0.1)

thorsdk=TLCameraSDK()
camera = thorsdk.open_camera(thorsdk.discover_available_cameras()[0])
camera.exposure_time_us = 40
camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
camera.image_poll_timeout_ms = 1000  # 1 second polling timeout

camera.arm(2)

camera.issue_software_trigger()

frame=None
while frame is None:
    frame=camera.get_pending_frame_or_null()
reference_frame = numpy.copy(frame.image_buffer).astype(float)
for i in range(4):
    frame = None
    while frame is None:
        frame = camera.get_pending_frame_or_null()
    reference_frame += numpy.copy(frame.image_buffer)
reference_frame=reference_frame/5

image=reference_frame.astype(float)
image=image-THRESHOLD_1
image[image<0]=0

x_profile=numpy.mean(image, axis=0)
peaks_x, peaks_x_prop=find_peaks(x_profile,distance=PITCH/2)
y_profile=numpy.mean(image, axis=1)
peaks_y, peaks_y_prop=find_peaks(y_profile,distance=PITCH/2)

center_x = numpy.mean(peaks_x)
center_y = numpy.mean(peaks_y)

n_peaks_x=peaks_x.shape[0]
n_peaks_y=peaks_y.shape[0]

n_peaks=numpy.amax(numpy.asarray([n_peaks_y,n_peaks_x]))
width=(n_peaks-1.0)*PITCH


# plt.plot(x_profile)
# plt.scatter(peaks_x,x_profile[peaks_x])
# plt.plot(y_profile)
# plt.scatter(peaks_y,y_profile[peaks_y])
# plt.show()

spots_array_1d=numpy.linspace(-width/2,width/2,n_peaks)
spots_x,spots_y = numpy.meshgrid(center_x-spots_array_1d,center_y-spots_array_1d)
radius=width/2

good_spots_indexes=numpy.where((spots_x-center_x)**2+(spots_y-center_y)**2<radius**2)
spots_x=spots_x[good_spots_indexes]
spots_y=spots_y[good_spots_indexes]


# spots_x=[]
# spots_y=[]
#
# while numpy.any(image>0):
#
#     spot = numpy.unravel_index(numpy.argmax(image),image.shape)
#     spots_y.append(spot[0])
#     spots_x.append(spot[1])
#     image[spot[0]-int(1.3*PITCH/2):spot[0]+int(1.3*PITCH/2),spot[1]-int(1.3*PITCH/2):spot[1]+int(1.3*PITCH/2)] = 0

cent_num=len(spots_x)
# spots_x = numpy.asarray(spots_x)
# spots_y = numpy.asarray(spots_y)
# center_x = numpy.mean(spots_x)
# center_y = numpy.mean(spots_y)
# radius=numpy.amax(numpy.sqrt((spots_x-center_x)**2+(spots_y-center_y)**2))+PITCH/2

camera_coords_x,camera_coords_y=numpy.meshgrid(numpy.arange(image.shape[1]),numpy.arange(image.shape[0]))
camera_coords_x=(camera_coords_x-center_x)/radius
camera_coords_y=(camera_coords_y-center_y)/radius


circ_points=1000
angle=numpy.linspace(0,2*numpy.pi,circ_points)

circle_x=center_x+radius*numpy.cos(angle)
circle_y=center_y+radius*numpy.sin(angle)


plt.imshow(reference_frame)
plt.scatter(spots_x,spots_y)
plt.scatter(center_x,center_y)
plt.plot(circle_x,circle_y)
plt.show()

frame=None
while frame is None:
    frame=camera.get_pending_frame_or_null()
reference_frame = numpy.copy(frame.image_buffer).astype(float)
for i in range(4):
    frame = None
    while frame is None:
        frame = camera.get_pending_frame_or_null()
    reference_frame += numpy.copy(frame.image_buffer)
reference_frame=reference_frame/5

reference_centroids=find_centroids(reference_frame.astype(float),spots_x,spots_y)


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
    relax_mirror(dm)
    inps = numpy.zeros(69)
    inps[i]=POKE
    dm.Send(inps)
    time.sleep(0.05)
    frame = camera.get_pending_frame_or_null()
    while frame is None:
        pass
        # print("frame #{} received!".format(frame.frame_count))
    image = frame.image_buffer.astype(float)
    centroids_p=find_centroids(image, spots_x, spots_y) - reference_centroids

    # plt.imshow(image)
    # plot_i=numpy.where(numpy.logical_or(numpy.abs(centroids_p[:cent_num])>CENT_THRESHOLD,numpy.abs(centroids_p[cent_num:])>CENT_THRESHOLD))
    # plt.quiver(reference_centroids[:cent_num][plot_i], reference_centroids[cent_num:][plot_i],centroids_p[:cent_num][plot_i], -centroids_p[cent_num:][plot_i],
    #            numpy.sqrt(centroids_p[:cent_num][plot_i]**2+centroids_p[cent_num:][plot_i]**2), scale=80)
    # plt.scatter(reference_centroids[:cent_num], reference_centroids[cent_num:], c=numpy.sqrt(centroids_p[:cent_num]**2+centroids_p[cent_num:]**2))
    # plt.scatter(center_x, center_y)
    # plt.plot(circle_x, circle_y)
    # plt.show()

    inps[i]=-POKE
    dm.Send(inps)
    time.sleep(0.05)
    frame = camera.get_pending_frame_or_null()
    while frame is None:
        pass
        # print("frame #{} received!".format(frame.frame_count))
    image = frame.image_buffer.astype(float)
    centroids_n=find_centroids(image ,spots_x, spots_y)-reference_centroids

    # plt.imshow(image)
    # plot_i=numpy.where(numpy.logical_or(numpy.abs(centroids_n[:cent_num])>CENT_THRESHOLD,numpy.abs(centroids_n[cent_num:])>CENT_THRESHOLD))
    # plt.quiver(reference_centroids[:cent_num][plot_i], reference_centroids[cent_num:][plot_i],centroids_n[:cent_num][plot_i], -centroids_n[cent_num:][plot_i],
    #            numpy.sqrt(centroids_n[:cent_num][plot_i]**2+centroids_n[cent_num:][plot_i]**2), scale=80)
    # plt.scatter(reference_centroids[:cent_num], reference_centroids[cent_num:], c=numpy.sqrt(centroids_n[:cent_num]**2+centroids_n[cent_num:]**2))
    # plt.scatter(center_x, center_y)
    # plt.plot(circle_x, circle_y)
    # plt.show()

    centroids=(centroids_p-centroids_n)/2
    centroids[numpy.abs(centroids)<CENT_THRESHOLD]=0
    acts_to_cent_matrix[:, i] = centroids/POKE

    # plt.imshow(image)
    # plt.scatter(spots_x, spots_y, c=numpy.sqrt(centroids[:cent_num]**2+centroids[cent_num:]**2))
    # plt.scatter(center_x, center_y)
    # plt.plot(circle_x, circle_y)
    # plt.show()


cent_to_act_matrix=numpy.linalg.pinv(acts_to_cent_matrix)


closed_loop_gain=0.2

acts=numpy.zeros(69)

reference_centroids=numpy.concatenate((spots_x, spots_y))

relax_mirror(dm)
for i in range(20):
    frame = camera.get_pending_frame_or_null()
    while frame is None:
        pass
        # print("frame #{} received!".format(frame.frame_count))
    image = frame.image_buffer.astype(float)
    centroids=find_centroids(image, spots_x, spots_y)-reference_centroids
    print(i, numpy.mean(numpy.abs(centroids)))
    acts-=closed_loop_gain*numpy.dot(cent_to_act_matrix,centroids)
    print(acts)
    relax_mirror(dm)
    dm.Send(acts)
    time.sleep(0.05)

plt.imshow(image)
plt.quiver(reference_centroids[:cent_num], reference_centroids[cent_num:],
           centroids[:cent_num], -centroids[cent_num:],
           numpy.sqrt(centroids[:cent_num] ** 2 + centroids[cent_num:] ** 2), scale=80)
plt.scatter(reference_centroids[:cent_num], reference_centroids[cent_num:],
            c=numpy.sqrt(centroids[:cent_num] ** 2 + centroids[cent_num:] ** 2))
plt.scatter(center_x, center_y)
plt.plot(circle_x, circle_y)
plt.show()


zer_to_act_matrix=numpy.dot(cent_to_act_matrix,zer_to_cent_matrix)
act_to_zer_matrix=numpy.linalg.pinv(zer_to_act_matrix)

numpy.save("flat_voltages.npy", acts)

print(numpy.dot(act_to_zer_matrix,acts))


for i in range(27):
    zer_to_act_matrix[:,i]=zer_to_act_matrix[:,i]/numpy.amax(numpy.abs(zer_to_act_matrix[:,i]))

numpy.save("zernike_matrix.npy", zer_to_act_matrix)

for z in range(zer_gen.zernike_num - 1):
    relax_mirror(dm)
    inps = numpy.zeros(27)
    inps[z] = 1.0
    zer_amp=POKE/numpy.amax(numpy.dot(zer_to_act_matrix,inps))
    inps[z]=zer_amp
    dm.Send(numpy.dot(zer_to_act_matrix,inps)+acts)
    time.sleep(0.05)
    frame = camera.get_pending_frame_or_null()
    while frame is None:
        pass
        # print("frame #{} received!".format(frame.frame_count))
    image = frame.image_buffer.astype(float)
    centroids_p=find_centroids(image ,spots_x, spots_y)-reference_centroids

    plt.imshow(image)
    plot_i=numpy.where(numpy.logical_or(numpy.abs(centroids_p[:cent_num])>CENT_THRESHOLD,numpy.abs(centroids_p[cent_num:])>CENT_THRESHOLD))
    plt.quiver(reference_centroids[:cent_num][plot_i], reference_centroids[cent_num:][plot_i],centroids_p[:cent_num][plot_i], -centroids_p[cent_num:][plot_i],
               numpy.sqrt(centroids_p[:cent_num][plot_i]**2+centroids_p[cent_num:][plot_i]**2), scale=80)
    plt.scatter(reference_centroids[:cent_num], reference_centroids[cent_num:], c=numpy.sqrt(centroids_p[:cent_num]**2+centroids_p[cent_num:]**2))
    plt.scatter(center_x, center_y)
    plt.plot(circle_x, circle_y)
    plt.show()

    inps[z]=-zer_amp
    dm.Send(numpy.dot(zer_to_act_matrix,inps)+acts)
    time.sleep(0.05)
    frame = camera.get_pending_frame_or_null()
    while frame is None:
        pass
        # print("frame #{} received!".format(frame.frame_count))
    image = frame.image_buffer.astype(float)
    centroids_n=find_centroids(image ,spots_x, spots_y)-reference_centroids

    plt.imshow(image)
    plot_i=numpy.where(numpy.logical_or(numpy.abs(centroids_n[:cent_num])>CENT_THRESHOLD,numpy.abs(centroids_n[cent_num:])>CENT_THRESHOLD))
    plt.quiver(reference_centroids[:cent_num][plot_i], reference_centroids[cent_num:][plot_i],centroids_n[:cent_num][plot_i], -centroids_n[cent_num:][plot_i],
               numpy.sqrt(centroids_n[:cent_num][plot_i]**2+centroids_n[cent_num:][plot_i]**2), scale=80)
    plt.scatter(reference_centroids[:cent_num], reference_centroids[cent_num:], c=numpy.sqrt(centroids_n[:cent_num]**2+centroids_n[cent_num:]**2))
    plt.scatter(center_x, center_y)
    plt.plot(circle_x, circle_y)
    plt.show()



camera.disarm()