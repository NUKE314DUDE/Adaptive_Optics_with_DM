import numpy as np
import matplotlib.pyplot as plt
import os
from Coordinates_Finder_Modules import precise_coord, precise_coord_fixed_ref, extrema_coordinates_gradient, show_coord, \
    exclude_proxi_points, img_preprocess, center_of_gravity_with_coord, average_distance, coord_diff
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)

FRAME_RATE = 1000
MIN_DIST = 10
COG_WINDOW_SCALER_1 = 0.2
COG_WINDOW_SCALER_2 = 0.25
STRIP_SCALER = 0.3
STRIP_OFFSET = -80

def strip_cut(image, scale = 1, off_set = 20):
    row,col = image.shape
    for col_front in range(col):
        if np.sum(img[:,col_front]) != 0:
            break
    for col_back in range(-1, -col, -1):
        if np.sum(img[:,col_back]) != 0:
            break
    col_back = col + 1 + col_back
    size = (col_back - col_front)
    col_front = col_front + round((1-scale)*size/2)
    col_back = col_back - round((1-scale)*size/2)
    return np.array((col_front+off_set, col_back+off_set))

if __name__ == '__main__':
    data_path = 'Data_Deposit/SH_response_whole_membrane_defocus.tif'
    raw_video = np.array(imageio.v2.imread(data_path))
    for lay in range(len(raw_video)):
        if np.sum(raw_video[lay,:]) != 0:
            cut_video = raw_video[lay:-1, :]
            break
    tester = cut_video[0]
    lim_front, lim_back = strip_cut(tester, scale = STRIP_SCALER, off_set = STRIP_OFFSET)
    strip_video = img_preprocess(cut_video[:,:,lim_front:lim_back])
    ref_coord = precise_coord(strip_video[0], MIN_DIST, COG_WINDOW_SCALER_1)
    ref_coord = center_of_gravity_with_coord(strip_video[0], ref_coord, COG_WINDOW_SCALER_2)
    coord_diff_history = []

    # length_jumper = 100
    # coord_jumper = [];img_jumper = [];index_jumper = 0
    # for i, img in enumerate(strip_video):
    #     current_coord = precise_coord(img, MIN_DIST, COG_WINDOW_SCALER_1)
    #     current_coord = center_of_gravity_with_coord(img, current_coord, COG_WINDOW_SCALER_2)
    #     # current_coord = precise_coord_fixed_ref(img, ref_coord, COG_WINDOW_SCALER_2)
    #     current_length = len(current_coord)
    #     if current_length < length_jumper:
    #         length_jumper = current_length
    #         coord_jumper = current_coord
    #         img_jumper = img
    #         index_jumper = i
    #     coord_history.append(current_coord)
    #     print(f'we are at {100*(i+1)/(len(strip_video))} percent.')
    # show_coord(img, current_coord); print(i)

    for i, img in enumerate(strip_video):
        current_coord = precise_coord_fixed_ref(img, ref_coord, COG_WINDOW_SCALER_2)
        coord_diff_history.append(coord_diff(ref_coord, current_coord))
        print(f'we are at {100*(i+1)/(len(strip_video))} percent.')
    coord_diff_history = np.array(coord_diff_history)
    plt.show()