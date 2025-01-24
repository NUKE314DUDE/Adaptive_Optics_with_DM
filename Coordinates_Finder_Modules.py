
import time
import matplotlib.pyplot as plt
import numpy as np
import os
os.add_dll_directory(os.getcwd())
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, KDTree
from Zernike_Polynomials_Modules import min_circle

def closest_to_center(coordinates):
    """
    Find the point that is closest to the center
    :param coordinates: a set of coordinates    :return:  of the proxi point
    """
    center,_ = min_circle(coordinates)
    distance_to_center = np.linalg.norm(coordinates - center, axis = 1)
    proxi_index = np.argmin(distance_to_center)
    proxi_coord = coordinates[proxi_index]
    return np.array(proxi_coord)

def grid_from_proxi_center(coordinates, avg_mode = False, pitch = 150e-6, pixel_size = 4.8e-6):
    """
    Calculate the ideal position of the centroids using the specs of the lens-array
    :param coordinates: ref_coord obtained from reference measurement
    :param avg_mode: choose whether to use average distance of the ref_coord as grid spacing
    :param pitch: of the lens-array
    :param pixel_size: of the WFS camera
    :return: grid nodes position
    """
    proxi_center = closest_to_center(coordinates)
    center, radius = min_circle(coordinates, scale = 1.05)
    if avg_mode:
        grid_gap = average_distance(coordinates, k = 7)
    else:
        grid_gap = pitch/pixel_size
    lin_x = np.arange(- radius, radius + grid_gap, grid_gap)
    lin_y = np.arange(- radius, radius + grid_gap, grid_gap)
    grid_x, grid_y = np.meshgrid(lin_x, lin_y)
    grid_x += proxi_center[0]
    grid_y += proxi_center[1]
    grid_nodes = np.stack((grid_x, grid_y), axis = -1).reshape(-1, 2)
    distance_to_proxi_center = np.linalg.norm(grid_nodes - proxi_center, axis = 1)
    proxi_amend = grid_nodes[np.argmin(distance_to_proxi_center)] - proxi_center
    grid_nodes = sorted(grid_nodes, key = lambda grid_nodes: (grid_nodes[0], grid_nodes[1])) - proxi_amend
    distance_to_actual_center = np.linalg.norm(grid_nodes - center, axis = 1)
    include_index = distance_to_actual_center < radius
    return np.array(grid_nodes[include_index])

def grid_nodes_refine(grid_coord, ref_coord):
    """
    Refine the grid nodes to match the quantity and the rotation of the ref_coord
    :param grid_coord: grid nodes generated
    :param ref_coord: padded ref_coord
    :return: refined grid nodes
    """
    refined_grid_nodes = []
    for i in ref_coord:
        dist = np.linalg.norm(grid_coord - i, axis = 1)
        refined_grid_nodes.append(grid_coord[np.argmin(dist)])
    refined_grid_nodes = sorted(refined_grid_nodes, key = lambda refined_grid_nodes: (refined_grid_nodes[0], refined_grid_nodes[1]))
    _, index_map = coord_diff(ref_coord, refined_grid_nodes, return_index = True)
    matched_ref_coord = []
    matched_grid_coord = []
    for i, j in enumerate(index_map):
        matched_ref_coord.append(ref_coord[j[0]])
        matched_grid_coord.append(refined_grid_nodes[j[1]])
    matched_ref_coord = np.array(matched_ref_coord)
    matched_grid_coord = np.array(matched_grid_coord)
    ref_center,_ = min_circle(ref_coord)
    grid_center,_ = min_circle(refined_grid_nodes)
    centered_ref_coord = matched_ref_coord - ref_center
    centered_grid_coord = matched_grid_coord - grid_center
    norm_ref_coord = np.linalg.norm(centered_ref_coord)
    norm_grid_coord = np.linalg.norm(centered_grid_coord)
    ref_normalized = centered_ref_coord / norm_ref_coord
    grid_normalized = centered_grid_coord / norm_grid_coord
    H = ref_normalized.T @ grid_normalized
    U, S, Vt = np.linalg.svd(H)
    Rotation = np.array(Vt.T @ U.T)
    if np.linalg.det(Rotation) < 0:
        Vt[1, :] *= -1
        Rotation = np.array(Vt.T @ U.T)
    final_grid_coord = centered_grid_coord @ Rotation
    angle = np.arccos(Rotation[0, 0])
    print(f"grid is rotated by {angle} rad.")
    return np.array(final_grid_coord + grid_center)

def img_threshold(image, intensity_threshold = 30):
    """
    Exclude certain intensities
    :param image: targeted image
    :param intensity_threshold: minimum pixel value
    :return: filtered image
    """
    filtered_img = np.where(image > intensity_threshold, image, 1e-6)
    return filtered_img

def show_coord(image, coord, circle = None):
    """
    Mark desired coordinates on the image
    :param circle: (x, y), radius
    :param image: targeted image
    :param coord: coordinates to mark
    :return: plot the marked image im a new window
    """
    if circle is None:
        [height, width] = image.shape
        plt.ion()
        fig, ax = plt.subplots()
        ax.imshow(image)
        for x, y in coord:
            x = max(0, min(x, height - 1))
            y = max(0, min(y, width - 1))
            ax.text(y, x, u'\u2715', color='red', fontsize=12, ha='center', va='center')  # Add markers for founded extrema
        ax.set_title(f'A total of {int(len(coord))} centroids marked')
        plt.show()
        plt.ioff()
    else:
        [height, width] = image.shape
        plt.ion()
        fig, ax = plt.subplots()
        ax.imshow(image)
        for x, y in coord:
            x = max(0, min(x, height - 1))
            y = max(0, min(y, width - 1))
            ax.text(y, x, u'\u2715', color='red', fontsize=12, ha='center', va='center')
        center = circle[0]
        radius = circle[1]
        ax.text(center[1], center[0], u'\u2715', color='g', fontsize=16, ha='center', va='center') # mark the center
        theta = np.linspace(0, 2 * np.pi, 400)
        x_circle = center[0] + radius * np.cos(theta)
        y_circle = center[1] + radius * np.sin(theta)
        ax.plot(y_circle, x_circle, color="b") # draw a circle
        ax.set_title(f'A total of {int(len(coord))} centroids marked \n Centered on {center} radius: {radius}')
        plt.show()
        plt.ioff()

def show_coord_diff(image, ref_coord, test_coord):
    """
    Draw a line between two sets of coordinates
    :param image: original image
    :param ref_coord: reference coordinates
    :param test_coord: targeted coordinates
    :return: plot on the image with markers
    """
    if np.array(ref_coord).shape != np.array(test_coord).shape:
        raise ValueError('Coordinates quantity does not match!')
    [height, width] = image.shape
    diff = coord_diff(ref_coord, test_coord)
    plt.ion()
    plt.figure()
    plt.imshow(image)
    for x, y in ref_coord:
        x = max(0, min(x, height - 1))
        y = max(0, min(y, width - 1))
        plt.plot(y, x, 'go', markersize = 1)
    for x, y in test_coord:
        x = max(0, min(x, height - 1))
        y = max(0, min(y, width - 1))
        plt.plot(y, x, 'ro', markersize = 1)
    for i, j in enumerate(diff):
        plt.arrow(ref_coord[i][1], ref_coord[i][0], j[1], j[0],head_width=0.05, head_length=0.1, fc='blue', ec='blue',linewidth = 1)
    plt.show()
    plt.ioff()

def center_of_gravity_with_coord(img, coord, scaler = 0.5, disp_time = False):
    """
    Calculate the precise location of each centroid based on their pixel coordinates
    :param img: intensity distribution
    :param coord: extrema coordinates
    :param scaler: window size is calculated using the average distance between points, this adjusts the ratio
    :param disp_time: to give the time consumption
    :return:
    """
    start_time = time.time()
    row, col = img.shape
    coord = np.array(coord).astype(int)
    window_size = int(np.floor(average_distance(coord)*scaler))
    window_parameter = (- window_size, window_size + 1)
    center_of_gravity_list = []
    for i, j in enumerate(coord):
        window_intensity_data = img[j[0]+window_parameter[0]:j[0]+window_parameter[1],
                                j[1]+window_parameter[0]:j[1]+window_parameter[1]]
        window_index_data = np.meshgrid(np.arange(j[0] + window_parameter[0], j[0] + window_parameter[1]),
                                        np.arange(j[1] + window_parameter[0], j[1] + window_parameter[1]),
                                        indexing='ij')
        # print(window_intensity_data.shape, window_index_data[0].shape)
        if window_intensity_data.shape != window_index_data[0].shape:
            continue
        cog_x = np.sum(np.multiply(window_intensity_data, window_index_data[0]))/np.sum(window_intensity_data)
        cog_y = np.sum(np.multiply(window_intensity_data, window_index_data[1]))/np.sum(window_intensity_data)
        center_of_gravity_list.append((cog_x,cog_y))
        center_of_gravity_list = sorted(center_of_gravity_list, key = lambda center_of_gravity: (center_of_gravity[0], center_of_gravity[1]))
    end_time = time.time()
    if disp_time:
        print(f'Spent {end_time - start_time} seconds calculating the COG')
    return np.array(center_of_gravity_list)

def exclude_proxi_points(coord, min_dist = 30, disp_time = False):
    """
    筛选出彼此距离过近的点，但仅保留其中一个。 (AI powered)
    参数:
    coord (numpy.ndarray): 形状为 (n, 2) 的坐标数组。
    min_dist (float): 用于筛选的最小距离。
    返回:
    numpy.ndarray: 筛选后的坐标。
    """
    start_time = time.time()
    tree = KDTree(coord)
    refined_coord = []
    checked_points = set()
    for i, point in enumerate(coord):
        if i in checked_points:
            continue
        # 查询树以获取指定距离内的邻居
        indices = tree.query_ball_point(point, r = min_dist, workers = -1)
        # 将当前点添加到筛选后的列表中
        refined_coord.append(point)
        # 将邻域内的所有点标记为已处理
        for idx in indices:
            checked_points.add(idx)
    refined_coord = sorted(refined_coord, key = lambda refined_coord: (refined_coord[0], refined_coord[1]))
    end_time = time.time()
    if disp_time:
        print(f'Spent {end_time - start_time} seconds filtering the coordinates')
    return np.array(refined_coord)

def extrema_coordinates_ergodic(image, window_size, mode = 'max', min_dist = 10, disp_time = False, disp_coord = False):
    """
    Loop over the entire image to find the coordinates of the extrema, accurate but slow
    :param image: targeted image for centroids positioning
    :param window_size: search parameter for each pixel
    :param mode: to find maximum or minimum value
    :param min_dist: minimum distance between coordinates
    :param disp_time: to give the time consumption
    :param disp_coord: to print hte results
    :return: set of coordinates
    """
    start_time = time.time()
    if len(image.shape) != 2:
        raise ValueError('Input should be a 2D array')
    row, col = image.shape
    extrema_coord = []
    for idy in range(window_size, row - window_size):
        for idx in range(window_size, col - window_size):

            top_w, bot_w = idy - window_size, idy + window_size + 1
            lft_w, rgt_w = idx - window_size, idx + window_size + 1
            top_e, bot_e = idy - min_dist, idy + min_dist + 1
            lft_e, rgt_e = idx - min_dist, idx + min_dist + 1

            clipped = image[top_w:bot_w, lft_w:rgt_w]
            center_val = clipped[window_size, window_size]#Take out a small portion of data for local comparison

            if mode == 'max' and np.all(clipped <= center_val) and image[idy, idx] != 0\
                    or mode == 'min' and np.all(clipped >= center_val) and image[idy, idx] != 0:
                    extrema_coord.append((idy, idx))#Determine whether the central value is truly an extrema
                    image[top_e:bot_e, lft_e:rgt_e] = 0
    end_time = time.time()
    extrema_coord = sorted(extrema_coord, key = lambda extrema_coord: (extrema_coord[0], extrema_coord[1]))
    if disp_time:
        print(f'Spent {end_time - start_time} seconds finding the extrema (Ergodic)')
    if disp_coord:
        print(f'Found {len(extrema_coord)} {mode} value(s)')
    return extrema_coord

def extrema_coordinates_gradient(image, gradient_threshold_percentage = None, disp_time = False, disp_coord = False):
    """
    Find the rough coordinates of extrema using local gradient
    :param image: targeted image for centroids positioning
    :param gradient_threshold_percentage: [low. high] thresholds in percentage based on the maximum gradient_abs
    :param disp_time: to give the time consumption
    :param disp_coord: to print the results
    :return: set of rough coordinates
    """
    if gradient_threshold_percentage is None:
        gradient_threshold_percentage = [0, 0.2]
    start_time = time.time()
    if len(image.shape) != 2:
        raise ValueError('Input should be a 2D array')
    extrema_coord = []
    gradient = np.gradient(image)
    gradient_abs_sq = (gradient[0] ** 2 + gradient[1] ** 2)
    rough_mask = np.logical_and(np.max(gradient_abs_sq) * gradient_threshold_percentage[0] < gradient_abs_sq,
                                gradient_abs_sq < np.max(gradient_abs_sq) * (gradient_threshold_percentage[1]))

    end_time = time.time()

    if disp_time:
        print(f'Spent {end_time - start_time} seconds finding the extrema (Gradient)')
    if disp_coord:
        print(f'Found {len(extrema_coord)} extreme value(s)')
    rough_coord = np.argwhere(rough_mask)
    rough_coord = sorted(rough_coord, key = lambda rough_coord: (rough_coord[0], rough_coord[1]))
    # print(np.array(gradient).shape)
    return rough_coord

def average_distance(coord, k = 4):
    """
    Calculate the average distance between neighbouring points
    :param k: neighbours to find
    :param coord: set of coordinates
    :return: average distance
    """
    tree = cKDTree(coord)
    distances = []
    for i, p1 in enumerate(coord):
        dist,_ = tree.query(p1, k, workers = -1)
        distances.append(dist)
    if len(distances) > 0:
        return np.mean(distances)
    else:
        return 0

def coord_diff(ref_coord, test_coord, return_index = False):
    if len(ref_coord) != len(test_coord):
        raise ValueError('Coordinates quantity does not match!')
    else:
        ref_coord = np.array(ref_coord)
        test_coord = np.array(test_coord)
        matched_diff = []
        index = []
        dist = np.zeros((len(ref_coord), len(ref_coord)))
        for i in range(len(ref_coord)):
            for j in range(len(test_coord)):
                dist[i, j] = np.linalg.norm(ref_coord[i] - test_coord[j])
        matched_col, matched_row = linear_sum_assignment(dist)
        matched_index = zip(matched_col, matched_row)
    for i in matched_index:
        index.append(i)
        matched_diff.append(test_coord[i[1]] - ref_coord[i[0]])
    if return_index:
        return np.array(matched_diff), np.array(index)
    else:
        return np.array(matched_diff)

def precise_coord(image, min_dis = 20, cog_scale = 0.5):
    pre_processed_img = img_threshold(gaussian_filter(image, sigma = 1))
    pixel_coord = extrema_coordinates_gradient(pre_processed_img)
    refined_coord = exclude_proxi_points(pixel_coord, min_dis)
    cog_coord = center_of_gravity_with_coord(pre_processed_img, refined_coord, cog_scale)
    return cog_coord

def precise_coord_fixed_ref(image, refined_ref_coord, cog_scale = 0.5):
    pre_processed_img = img_threshold(gaussian_filter(image, sigma=1))
    cog_coord = center_of_gravity_with_coord(pre_processed_img, refined_ref_coord, cog_scale)
    return cog_coord

def deformable_mirror_random_test_input(test_range, test_length, degree_of_freedom = 57, sym = False):
    """
    Generate random voltages for testing the probed matrix
    :param sym: chose to add minus input or not
    :param degree_of_freedom: degree of freedom of the DM
    :param test_range: range im which the voltage is picked
    :param test_length: desired number of test patterns
    :return: columns of input voltage(s)
    """
    if sym:
        input_matrix = np.random.uniform(- test_range, test_range, (test_length, degree_of_freedom))
    else:
        input_matrix = np.random.uniform(0, test_range, (test_length, degree_of_freedom))
    return input_matrix

def deformable_mirror_single_probe_input(probe_ran, degree_of_freedom = 57):
    """
    Generate scan voltage set
    :param probe_ran:
    :param degree_of_freedom: DoF of the deformable mirror
    :return: an initial voltage (all zero) and a matrix of scan voltages
    """
    input_matrix = np.zeros((int(degree_of_freedom),int(degree_of_freedom)))
    np.fill_diagonal(input_matrix, probe_ran)
    initial_input = np.zeros(degree_of_freedom)
    return initial_input, input_matrix

def img_preprocess(img):
    processed_img = []
    for i in img:
        processed_img.append(img_threshold(gaussian_filter(i, 1)))
    return np.array(processed_img)
