# Post_01_1_LineStitch

import numpy as np
from numba import njit
from sklearn.metrics import mean_squared_error

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

@njit()
def Boolean_generator(block, Boolean, i, step):
    tof = []
    for m in range(block):
        B_select = Boolean[i + m * step] > 0
        tof.append(B_select)
    tof = np.array(tof)
    return tof

def search_and_count_for_block(Boolean, block, begin, end, step):
    out = np.zeros_like(Boolean)
    counts = 0
    for i in range(begin, end, step):
        tof = Boolean_generator(block, Boolean, i, step)
        if tof.all() and step < 0:
            out[i - block + 1:i + 1] += 1
            counts += 1
        elif tof.all() and step > 0:
            out[i:i + block] += 1
            counts += 1
    out[out > 0] = 1
    return out, counts

def match(array1, array2,n_line,erro_tolerance, step=1, ToLeft=False):
    start = len(array1[:, 0]) // 2
    Boolean = np.zeros(len(array2))
    continue_match = True
    block = 1
    if ToLeft:
        begin = len(Boolean) - 1
        end_O = -2
    else:
        begin = 0
        end_O = len(Boolean) + 1
    while continue_match:
        if abs(start) >= array1.shape[0]:
            return np.array([0])
        Boolean = np.where(abs(array2[:, 3] - array1[start, 3]) >= erro_tolerance * abs(array1[start, 3]), Boolean, 1)
        end = -step * block + end_O
        Boolean, counts = search_and_count_for_block(Boolean, block, begin, end, step)
        # print(counts)
        if counts <= n_line:
            return Boolean
        start += step
        block += 1

def find_overlap_region(data2,length_z_2,n_edge,reference_Vx):
    MSE = []
    for j in range(0,3):
        split_Z_top = find_nearest(data2[:, 2], data2[-1, 2] - length_z_2 * j)
        split_Z_bottom = find_nearest(data2[:, 2], data2[-1, 2] - length_z_2 * (j + 1))
        split_Z_mid = find_nearest(data2[:, 2], 0.5 * (data2[-1, 2] - length_z_2 * j + data2[-1, 2] - length_z_2 * (j + 1)))
        data2_top = data2[abs(data2[:, 2] - split_Z_top) <= 0.1]
        data2_top_Vx = data2_top[n_edge:-n_edge, 3]
        data2_mid = data2[abs(data2[:, 2] - split_Z_mid) <= 1]
        data2_mid_Vx = data2_mid[n_edge:-n_edge, 3]
        data2_bottom = data2[abs(data2[:, 2] - split_Z_bottom) <= 1]
        data2_bottom_Vx = data2_bottom[n_edge:-n_edge, 3]
        mse1 = mean_squared_error(reference_Vx, data2_top_Vx)
        mse2 = mean_squared_error(reference_Vx, data2_mid_Vx)
        mse3 = mean_squared_error(reference_Vx, data2_bottom_Vx)
        mse = mse1 + mse2 + mse3
        MSE.append(mse)
    index = MSE.index(min(MSE))
    split_Z_top = find_nearest(data2[:, 2], data2[-1, 2] - length_z_2 * index)
    split_Z_bottom = find_nearest(data2[:, 2], data2[-1, 2] - length_z_2 * (index + 1))
    con = np.array([data2[:, 2] <= split_Z_top, data2[:, 2] >= split_Z_bottom])
    data2_overlap = data2[con.all(axis=0)]
    return data2_overlap


def z_axis_stitch_2d(data1,data2,erro_tolerance,n_edge,reference_line_location,n_line):
    length_z = reference_line_location * abs(data1[0, 2] - data1[-1, 2])
    ref_z = find_nearest(data1[n_edge:-n_edge, 2], data1[0, 2] + length_z)
    reference_line = data1[abs(data1[:, 2] - ref_z) <= 0.1]  # find the reference line
    reference_line_wo = reference_line[n_edge:-n_edge, :]
    reference_Vx = reference_line_wo[:, 3]
    reference_Vz = reference_line_wo[:, 4]

    length_z_2 = (1 / 3) * abs(data2[0, 2] - data2[-1, 2])
    data2_overlap = find_overlap_region(data2,length_z_2,n_edge,reference_Vx)  # further minimize the overlapped region on FOV2

    location = match(reference_line, data2_overlap,n_line, erro_tolerance,step=-1, ToLeft=True)
    print("matching to the right")

    '''location=match(reference_line,data2_overlap,n_line, erro_tolerance,step=1,ToLeft=False)
    print("overlap from bottom left corner")'''

    processed_x = data2_overlap[:, 0] * location
    processed_FOV2_x = processed_x[processed_x != 0]
    processed_z = data2_overlap[:, 2] * location
    processed_FOV2_z = np.unique(processed_z[processed_z != 0])
    n_mse_overlap = []
    for n in range(len(processed_FOV2_z)):
        Overlap_data = data2[abs(data2[:, 2] - processed_FOV2_z[n]) <= 0.1]
        Overlap_Vx = Overlap_data[n_edge:-n_edge, 3]
        Overlap_Vz = Overlap_data[n_edge:-n_edge, 4]
        nrmseVx = np.linalg.norm(reference_Vx - Overlap_Vx) / np.linalg.norm(reference_Vx)
        nrmseVz = np.linalg.norm(reference_Vz - Overlap_Vz) / np.linalg.norm(reference_Vz)
        nrmse = 0.5 * (nrmseVx + nrmseVz)*100
        n_mse_overlap.append(nrmse)
    print(n_mse_overlap)
    min_index = np.array(n_mse_overlap).argmin()
    overlap_z = processed_FOV2_z - ref_z
    D_move = overlap_z[min_index]
    print(D_move)
    return D_move


