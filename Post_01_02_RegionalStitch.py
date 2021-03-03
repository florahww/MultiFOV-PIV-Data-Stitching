# Post_01_02_RegionalStitch

import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def find_match_region(n_edge,Vx_overlap, Vx_reference,Vz_overlap,Vz_reference,data2_overlap_x,reference_region_x,data2_overlap_z,reference_region_z,reference_x,reference_z):
    nmse = []

    p_list = []
    q_list = []
    for p in range(n_edge, len(data2_overlap_z) - len(reference_region_z) + 1):
        for q in range(n_edge, len(data2_overlap_x) - len(reference_region_x) + 1):
            match_region_Vx = Vx_overlap[p:p + reference_z, q:q + reference_x]
            match_region_Vz = Vz_overlap[p:p + reference_z, q:q + reference_x]

            Vx_nmse_overlap = np.linalg.norm(Vx_reference - match_region_Vx) / np.linalg.norm(Vx_reference)
            Vz_nmse_overlap = np.linalg.norm(Vz_reference - match_region_Vz) / np.linalg.norm(Vz_reference)

            overall_nmse=0.5*(Vx_nmse_overlap+Vz_nmse_overlap)*100
            nmse.append(overall_nmse)

            p_list.append(p)
            q_list.append(q)
    index = nmse.index(min(nmse))
    p_i = p_list[index]
    q_i = q_list[index]
    return nmse, p_i, q_i,index



def x_axis_stitch_2d (data_FOV1,data_FOV2,reference_x,reference_z,region,n_edge):
    x_raw = data_FOV1[:, 0]
    x = np.unique(x_raw)
    z_raw = data_FOV1[:, 2]
    z = np.unique(z_raw)
    n_region = int(region * len(x))
    Vx1 = data_FOV1[:, 3]
    Vz1 = data_FOV1[:, 4]
    x_FOV1 = np.reshape(x_raw, (len(z), len(x)))
    z_FOV1 = np.reshape(z_raw, (len(z), len(x)))
    Vx_FOV1 = np.reshape(Vx1, (len(z), len(x)))
    Vz_FOV1 = np.reshape(Vz1, (len(z), len(x)))

    x_crop1 = np.delete(x_FOV1, np.s_[0:n_edge], 0)
    x_crop2 = np.delete(x_crop1, np.s_[-n_edge:], 0)
    x_crop3 = np.delete(x_crop2, np.s_[-n_edge:], 1)
    x_crop = np.delete(x_crop3, np.s_[0:-n_region], 1)

    z_crop1 = np.delete(z_FOV1, np.s_[0:n_edge], 0)
    z_crop2 = np.delete(z_crop1, np.s_[-n_edge:], 0)
    z_crop3 = np.delete(z_crop2, np.s_[-n_edge:], 1)
    z_crop = np.delete(z_crop3, np.s_[0:-n_region], 1)

    Vx_crop1 = np.delete(Vx_FOV1, np.s_[0:n_edge], 0)
    Vx_crop2 = np.delete(Vx_crop1, np.s_[-n_edge:], 0)
    Vx_crop3 = np.delete(Vx_crop2, np.s_[-n_edge:], 1)
    Vx_crop = np.delete(Vx_crop3, np.s_[0:-n_region], 1)

    Vz_crop1 = np.delete(Vz_FOV1, np.s_[0:n_edge], 0)
    Vz_crop2 = np.delete(Vz_crop1, np.s_[-n_edge:], 0)
    Vz_crop3 = np.delete(Vz_crop2, np.s_[-n_edge:], 1)
    Vz_crop = np.delete(Vz_crop3, np.s_[0:-n_region], 1)
    dVz = np.gradient(Vz_crop, axis=1)
    dx = np.gradient(x_crop, axis=1) / 1000
    dVx = np.gradient(Vx_crop, axis=0)
    dz = np.gradient(z_crop, axis=0) / 1000
    vorticity = (dVz / dx) - (dVx / dz)
    abs_vor = abs(vorticity)
    shape = np.shape(abs_vor)
    max_index = abs_vor.argmax()
    loc = np.unravel_index(max_index, abs_vor.shape)

    if loc[0] <= reference_z // 2:
        ref_z1 = 0
    elif shape[0]- loc[0] < reference_z // 2:
        ref_z1 = shape[0] - reference_z
    else:
        ref_z1 = loc[0] - reference_z // 2

    if loc[1] <= reference_x // 2:
        ref_x1 = 0
    elif n_region - loc[1] < reference_x // 2:
        ref_x1 = n_region - reference_x
    else:
        ref_x1 = loc[1] - reference_x // 2

    x_crop_c1 = np.delete(x_crop, np.s_[0:ref_z1], 0)
    x_crop_c2 = np.delete(x_crop_c1, np.s_[reference_z:], 0)
    x_crop_c3 = np.delete(x_crop_c2, np.s_[0:ref_x1], 1)
    x_c = np.delete(x_crop_c3, np.s_[reference_x:], 1)

    z_crop_c1 = np.delete(z_crop, np.s_[0:ref_z1], 0)
    z_crop_c2 = np.delete(z_crop_c1, np.s_[reference_z:], 0)
    z_crop_c3 = np.delete(z_crop_c2, np.s_[0:ref_x1], 1)
    z_c = np.delete(z_crop_c3, np.s_[reference_x:], 1)

    Vx_crop_c1 = np.delete(Vx_crop, np.s_[0:ref_z1], 0)
    Vx_crop_c2 = np.delete(Vx_crop_c1, np.s_[reference_z:], 0)
    Vx_crop_c3 = np.delete(Vx_crop_c2, np.s_[0:ref_x1], 1)
    Vx_c = np.delete(Vx_crop_c3, np.s_[reference_x:], 1)

    Vz_crop_c1 = np.delete(Vz_crop, np.s_[0:ref_z1], 0)
    Vz_crop_c2 = np.delete(Vz_crop_c1, np.s_[reference_z:], 0)
    Vz_crop_c3 = np.delete(Vz_crop_c2, np.s_[0:ref_x1], 1)
    Vz_c = np.delete(Vz_crop_c3, np.s_[reference_x:], 1)

    reference_region_z = z_c[:, 0]
    reference_region_x = x_c[0]
    reference_region_Vx = Vx_c
    reference_region_Vz = Vz_c


    data2_overlap_x = np.unique(data_FOV2[:, 0])
    data2_overlap_z = np.unique(data_FOV2[:, 2])
    data2_overlap_Vx = np.reshape(data_FOV2[:, 3], (len(data2_overlap_z), len(data2_overlap_x)))
    data2_overlap_Vz = np.reshape(data_FOV2[:, 4], (len(data2_overlap_z), len(data2_overlap_x)))

    nmse, p, q,index = find_match_region(n_edge,data2_overlap_Vx, reference_region_Vx, data2_overlap_Vz,
                                                            reference_region_Vz,data2_overlap_x,reference_region_x,data2_overlap_z,reference_region_z,reference_x,reference_z)
    ref_x = [reference_region_x[0], reference_region_x[-1]]
    ref_z = [reference_region_z[0], reference_region_z[-1]]

    match_x = [data2_overlap_x[q], data2_overlap_x[q + reference_x - 1]]
    match_z = [data2_overlap_z[p], data2_overlap_z[p + reference_z - 1]]

    match = data2_overlap_x[q:q + reference_x]
    D_move = match_x[0] - ref_x[0]
    return D_move,nmse

def z_axis_stitch_2d (data_FOV1,data_FOV2,reference_x,reference_z,region,n_edge):
    x_raw = data_FOV1[:, 0]
    x = np.unique(x_raw)
    z_raw = data_FOV1[:, 2]
    z = np.unique(z_raw)
    n_region = int(region * len(z))
    Vx1 = data_FOV1[:, 3]
    Vz1 = data_FOV1[:, 4]
    x_FOV1 = np.reshape(x_raw, (len(z), len(x)))
    z_FOV1 = np.reshape(z_raw, (len(z), len(x)))
    Vx_FOV1 = np.reshape(Vx1, (len(z), len(x)))
    Vz_FOV1 = np.reshape(Vz1, (len(z), len(x)))

    x_crop1 = np.delete(x_FOV1, np.s_[0:n_edge], 0)
    x_crop2 = np.delete(x_crop1, np.s_[n_region:], 0)
    x_crop3 = np.delete(x_crop2, np.s_[0:n_edge], 1)
    x_crop = np.delete(x_crop3, np.s_[-n_edge:], 1)

    z_crop1 = np.delete(z_FOV1, np.s_[0:n_edge], 0)
    z_crop2 = np.delete(z_crop1, np.s_[n_region:], 0)
    z_crop3 = np.delete(z_crop2, np.s_[0:n_edge], 1)
    z_crop = np.delete(z_crop3, np.s_[-n_edge:], 1)

    Vx_crop1 = np.delete(Vx_FOV1, np.s_[0:n_edge], 0)
    Vx_crop2 = np.delete(Vx_crop1, np.s_[n_region:], 0)
    Vx_crop3 = np.delete(Vx_crop2, np.s_[0:n_edge], 1)
    Vx_crop = np.delete(Vx_crop3, np.s_[-n_edge:], 1)

    Vz_crop1 = np.delete(Vz_FOV1, np.s_[0:n_edge], 0)
    Vz_crop2 = np.delete(Vz_crop1, np.s_[n_region:], 0)
    Vz_crop3 = np.delete(Vz_crop2, np.s_[0:n_edge], 1)
    Vz_crop = np.delete(Vz_crop3, np.s_[-n_edge:], 1)

    dVz = np.gradient(Vz_crop, axis=1)
    dx = np.gradient(x_crop, axis=1) / 1000
    dVx = np.gradient(Vx_crop, axis=0)
    dz = np.gradient(z_crop, axis=0) / 1000
    vorticity = (dVz / dx) - (dVx / dz)
    abs_vor = abs(vorticity)
    shape = np.shape(abs_vor)
    max_index = abs_vor.argmax()
    loc = np.unravel_index(max_index, abs_vor.shape)

    if loc[0] <= reference_z // 2:
        ref_z1 = 0
    elif n_region - loc[0] < reference_z // 2:
        ref_z1 = n_region - reference_z
    else:
        ref_z1 = loc[0] - reference_z // 2
    if loc[1] <= reference_x // 2:
        ref_x1 = 0
    elif shape[1] - loc[1] < reference_x // 2:
        ref_x1 = shape[1] - reference_x
    else:
        ref_x1 = loc[1] - reference_x // 2

    x_crop_c1 = np.delete(x_crop, np.s_[0:ref_z1], 0)
    x_crop_c2 = np.delete(x_crop_c1, np.s_[reference_z:], 0)
    x_crop_c3 = np.delete(x_crop_c2, np.s_[0:ref_x1], 1)
    x_c = np.delete(x_crop_c3, np.s_[reference_x:], 1)

    z_crop_c1 = np.delete(z_crop, np.s_[0:ref_z1], 0)
    z_crop_c2 = np.delete(z_crop_c1, np.s_[reference_z:], 0)
    z_crop_c3 = np.delete(z_crop_c2, np.s_[0:ref_x1], 1)
    z_c = np.delete(z_crop_c3, np.s_[reference_x:], 1)

    Vx_crop_c1 = np.delete(Vx_crop, np.s_[0:ref_z1], 0)
    Vx_crop_c2 = np.delete(Vx_crop_c1, np.s_[reference_z:], 0)
    Vx_crop_c3 = np.delete(Vx_crop_c2, np.s_[0:ref_x1], 1)
    Vx_c = np.delete(Vx_crop_c3, np.s_[reference_x:], 1)

    Vz_crop_c1 = np.delete(Vz_crop, np.s_[0:ref_z1], 0)
    Vz_crop_c2 = np.delete(Vz_crop_c1, np.s_[reference_z:], 0)
    Vz_crop_c3 = np.delete(Vz_crop_c2, np.s_[0:ref_x1], 1)
    Vz_c = np.delete(Vz_crop_c3, np.s_[reference_x:], 1)

    reference_region_z = z_c[:, 0]
    reference_region_x = x_c[0]
    reference_region_Vx = Vx_c
    reference_region_Vz = Vz_c


    data2_overlap_x = np.unique(data_FOV2[:, 0])
    data2_overlap_z = np.unique(data_FOV2[:, 2])

    data2_overlap_Vx = np.reshape(data_FOV2[:, 3], (len(data2_overlap_z), len(data2_overlap_x)))

    data2_overlap_Vz = np.reshape(data_FOV2[:, 4], (len(data2_overlap_z), len(data2_overlap_x)))

    nmse, p, q,index = find_match_region(n_edge,data2_overlap_Vx, reference_region_Vx, data2_overlap_Vz,
                                                            reference_region_Vz,data2_overlap_x,reference_region_x,data2_overlap_z,reference_region_z,reference_x,reference_z)
    ref_x = [reference_region_x[0], reference_region_x[-1]]
    ref_z = [reference_region_z[0], reference_region_z[-1]]

    match_x = [data2_overlap_x[q], data2_overlap_x[q + reference_x - 1]]
    match_z = [data2_overlap_z[p], data2_overlap_z[p + reference_z - 1]]

    match = data2_overlap_x[q:q + reference_x]
    D_move = match_z[0] - ref_z[0]
    print(reference_region_x[0],reference_region_x[-1])
    print(reference_region_z[0], reference_region_z[-1])
    print('nmse',min(nmse))
    '''print(D_move)

    print('overlap min rmse=', min(rmse), rmse[index], Vx_rmse[index], Vz_rmse[index])
    print('match x , z=', match_x, match_z)
    print('reference x , z=', ref_x, ref_z)
    print('move in z-axis=', match_z[0] - ref_z[0], match_z[1] - ref_z[1])
    print('move in x-axis=', match_x[0] - ref_x[0], match_x[1] - ref_x[1])'''
    return D_move,nmse

def find_match_region_3d (n_edge,Vx_overlap, Vx_reference,Vz_overlap,Vz_reference,Vy_overlap,Vy_reference,data2_overlap_x,reference_region_x,data2_overlap_y,reference_region_y,reference_x,reference_y):
    nmse = []
    p_list = []
    q_list = []
    for p in range(n_edge, len(data2_overlap_y) - len(reference_region_y) + 1):
        for q in range(n_edge, len(data2_overlap_x) - len(reference_region_x) + 1):
            match_region_Vx = Vx_overlap[p:p + reference_y, q:q + reference_x]
            match_region_Vy = Vy_overlap[p:p + reference_y, q:q + reference_x]
            match_region_Vz = Vz_overlap[p:p + reference_y, q:q + reference_x]

            Vx_nmse_overlap = np.linalg.norm(Vx_reference - match_region_Vx) / np.linalg.norm(Vx_reference)
            Vz_nmse_overlap = np.linalg.norm(Vz_reference - match_region_Vz) / np.linalg.norm(Vz_reference)
            Vy_nmse_overlap = np.linalg.norm(Vy_reference - match_region_Vy) / np.linalg.norm(Vy_reference)

            overall_nmse = (1/3) * (Vx_nmse_overlap + Vy_nmse_overlap + Vz_nmse_overlap)*100

            nmse.append(overall_nmse)

            p_list.append(p)
            q_list.append(q)
    index = nmse.index(min(nmse))
    p_i = p_list[index]
    q_i = q_list[index]
    return nmse, p_i, q_i,index



def y_axis_stitch_3d (data_FOV1,data_FOV2,reference_x,reference_y,region,n_edge):
    x_raw = data_FOV1[:, 0]
    x = np.unique(x_raw)
    y_raw = data_FOV1[:, 1]
    y = np.unique(y_raw)
    n_region = int(region * len(y))
    Vx1 = data_FOV1[:, 3]
    Vy1 = data_FOV1[:, 4]
    Vz1 = data_FOV1[:, 5]

    x_FOV1 = np.reshape(x_raw, (len(y), len(x)))
    y_FOV1 = np.reshape(y_raw, (len(y), len(x)))
    Vx_FOV1 = np.reshape(Vx1, (len(y), len(x)))
    Vy_FOV1 = np.reshape(Vy1, (len(y), len(x)))
    Vz_FOV1 = np.reshape(Vz1, (len(y), len(x)))

    x_crop1 = np.delete(x_FOV1, np.s_[0:n_edge], 0)
    x_crop2 = np.delete(x_crop1, np.s_[n_region:], 0)
    x_crop3 = np.delete(x_crop2, np.s_[0:n_edge], 1)
    x_crop = np.delete(x_crop3, np.s_[-n_edge:], 1)

    y_crop1 = np.delete(y_FOV1, np.s_[0:n_edge], 0)
    y_crop2 = np.delete(y_crop1, np.s_[n_region:], 0)
    y_crop3 = np.delete(y_crop2, np.s_[0:n_edge], 1)
    y_crop = np.delete(y_crop3, np.s_[-n_edge:], 1)

    Vx_crop1 = np.delete(Vx_FOV1, np.s_[0:n_edge], 0)
    Vx_crop2 = np.delete(Vx_crop1, np.s_[n_region:], 0)
    Vx_crop3 = np.delete(Vx_crop2, np.s_[0:n_edge], 1)
    Vx_crop = np.delete(Vx_crop3, np.s_[-n_edge:], 1)

    Vy_crop1 = np.delete(Vy_FOV1, np.s_[0:n_edge], 0)
    Vy_crop2 = np.delete(Vy_crop1, np.s_[n_region:], 0)
    Vy_crop3 = np.delete(Vy_crop2, np.s_[0:n_edge], 1)
    Vy_crop = np.delete(Vy_crop3, np.s_[-n_edge:], 1)

    Vz_crop1 = np.delete(Vz_FOV1, np.s_[0:n_edge], 0)
    Vz_crop2 = np.delete(Vz_crop1, np.s_[n_region:], 0)
    Vz_crop3 = np.delete(Vz_crop2, np.s_[0:n_edge], 1)
    Vz_crop = np.delete(Vz_crop3, np.s_[-n_edge:], 1)

    dVy = np.gradient(Vy_crop, axis=1)
    dx = np.gradient(x_crop, axis=1) / 1000
    dVx = np.gradient(Vx_crop, axis=0)
    dy = np.gradient(y_crop, axis=0) / 1000
    vorticity = (dVy / dx) - (dVx / dy)
    abs_vor = abs(vorticity)
    shape = np.shape(abs_vor)
    max_index = abs_vor.argmax()
    loc = np.unravel_index(max_index, abs_vor.shape)

    if loc[0] <= reference_y // 2:
        ref_z1 = 0
    elif n_region - loc[0] < reference_y // 2:
        ref_z1 = n_region - reference_y
    else:
        ref_z1 = loc[0] - reference_y // 2
    if loc[1] <= reference_x // 2:
        ref_x1 = 0
    elif shape[1] - loc[1] < reference_x // 2:
        ref_x1 = shape[1] - reference_x
    else:
        ref_x1 = loc[1] - reference_x // 2

    x_crop_c1 = np.delete(x_crop, np.s_[0:ref_z1], 0)
    x_crop_c2 = np.delete(x_crop_c1, np.s_[reference_y:], 0)
    x_crop_c3 = np.delete(x_crop_c2, np.s_[0:ref_x1], 1)
    x_c = np.delete(x_crop_c3, np.s_[reference_x:], 1)

    y_crop_c1 = np.delete(y_crop, np.s_[0:ref_z1], 0)
    y_crop_c2 = np.delete(y_crop_c1, np.s_[reference_y:], 0)
    y_crop_c3 = np.delete(y_crop_c2, np.s_[0:ref_x1], 1)
    y_c = np.delete(y_crop_c3, np.s_[reference_x:], 1)

    Vx_crop_c1 = np.delete(Vx_crop, np.s_[0:ref_z1], 0)
    Vx_crop_c2 = np.delete(Vx_crop_c1, np.s_[reference_y:], 0)
    Vx_crop_c3 = np.delete(Vx_crop_c2, np.s_[0:ref_x1], 1)
    Vx_c = np.delete(Vx_crop_c3, np.s_[reference_x:], 1)

    Vy_crop_c1 = np.delete(Vy_crop, np.s_[0:ref_z1], 0)
    Vy_crop_c2 = np.delete(Vy_crop_c1, np.s_[reference_y:], 0)
    Vy_crop_c3 = np.delete(Vy_crop_c2, np.s_[0:ref_x1], 1)
    Vy_c = np.delete(Vy_crop_c3, np.s_[reference_x:], 1)

    Vz_crop_c1 = np.delete(Vz_crop, np.s_[0:ref_z1], 0)
    Vz_crop_c2 = np.delete(Vz_crop_c1, np.s_[reference_y:], 0)
    Vz_crop_c3 = np.delete(Vz_crop_c2, np.s_[0:ref_x1], 1)
    Vz_c = np.delete(Vz_crop_c3, np.s_[reference_x:], 1)

    reference_region_y = y_c[:, 0]
    reference_region_x = x_c[0]
    reference_region_Vx = Vx_c
    reference_region_Vy = Vy_c
    reference_region_Vz = Vz_c

    data2_overlap_x = np.unique(data_FOV2[:, 0])
    data2_overlap_y = np.unique(data_FOV2[:, 1])

    data2_overlap_Vx = np.reshape(data_FOV2[:, 3], (len(data2_overlap_y), len(data2_overlap_x)))
    data2_overlap_Vy = np.reshape(data_FOV2[:, 4], (len(data2_overlap_y), len(data2_overlap_x)))
    data2_overlap_Vz = np.reshape(data_FOV2[:, 5], (len(data2_overlap_y), len(data2_overlap_x)))

    nmse, p, q,index = find_match_region_3d(n_edge,data2_overlap_Vx, reference_region_Vx,data2_overlap_Vz,reference_region_Vz,data2_overlap_Vy,reference_region_Vy,data2_overlap_x,reference_region_x,data2_overlap_y,reference_region_y,reference_x,reference_y)
    ref_x = [reference_region_x[0], reference_region_x[-1]]
    ref_y = [reference_region_y[0], reference_region_y[-1]]
    match_x = [data2_overlap_x[q], data2_overlap_x[q + reference_x - 1]]
    match_y = [data2_overlap_y[p], data2_overlap_y[p + reference_y - 1]]
    match = data2_overlap_x[q:q + reference_x]
    D_move = match_y[0] - ref_y[0]
    print(min(nmse))
    return D_move,nmse



