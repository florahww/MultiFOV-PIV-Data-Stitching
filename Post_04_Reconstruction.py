# 04_Reconstruction

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from Post_03_BoundarySettings import PathFOV, SavePath, Move, Boundary

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def Main(i):
    # FOV VxVz
    fname = '/B0%04d' % i + '.dat'
    d_list=np.ones(5)
    for n in range(9):
        FOV1 = pd.read_csv(PathFOV[n] + fname, skiprows=2, sep=" ")
        data_FOV1 = np.array(FOV1.iloc[:], dtype=float)

        data_FOV1_x = np.unique(Move[n, 0] - 1 * data_FOV1[:, 1])
        data_FOV1_z = np.unique(Move[n, 2] + data_FOV1[:, 0])
        data_FOV1_Vx = -1 * data_FOV1[:, 3]
        data_FOV1_Vz = data_FOV1[:, 2]

        Vx_Mat = np.reshape(data_FOV1_Vx, (len(data_FOV1_x), len(data_FOV1_z)))
        Vx_Matrix = np.transpose(Vx_Mat)
        Vz_Mat = np.reshape(data_FOV1_Vz, (len(data_FOV1_x), len(data_FOV1_z)))
        Vz_Matrix = np.transpose(Vz_Mat)
        x_Mat = np.reshape(Move[n, 0] - 1 * data_FOV1[:, 1], (len(data_FOV1_x), len(data_FOV1_z)))
        x_Matrix = np.transpose(x_Mat)
        z_Mat = np.reshape(Move[n, 2] + data_FOV1[:, 0], (len(data_FOV1_x), len(data_FOV1_z)))
        z_Matrix = np.transpose(z_Mat)


        Z_top = find_nearest(data_FOV1_z, Boundary[n, 0])
        Z_bottom = find_nearest(data_FOV1_z, Boundary[n, 1])
        X_left = find_nearest(data_FOV1_x, Boundary[n, 2])
        X_right = find_nearest(data_FOV1_x, Boundary[n, 3])

        Vx_M = Vx_Matrix[Z_bottom:Z_top+1, X_left:X_right+1]
        Vz_M = Vz_Matrix[Z_bottom:Z_top+1, X_left:X_right+1]
        x_M = x_Matrix[Z_bottom:Z_top+1, X_left:X_right+1]
        z_M = z_Matrix[Z_bottom:Z_top+1, X_left:X_right+1]

        Vx_Flat = Vx_M.flatten()
        Vz_Flat = Vz_M.flatten()

        x_Flat = x_M.flatten()
        z_Flat = z_M.flatten()
        y_Flat = np.ones(len(z_Flat)) * Move[n, 1]
        print(x_Flat.shape)
        data = np.stack((x_Flat, y_Flat, z_Flat, Vx_Flat, Vz_Flat))
        print('data',data.shape)
        data_t=np.transpose(data)
        d_list=np.vstack((d_list,data_t))

    Data = d_list[1:,:]
    print('Data',Data.shape)
    Data_sort = Data[np.lexsort((Data[:, 0], Data[:, 2]))]

    I = len(np.unique(Data_sort[:, 0]))
    K = len(np.unique(Data_sort[:, 2]))

    header = 'TITLE = "B0%04d"' % i + '\n'
    header += 'VARIABLES = "x", "y", "z", "Vx", "Vz"\n'
    header += 'ZONE T="Frame 0", I=' + str(I) + ', J=1, K=' + str(K) + ', F=POINT'
    return np.savetxt(SavePath + fname, Data_sort, header=header, fmt='%.4f', comments='')


start = timer()
save = Parallel(n_jobs=8)(delayed(Main)(i) for i in range(1, 2001))
end = timer()
print('running time is %.8f' % (end - start))