# Post_02_02_KeyRegionalStitchstereo

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from Post_01_02_RegionalStitch import y_axis_stitch_3d

pathFOV10 = 'C:\\Project\\Verticalmeasurement\\DT42'
pathFOV11 = 'C:\\Project\\Verticalmeasurement\\DT41'
pathFOV12 = 'C:\\Project\\Verticalmeasurement\\DT52'
pathFOV13 = 'C:\\Project\\Verticalmeasurement\\DT51'

pathup = [pathFOV10,pathFOV12]
pathbo = [pathFOV11,pathFOV13]

st = timer()
D_move=[]
n_mse=[]
n_edge = 5
overlap_region = 0.3
reference_y = 10
reference_x = 30

# FOV xyz
def Preprocess (path):
    read_FOV = pd.read_csv(path + '/B00001.dat', skiprows=2, sep=" ")
    data_FOV = np.array(read_FOV.iloc[:], dtype=float)
    x = -1 * data_FOV[:, 1]
    y = data_FOV[:, 0]
    z = np.zeros(len(x))
    Vx=[]
    Vy=[]
    Vz=[]
    for i in range(1,2001):
        fname = '/B0%04d' % i + '.dat'
        r_FOV = pd.read_csv(path+ fname, skiprows=2, sep=" ")
        FOV = np.array(r_FOV.iloc[:], dtype=float)
        Vx_FOV = -1 * FOV[:, 3]
        Vy_FOV = FOV[:, 2]
        Vz_FOV = FOV[:, 4]
        Vx.append(Vx_FOV)
        Vy.append(Vy_FOV)
        Vz.append(Vz_FOV)
    U=np.array(Vx)
    V=np.array(Vy)
    W=np.array(Vz)
    Vx_ave=np.average(U,axis=0)
    Vy_ave=np.average(V,axis=0)
    Vz_ave=np.average(W,axis=0)
    Data = np.transpose([x, y, z, Vx_ave,Vy_ave,Vz_ave])
    Data_FOV = Data[np.lexsort((Data[:, 0], Data[:, 1]))]
    return Data_FOV


up=[]
bo=[]
for n in range (2):
    Data_FOV_up=Preprocess(pathup[n])
    up.append(Data_FOV_up)
    Data_FOV_bo=Preprocess(pathbo[n])
    bo.append(Data_FOV_bo)

for m in range (2):
    Data_FOV1 = up[m]
    Data_FOV2 = bo[m]
    d_move,nmse = y_axis_stitch_3d (Data_FOV1,Data_FOV2,reference_x,reference_y,overlap_region,n_edge)
    D_move.append(d_move)
    n_mse.append(nmse)
print(D_move)
ed = timer()
print('running time %.8f' % (ed - st))
