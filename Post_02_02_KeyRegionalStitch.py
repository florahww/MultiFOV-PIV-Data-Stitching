# 01_PreStitch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from Post_01_02_RegionalStitch import z_axis_stitch_2d,x_axis_stitch_2d


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"})

pathFOV1 = 'C:\\Project\\Verticalmeasurement\\DT101'
pathFOV2 = 'C:\\Project\\Verticalmeasurement\\DT102'
pathFOV3 = 'C:\\Project\\Verticalmeasurement\\DT103'
pathFOV4 = 'C:\\Project\\Verticalmeasurement\\DT104'
pathFOV5 = 'C:\\Project\\Verticalmeasurement\\DT105'
pathFOV6 = 'C:\\Project\\Verticalmeasurement\\DT106'
pathFOV7 = 'C:\\Project\\Verticalmeasurement\\DT107'
pathFOV8 = 'C:\\Project\\Verticalmeasurement\\DT108'
pathFOV9 = 'C:\\Project\\Verticalmeasurement\\DT109'

pathup=[pathFOV2,pathFOV3,pathFOV4,pathFOV5]
pathbo=[pathFOV6,pathFOV7,pathFOV8,pathFOV9]

# FOV xyz
def Preprocess (n_path):
    read_FOV = pd.read_csv(n_path + '/B00001.dat', skiprows=2, sep=" ")
    data_FOV = np.array(read_FOV.iloc[:], dtype=float)
    x = -1 * data_FOV[:, 1]
    y = np.zeros(len(x))
    z = data_FOV[:, 0]
    Vx=[]
    Vz=[]
    for i in range(1,2001):
        fname = '/B0%04d' % i + '.dat'
        r_FOV = pd.read_csv(n_path + fname, skiprows=2, sep=" ")
        FOV = np.array(r_FOV.iloc[:], dtype=float)
        Vx_FOV = -1 * FOV[:, 3]
        Vz_FOV = FOV[:, 2]
        Vx.append(Vx_FOV)
        Vz.append(Vz_FOV)
    U=np.array(Vx)
    W=np.array(Vz)
    Vx_ave=np.average(U,axis=0)
    Vz_ave=np.average(W,axis=0)
    Data = np.transpose([x, y, z, Vx_ave,Vz_ave])
    Data_FOV = Data[np.lexsort((Data[:, 0], Data[:, 2]))]
    return Data_FOV
up=[]
bo=[]
for n in range (4):
    Data_FOV_up=Preprocess(pathup[n])
    up.append(Data_FOV_up)
    Data_FOV_bo=Preprocess(pathbo[n])
    bo.append(Data_FOV_bo)

st = timer()
D_move=[]
n_mse=[]
n_edge = 5
overlap_region = 0.05
reference_z = 10
reference_x = 30
for m in range (4):
    Data_FOV1 = up[m]
    Data_FOV2 = bo[m]
    d_move,nmse = z_axis_stitch_2d(Data_FOV1, Data_FOV2, reference_x,reference_z,overlap_region,n_edge)
    D_move.append(d_move)
    n_mse.append(nmse)
print(D_move)
ed = timer()
print('running time %.8f' % (ed - st))




