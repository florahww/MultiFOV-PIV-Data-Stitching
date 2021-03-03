# Post_05_AverageCalculation

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from joblib import Parallel, delayed
from Post_03_BoundarySettings import SavePath


def Main(i):
    # FOV VxVz
    fname = '/B0%04d' % i + '.dat'
    read_data = pd.read_csv(SavePath + fname, skiprows=2, sep=" ")
    d = np.array(read_data.iloc[:], dtype=float)
    Vx = d[:, 3]
    Vz = d[:, 4]
    return Vx, Vz

start = timer()
f1name = '/B00001.dat'
data1 = pd.read_csv(SavePath + f1name, skiprows=2, sep=" ")
d1 = np.array(data1.iloc[:], dtype=float)
Vx, Vz = zip(*Parallel(n_jobs=8)(delayed(Main)(i) for i in range(1, 2001)))
Vx_all = np.asarray(Vx)
Vz_all = np.asarray(Vz)
x = d1[:, 0]
y = d1[:, 1]
z = d1[:, 2]
I = len(np.unique(x))
K = len(np.unique(z))
Average_Vx = np.sum(Vx_all, axis=0) / 2000
Average_Vz = np.sum(Vz_all, axis=0) / 2000
Data = np.transpose([x, y, z, Average_Vx, Average_Vz])
header = 'TITLE = "DT_average"\n'
header += 'VARIABLES = "x", "y", "z", "Vx_average", "Vz_average"\n'
header += 'ZONE T="Frame 0", I=' + str(I) + ', J=1, K=' + str(K) + ', F=POINT'
np.savetxt(SavePath + '/DT_Average.dat', Data, header=header, fmt='%.4f', comments='')
