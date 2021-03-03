# Post_03_BoundarySettings

import pandas as pd
import numpy as np
from Post_02_02_KeyRegionalStitch import D_move

pathFOV1 = 'C:\\Project\\Verticalmeasurement\\DT101'
pathFOV2 = 'C:\\Project\\Verticalmeasurement\\DT102'
pathFOV3 = 'C:\\Project\\Verticalmeasurement\\DT103'
pathFOV4 = 'C:\\Project\\Verticalmeasurement\\DT104'
pathFOV5 = 'C:\\Project\\Verticalmeasurement\\DT105'
pathFOV6 = 'C:\\Project\\Verticalmeasurement\\DT106'
pathFOV7 = 'C:\\Project\\Verticalmeasurement\\DT107'
pathFOV8 = 'C:\\Project\\Verticalmeasurement\\DT108'
pathFOV9 = 'C:\\Project\\Verticalmeasurement\\DT109'

SavePath  = 'C:\Project\Stitching\stitch'

PathFOV = [pathFOV1, pathFOV2, pathFOV3, pathFOV4, pathFOV5, pathFOV6, pathFOV7, pathFOV8, pathFOV9]

# Building model shape
shape_B1 = [133, 480, 128]
shape_B2 = [188, 480, 173]
shape_B3 = [188, 480, 436]
shape_C = [188, 198, 0]
shape = np.array([shape_B1, shape_B2, shape_B3, shape_C])
move=np.array(D_move)
pathFOV1 = PathFOV[0]
FOV1 = pd.read_csv(pathFOV1 + '/B00001.dat', skiprows=2, sep=" ")
data_FOV1 = np.array(FOV1.iloc[:], dtype=float)
x_data_FOV1 = data_FOV1[:, 0]
z_data_FOV1 = data_FOV1[:, 1]

# Move Distance
L1x_vertical = x_data_FOV1[0]
L2x_vertical = x_data_FOV1[- 1]
L2z_vertical = z_data_FOV1[0]
L1z_vertical = z_data_FOV1[- 1]

x_verFOV5 = shape[0,0] + L1z_vertical
x_verFOV9 = x_verFOV5
x_verFOV4 = x_verFOV5 + shape[3,0]
x_verFOV8 = x_verFOV4
x_verFOV3 = x_verFOV4 + shape[1,0]
x_verFOV7 = x_verFOV3
x_verFOV2 = x_verFOV3 + shape[3,1]
x_verFOV6 = x_verFOV2
x_verFOV1 = x_verFOV2 + shape[2,0]
y_verFOV = 0.5 * shape[0, 1]
z_verFOV1_5 = move - L1x_vertical
z_verFOV6_9 = -1 * L1x_vertical
FOV_x_Move = np.hstack(
    (x_verFOV1, x_verFOV2, x_verFOV3, x_verFOV4, x_verFOV5, x_verFOV6, x_verFOV7, x_verFOV8, x_verFOV9))
FOV_y_Move = np.ones(9) * y_verFOV
FOV_z_Move = np.hstack((350-L1x_vertical, z_verFOV1_5, np.ones(4) * z_verFOV6_9))

Move = np.transpose([FOV_x_Move, FOV_y_Move, FOV_z_Move])

print('MOVE',Move)

# Boundary
Mid_z = 350 + ((L2x_vertical - L1x_vertical - 350) / 2)
z_bottomFOV2_5 = Mid_z+5
z_bottomFOV1 = shape[2, 2]
z_bottomFOV9 = shape[1, 2]
z_topFOV1_5 = 710
z_topFOV6_9 = Mid_z

x_leftFOV1 = shape[0, 0] + shape[3, 0] + shape[1, 0] + shape[3, 1]+2
x_rightFOV2 = x_leftFOV1-5
x_leftFOV2 = shape[0, 0] + shape[3, 0] + shape[1, 0]+2
x_rightFOV3 = x_leftFOV2-5
x_leftFOV3 = shape[0, 0] + shape[3, 0]+2
x_rightFOV4 = x_leftFOV3-5
x_leftFOV4 = shape[0, 0]+2
x_rightFOV5 = x_leftFOV4-5
x_rightFOV6 = x_rightFOV2
x_leftFOV6 = x_leftFOV2
x_rightFOV7 = x_rightFOV3
x_leftFOV7 = x_leftFOV3
z_bottomFOV7 = shape[1, 2]
x_rightFOV8 = x_rightFOV4
x_leftFOV8 = x_leftFOV4
x_rightFOV9 = x_rightFOV5

FOV_top = np.hstack((np.ones(5) * z_topFOV1_5, np.ones(4) *z_topFOV6_9))
FOV_bottom = np.hstack((z_bottomFOV1, np.ones(4) *z_bottomFOV2_5, 0, z_bottomFOV7, 0, z_bottomFOV9))
FOV_left = np.hstack((x_leftFOV1, x_leftFOV2, x_leftFOV3, x_leftFOV4, 0, x_leftFOV6, x_leftFOV7, x_leftFOV8, 0))
FOV_right = np.hstack(
    (895, x_rightFOV2, x_rightFOV3, x_rightFOV4, x_rightFOV5, x_rightFOV6, x_rightFOV7, x_rightFOV8, x_rightFOV9))

Boundary = np.transpose([FOV_top, FOV_bottom, FOV_left, FOV_right])






