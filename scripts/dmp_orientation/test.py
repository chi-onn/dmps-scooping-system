# """
# ====================
# Cartesian DMP on UR5
# ====================

# A trajectory is created manually, imitated with a Cartesian DMP, converted
# to a joint trajectory by inverse kinematics, and executed with a UR5.
# """
# print(__doc__)

import numpy as np
# import pytransform3d.visualizer as pv
# import pytransform3d.rotations as pr
# import pytransform3d.transformations as pt
# import pytransform3d.trajectories as ptr
# from movement_primitives.kinematics import Kinematics
from movement_primitives.dmp import CartesianDMP
import pandas as pd

#rotation_angle = np.deg2rad(45)
df = pd.read_csv('/home/chi-onn/fyp_ws/src/scooping/traj_files/scooping_bowl_left_pose.csv')

df_np = np.array(df)
n_steps = df_np.shape[0]

T = np.linspace(0, 1, n_steps)

dt = 0.05
#execution_time = (n_steps - 1) * dt
execution_time = 2.0
T = np.linspace(0, execution_time, n_steps)
print(T.shape)
Y = df_np
print(Y.shape)

dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=10)
dmp.imitate(T, Y)
_, Y = dmp.open_loop()

weights = dmp.get_weights
print(weights)

