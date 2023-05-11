import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import quaternion
import pandas as pd
from pose_transform import *

df_original = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original.csv')
df_2x = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original_+0.2x.csv')
# df_2x_2y = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original_+0.2x_+0.2y.csv')
# df_2y = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original_+0.2y.csv')
# df_neg2x_2y = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original_-0.2x_+0.2y.csv')
# df_neg2x = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original_-0.2x.csv')

x_ori = np.array(df_original.iloc[:,0])*100
y_ori = np.array(df_original.iloc[:,1])*100
z_ori = np.array(df_original.iloc[:,2])*100

x_2x = np.array(df_2x.iloc[:,0])*100
y_2x = np.array(df_2x.iloc[:,1])*100
z_2x = np.array(df_2x.iloc[:,2])*100

# x_2x_2y = np.array(df_2x_2y.iloc[:,0])
# y_2x_2y = np.array(df_2x_2y.iloc[:,1])
# z_2x_2y = np.array(df_2x_2y.iloc[:,2])

# x_2y = np.array(df_2y.iloc[:,0])
# y_2y = np.array(df_2y.iloc[:,1])
# z_2y = np.array(df_2y.iloc[:,2])

# x_neg2x_2y = np.array(df_neg2x_2y.iloc[:,0])
# y_neg2x_2y = np.array(df_neg2x_2y.iloc[:,1])
# z_neg2x_2y = np.array(df_neg2x_2y.iloc[:,2])

# x_neg2x_2y = np.array(df_neg2x_2y.iloc[:,0])
# y_neg2x_2y = np.array(df_neg2x_2y.iloc[:,1])
# z_neg2x_2y = np.array(df_neg2x_2y.iloc[:,2])

# x_neg2x = np.array(df_neg2x.iloc[:,0])
# y_neg2x = np.array(df_neg2x.iloc[:,1])
# z_neg2x = np.array(df_neg2x.iloc[:,2])


# FOR ORIENTATION PLOTTING OF START AND END POSITION
start = df_original.iloc[0,:]
end = df_original.iloc[39,:]

position_start = start[0:3]
orientation_start = np.quaternion(start[6],start[3],start[4],start[5])

position_end = end[0:3]
orientation_end = np.quaternion(end[6],end[3],end[4],end[5])

# Convert the quaternion orientation to a rotation matrix
orientation_matrix_start = quaternion.as_rotation_matrix(orientation_start)
orientation_matrix_end = quaternion.as_rotation_matrix(orientation_end)

# FOR 2X
start_2x = df_2x.iloc[0,:]
end_2x = df_2x.iloc[39,:]

position_start_2x = start_2x[0:3]
orientation_start_2x = np.quaternion(start_2x[6],start_2x[3],start_2x[4],start_2x[5])

position_end_2x = end_2x[0:3]
orientation_end_2x = np.quaternion(end_2x[6],end_2x[3],end_2x[4],end_2x[5])

orientation_matrix_start_2x = quaternion.as_rotation_matrix(orientation_start_2x)
orientation_matrix_end_2x = quaternion.as_rotation_matrix(orientation_end_2x)


# PLOTTING
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection ='3d')
ax.plot3D(x_ori, y_ori, z_ori, 'k',label = 'Original trajectory')
ax.plot3D(x_2x, y_2x, z_2x, 'b--',label = 'Offset trajectory')
# ax.plot3D(x_2x_2y, y_2x_2y, z_2x_2y, 'b--')
# ax.plot3D(x_2y, y_2y, z_2y, 'b--')
# ax.plot3D(x_neg2x_2y, y_neg2x_2y, z_neg2x_2y, 'b--')
# ax.plot3D(x_neg2x, y_neg2x, z_neg2x, 'b--')

colors = ['r', 'g', 'b']

# # Loop over the three axes and plot the corresponding arrow with a different color
# for i in range(3):
#     ax.quiver(position_start[0], position_start[1], position_start[2],
#               orientation_matrix_start[0][i], orientation_matrix_start[1][i], orientation_matrix_start[2][i],
#               length=0.01, color=colors[i])
    
# for i in range(3):
#     ax.quiver(position_end[0], position_end[1], position_end[2],
#               orientation_matrix_end[0][i], orientation_matrix_end[1][i], orientation_matrix_end[2][i],
#               length=0.01, color=colors[i])
    
# for i in range(3):
#     ax.quiver(position_start_2x[0], position_start_2x[1], position_start_2x[2],
#               orientation_matrix_start_2x[0][i], orientation_matrix_start_2x[1][i], orientation_matrix_start_2x[2][i],
#               length=0.01, color=colors[i])
    
# for i in range(3):
#     ax.quiver(position_end_2x[0], position_end_2x[1], position_end_2x[2],
#               orientation_matrix_end_2x[0][i], orientation_matrix_end_2x[1][i], orientation_matrix_end_2x[2][i],
#               length=0.01, color=colors[i])


ax.scatter(position_start[0]*100, position_start[1]*100, position_start[2]*100, marker='o', color='r',label='Starting pose')
ax.scatter(position_end[0]*100, position_end[1]*100, position_end[2]*100, marker='o', color='k',label = 'Goal pose')

ax.scatter(position_start_2x[0]*100, position_start_2x[1]*100, position_start_2x[2]*100, marker='o', color='r')
ax.scatter(position_end_2x[0]*100, position_end_2x[1]*100, position_end_2x[2]*100, marker='o', color='k')
ax.set_title('Offset scooping trajectory produced by DMPs')

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel('z', rotation=0)
ax.set_box_aspect([1,1,1])

plt.show()