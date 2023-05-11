import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import quaternion
import pandas as pd
from pose_transform import *

df_original = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original.csv')
df_short = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original_short.csv')
df_long = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original_long.csv')

x_ori = np.array(df_original.iloc[:,0])*100
y_ori = np.array(df_original.iloc[:,1])*100
z_ori = np.array(df_original.iloc[:,2])*100

x_short = np.array(df_short.iloc[:,0])*100
y_short = np.array(df_short.iloc[:,1])*100
z_short = np.array(df_short.iloc[:,2])*100

x_long = np.array(df_long.iloc[:,0])*100
y_long = np.array(df_long.iloc[:,1])*100
z_long = np.array(df_long.iloc[:,2])*100



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

# For long and short
start_short = df_short.iloc[0,:]
end_short = df_short.iloc[39,:]

position_start_short = start_short[0:3]
orientation_start_short = np.quaternion(start_short[6],start_short[3],start_short[4],start_short[5])

position_end_short = end_short[0:3]
orientation_end_short = np.quaternion(end_short[6],end_short[3],end_short[4],end_short[5])

start_long = df_long.iloc[0,:]
end_long = df_long.iloc[39,:]

position_start_long = start_long[0:3]
orientation_start_long = np.quaternion(start_long[6],start_long[3],start_long[4],start_long[5])

position_end_long = end_long[0:3]
orientation_end_long = np.quaternion(end_long[6],end_long[3],end_long[4],end_long[5])

# PLOTTING
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection ='3d')
ax.plot3D(x_ori, y_ori, z_ori, 'k',label = 'Original trajectory')
ax.plot3D(x_short, y_short, z_short, 'r--',label = 'Shorter trajectory')
ax.plot3D(x_long, y_long, z_long, 'b--',label='Longer trajectory')
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

ax.scatter(position_start_short[0]*100, position_start_short[1]*100, position_start_short[2]*100, marker='o', color='r')
ax.scatter(position_end_short[0]*100, position_end_short[1]*100, position_end_short[2]*100, marker='o', color='k')

ax.scatter(position_start_long[0]*100, position_start_long[1]*100, position_start_long[2]*100, marker='o', color='r')
ax.scatter(position_end_long[0]*100, position_end_long[1]*100, position_end_long[2]*100, marker='o', color='k')


ax.set_title('Length-varying scooping trajectory produced by DMPs')

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel('z', rotation=0)
ax.set_box_aspect([1,2,1])


#2d plot


fig, ax = plt.subplots()
ax.plot(y_ori,z_ori,'k',label = 'Original trajectory')
ax.plot(y_short,z_short,'r--',label = 'Shorter trajectory')
ax.plot(y_long,z_long,'b--',label = 'Longer trajectory')

ax.scatter(y_ori[0], z_ori[0], marker='o', color='r',label='Starting pose')
ax.scatter(y_ori[39], z_ori[39], marker='o', color='k',label = 'Goal pose')


ax.scatter(y_short[0], z_short[0], marker='o', color='r')
ax.scatter(y_short[39], z_short[39], marker='o', color='k')

ax.scatter(y_long[0], z_long[0], marker='o', color='r')
ax.scatter(y_long[39], z_long[39], marker='o', color='k')

ax.set_title('Length-varying scooping trajectory produced by DMPs')
plt.legend()
plt.xlabel("y")
plt.ylabel("z")
ax.invert_xaxis()
ax.yaxis.label.set_rotation(0)
plt.grid()

plt.show()