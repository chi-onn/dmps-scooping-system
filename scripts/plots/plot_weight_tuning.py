import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import quaternion
import pandas as pd
from pose_transform import *

df_original = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original.csv')
df_deep = pd.read_csv('/home/chi-onn/fyp_ws/scoop_deep.csv')

x_ori = np.array(df_original.iloc[:,0])*100
y_ori = np.array(df_original.iloc[:,1])*100
z_ori = np.array(df_original.iloc[:,2])*100

x_deep = np.array(df_deep.iloc[:,0])*100
y_deep = np.array(df_deep.iloc[:,1])*100
z_deep = np.array(df_deep.iloc[:,2])*100



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
start_deep = df_deep.iloc[0,:]
end_deep = df_deep.iloc[39,:]

position_start_deep = start_deep[0:3]
orientation_start_deep = np.quaternion(start_deep[6],start_deep[3],start_deep[4],start_deep[5])

position_end_deep = end_deep[0:3]
orientation_end_deep = np.quaternion(end_deep[6],end_deep[3],end_deep[4],end_deep[5])


# PLOTTING
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection ='3d')
ax.plot3D(x_ori, y_ori, z_ori, 'k', linewidth = 2.0, label = 'Trajectory after weight tuning')
ax.plot3D(x_deep, y_deep, z_deep, 'r', linewidth = 2.0, label = 'Trajectory before weight tuning')

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

ax.scatter(position_start_deep[0]*100, position_start_deep[1]*100, position_start_deep[2]*100, marker='o', color='r')
ax.scatter(position_end_deep[0]*100, position_end_deep[1]*100, position_end_deep[2]*100, marker='o', color='k')




ax.set_title('Weight tuning of scooping trajectory')

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel('z', rotation=0)
ax.set_box_aspect([1,2,1])


#2d plot


fig, ax = plt.subplots()
ax.plot(y_ori,z_ori,'k',label = 'Trajectory after weight tuning')
ax.plot(y_deep,z_deep,'r--',label = 'Trajectory before weight tuning')

ax.scatter(y_ori[0], z_ori[0], marker='o', color='r',label='Starting pose')
ax.scatter(y_ori[39], z_ori[39], marker='o', color='k',label = 'Goal pose')


ax.scatter(y_deep[0], z_deep[0], marker='o', color='r')
ax.scatter(y_deep[39], z_deep[39], marker='o', color='k')


ax.set_title('Weight tuning of scooping trajectory')
plt.legend()
plt.xlabel("y")
plt.ylabel("z")
ax.invert_xaxis()
ax.yaxis.label.set_rotation(0)
plt.grid()

plt.show()