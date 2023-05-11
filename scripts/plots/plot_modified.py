import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import quaternion
import pandas as pd
from pose_transform import *
import trimesh

df_original = pd.read_csv('/home/chi-onn/fyp_ws/scoop_original.csv')
df_mod1 = pd.read_csv('/home/chi-onn/fyp_ws/scoop_mod1.csv')
df_mod2 = pd.read_csv('/home/chi-onn/fyp_ws/scoop_mod2.csv')
df_mod3 = pd.read_csv('/home/chi-onn/fyp_ws/scoop_mod3.csv')
df_mod4 = pd.read_csv('/home/chi-onn/fyp_ws/scoop_mod4.csv')
df_mod5 = pd.read_csv('/home/chi-onn/fyp_ws/scoop_mod5.csv')

x_ori = np.array(df_original.iloc[:,0])*100
y_ori = np.array(df_original.iloc[:,1])*100
z_ori = np.array(df_original.iloc[:,2])*100

x_mod1 = np.array(df_mod1.iloc[:,0])*100+2
y_mod1 = np.array(df_mod1.iloc[:,1])*100+5
z_mod1 = np.array(df_mod1.iloc[:,2])*100

x_mod2 = np.array(df_mod2.iloc[:,0])*100
y_mod2 = np.array(df_mod2.iloc[:,1])*100+9.5
z_mod2 = np.array(df_mod2.iloc[:,2])*100

x_mod3 = (np.array(df_mod3.iloc[:,0])*100)-5.5
y_mod3 = (np.array(df_mod3.iloc[:,1])*100)-1.5
z_mod3 = np.array(df_mod3.iloc[:,2])*100

x_mod4 = np.array(df_mod4.iloc[:,0])*100+2
y_mod4 = np.array(df_mod4.iloc[:,1])*100+2
z_mod4 = np.array(df_mod4.iloc[:,2])*100

x_mod5 = np.array(df_mod5.iloc[:,0])*100-4
y_mod5 = np.array(df_mod5.iloc[:,1])*100-4
z_mod5 = np.array(df_mod5.iloc[:,2])*100

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

#For mod
start_mod1 = df_mod1.iloc[0,:]
end_mod1 = df_mod1.iloc[39,:]

position_start_mod1 = start_mod1[0:3]
position_end_mod1 = end_mod1[0:3]

start_mod2 = df_mod2.iloc[0,:]
end_mod2 = df_mod2.iloc[39,:]

position_start_mod2 = start_mod2[0:3]
position_end_mod2 = end_mod2[0:3]

start_mod3 = df_mod3.iloc[0,:]
end_mod3 = df_mod3.iloc[39,:]

position_start_mod3 = start_mod3[0:3]
position_end_mod3 = end_mod3[0:3]

start_mod4 = df_mod4.iloc[0,:]
end_mod4 = df_mod4.iloc[39,:]

position_start_mod4 = start_mod4[0:3]
position_end_mod4 = end_mod4[0:3]

start_mod5 = df_mod5.iloc[0,:]
end_mod5 = df_mod5.iloc[39,:]

position_start_mod5 = start_mod5[0:3]
position_end_mod5 = end_mod5[0:3]





# PLOTTING
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection ='3d')
ax.plot3D(x_ori, y_ori, z_ori, 'k',linewidth = 3.0, label = 'Original trajectory')
ax.plot3D(x_mod1, y_mod1, z_mod1, 'b', linewidth = 3.0, label = 'Modified trajectory 1')
ax.plot3D(x_mod2, y_mod2, z_mod2, 'm' ,linewidth = 3.0, label = 'Modified trajectory 2')
ax.plot3D(x_mod3, y_mod3, z_mod3, 'g' ,linewidth = 3.0, label = 'Modified trajectory 3')
# ax.plot3D(x_mod5, y_mod5, z_mod5, 'b--')
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

ax.scatter(position_start_mod1[0]*100+2, position_start_mod1[1]*100+5, position_start_mod1[2]*100, marker='o', color='r')
ax.scatter(position_end_mod1[0]*100+2, position_end_mod1[1]*100+5, position_end_mod1[2]*100, marker='o', color='k')

ax.scatter(position_start_mod2[0]*100, position_start_mod2[1]*100+9.5, position_start_mod2[2]*100, marker='o', color='r')
ax.scatter(position_end_mod2[0]*100, position_end_mod2[1]*100+9.5, position_end_mod2[2]*100, marker='o', color='k')

ax.scatter(position_start_mod3[0]*100-5.5, position_start_mod3[1]*100-1.5, position_start_mod3[2]*100, marker='o', color='r')
ax.scatter(position_end_mod3[0]*100-5.5, position_end_mod3[1]*100-1.5, position_end_mod3[2]*100, marker='o', color='k')

# ax.scatter(position_start_mod4[0]*100, position_start_mod4[1]*100, position_start_mod4[2]*100, marker='o', color='r')
# ax.scatter(position_end_mod4[0]*100, position_end_mod4[1]*100, position_end_mod4[2]*100, marker='o', color='k')

# ax.scatter(position_start_mod5[0]*100-4, position_start_mod5[1]*100-4, position_start_mod5[2]*100, marker='o', color='r')
# ax.scatter(position_end_mod5[0]*100-4, position_end_mod5[1]*100-4, position_end_mod5[2]*100, marker='o', color='k')


#plot the bowl
your_mesh = trimesh.load('/home/chi-onn/fyp_ws/src/meshes/simplified_bowl.STL')

# Get the vertices and faces of the mesh
vertices = your_mesh.vertices
faces = your_mesh.faces
# ax.plot_trisurf((vertices[:, 0]*0.085)+45, (vertices[:, 1]*0.085)-23.5, (vertices[:, 2]*0.085)+27.5, triangles=faces, color = (0, 1, 0),alpha=0.1)
ax.plot_trisurf((vertices[:, 0]*0.06)+45, (vertices[:, 1]*0.06)-23.5, (vertices[:, 2]*0.06)+27.5, triangles=faces, color = (0, 1, 0),alpha=0.1)
#bowl_pos = [0.5093849131582481, -0.28866651753570133, 0.20004996849552092]

ax.set_title('Modified scooping trajectory produced by DMPs')

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel('z', rotation=0)
ax.set_box_aspect([1,1,1])
ax.view_init(elev=90, azim=0)

plt.show()