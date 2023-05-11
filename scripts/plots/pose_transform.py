import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import quaternion
import pandas as pd


def rotate_pose(pose, angle):
    position_start = pose[0:3]
    orientation_start = np.quaternion(pose[6],pose[3],pose[4],pose[5])
    orientation_matrix_start = quaternion.as_rotation_matrix(orientation_start)
    theta = np.deg2rad(angle)
    rot_z = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0]])  # Rotate only about the Z-axis
    new_orientation_matrix = np.matmul(rot_z, orientation_matrix_start)
    new_quaternion = quaternion.from_rotation_matrix(new_orientation_matrix)
    w, x, y, z = new_quaternion.real, new_quaternion.imag[0], new_quaternion.imag[1], new_quaternion.imag[2]
    quat_array = np.array([x, y, z, w])
    pose[3:7]=quat_array
    return pose

def eef_to_spoon_traj(eef_scoop_traj_df):

    ee_traj = eef_scoop_traj_df.to_numpy()
    # compute trajectory of spoon from trajectory of end-effector
    spoon_offset = np.array([0.14,0.0,0.0])
    spoon_traj_normal = np.zeros_like(ee_traj)
    spoon_traj_moveit = np.zeros_like(ee_traj) # x, y, z, w
    for i in range(len(ee_traj)):
        ee_pose = ee_traj[i,:]

        # Extract rotation matrix and position from end-effector pose
        ee_quat = np.quaternion(ee_pose[6],ee_pose[3],ee_pose[4],ee_pose[5])
        ee_rot_mat = quaternion.as_rotation_matrix(ee_quat)
        ee_pos = ee_pose[:3]

        # Compute spoon position and orientation
        spoon_offset_world = ee_rot_mat.dot(spoon_offset)
        spoon_pos = ee_pos + spoon_offset_world
        spoon_quat = ee_quat

        spoon_traj_normal[i, :] = np.hstack((spoon_pos, spoon_quat.components)) # may need to change quaternion part to x,y,z,w
        spoon_traj_moveit[i, :] = np.array([spoon_pos[0], spoon_pos[1], spoon_pos[2], ee_quat.imag[0], ee_quat.imag[1], ee_quat.imag[2], ee_quat.real])

    return spoon_traj_moveit

def eef_to_spoon_pose(eef_pose):

    # compute trajectory of spoon from trajectory of end-effector
    spoon_offset = np.array([0.14,0.0,0.0])

    # Extract rotation matrix and position from end-effector pose, need to be w,x,y,z
    ee_quat = np.quaternion(eef_pose[6],eef_pose[3],eef_pose[4],eef_pose[5])
    ee_rot_mat = quaternion.as_rotation_matrix(ee_quat)
    ee_pos = eef_pose[:3]

    # Compute spoon position and orientation
    spoon_offset_world = ee_rot_mat.dot(spoon_offset)
    spoon_pos = ee_pos + spoon_offset_world
    spoon_quat = ee_quat

    spoon_pose_normal = np.hstack((spoon_pos, spoon_quat.components)) # may need to change quaternion part to x,y,z,w
    spoon_pose_moveit = np.array([spoon_pos[0], spoon_pos[1], spoon_pos[2], spoon_quat.imag[0], spoon_quat.imag[1], spoon_quat.imag[2], spoon_quat.real])

    return spoon_pose_moveit

def spoon_to_eef_traj(spoon_traj_arr):
    # # get back end-effector trajectory from spoon trajectory
    ee_traj_normal = np.zeros_like(spoon_traj_arr)
    ee_traj_moveit = np.zeros_like(spoon_traj_arr)
    spoon_offset = np.array([0.14,0.0,0.0])

    for i in range(len(spoon_traj_arr)):
        spoon_pose = spoon_traj_arr[i,:]

        # Extract rotation matrix and position from spoon pose
        spoon_quat = np.quaternion(spoon_pose[6], spoon_pose[3], spoon_pose[4], spoon_pose[5])
        spoon_rot_mat = quaternion.as_rotation_matrix(spoon_quat)
        spoon_pos = spoon_pose[:3]

        # Compute end-effector position and orientation
        ee_offset_world = spoon_rot_mat.dot(spoon_offset)
        ee_pos = spoon_pos - ee_offset_world
        ee_quat = spoon_quat

        ee_traj_normal[i,:] = np.hstack((ee_pos, ee_quat.components))
        ee_traj_moveit[i,:] = np.array([ee_pos[0], ee_pos[1], ee_pos[2], ee_quat.imag[0], ee_quat.imag[1], ee_quat.imag[2], ee_quat.real])

    return ee_traj_moveit

def spoon_to_eef_pose(spoon_pose):

    # compute trajectory of spoon from trajectory of end-effector
    spoon_offset = np.array([0.14,0.0,0.0])

    # Extract rotation matrix and position from end-effector pose, need to be w,x,y,z
    spoon_quat = np.quaternion(spoon_pose[6],spoon_pose[3],spoon_pose[4],spoon_pose[5])
    spoon_rot_mat = quaternion.as_rotation_matrix(spoon_quat)
    spoon_pos = spoon_pose[:3]

    # Compute spoon position and orientation
    spoon_offset_world = spoon_rot_mat.dot(spoon_offset)
    ee_pos = spoon_pos - spoon_offset_world
    ee_quat = spoon_quat

    ee_pose_normal = np.hstack((ee_pos, ee_quat.components)) # may need to change quaternion part to x,y,z,w
    ee_pose_moveit = np.array([ee_pos[0], ee_pos[1], ee_pos[2], ee_quat.imag[0], ee_quat.imag[1], ee_quat.imag[2], ee_quat.real])

    return ee_pose_moveit

def change_scoop_length(start,end,length):
    x_start = start[0]
    x_end = end[0]
    x_diff = np.abs(x_start-x_end)

    y_start = start[1]
    y_end = end[1]
    y_diff = np.abs(y_start-y_end)
    scaling_factor_long = 0.02
    scaling_factor_short = 0.02
    max_length = 0.1

    if length == 'long':
        if x_start > x_end:
            x_start += (x_diff/max_length)*(scaling_factor_long)
            x_end -= (x_diff/max_length)*(scaling_factor_long)
        else:
            x_start -= (x_diff/max_length)*(scaling_factor_long)
            x_end += (x_diff/max_length)*(scaling_factor_long)
        if y_start > y_end:
            y_start += (y_diff/max_length)*(scaling_factor_long)
            y_end -= (y_diff/max_length)*(scaling_factor_long)
        else:
            y_start -= (y_diff/max_length)*(scaling_factor_long)
            y_end += (y_diff/max_length)*(scaling_factor_long)
    elif length == 'short':
        if x_start > x_end:
            x_start -= (x_diff/max_length)*(scaling_factor_short)
            x_end += (x_diff/max_length)*(scaling_factor_short)
        else:
            x_start += (x_diff/max_length)*(scaling_factor_short)
            x_end -= (x_diff/max_length)*(scaling_factor_short)
        if y_start > y_end:
            y_start -= (y_diff/max_length)*(scaling_factor_short)
            y_end += (y_diff/max_length)*(scaling_factor_short)
        else:
            y_start += (y_diff/max_length)*(scaling_factor_short)
            y_end -= (y_diff/max_length)*(scaling_factor_short)

    start_new = start
    start_new[0], start_new[1] = x_start, y_start
    end_new = end
    end_new[0], end_new[1] = x_end, y_end
    return start_new, end_new



# df = pd.read_csv('/home/chi-onn/fyp_ws/src/scooping/traj_files/scooping_bowl_left_pose.csv')
# spoon_traj = eef_to_spoon_traj(df)
# eef_traj = spoon_to_eef_traj(spoon_traj)

# spoon_pose = eef_to_spoon_pose(eef_traj[0])

# print(spoon_pose)
# print(spoon_traj[0])


# x_ori = np.array(df.iloc[:,0])
# y_ori = np.array(df.iloc[:,1])
# z_ori = np.array(df.iloc[:,2])

# spoon_traj_df = pd.DataFrame(spoon_traj)
# x_spoon = np.array(spoon_traj_df.iloc[:,0])
# y_spoon = np.array(spoon_traj_df.iloc[:,1])
# z_spoon = np.array(spoon_traj_df.iloc[:,2])

# ee_traj_df = pd.DataFrame(eef_traj)
# x_new = np.array(ee_traj_df.iloc[:,0])
# y_new = np.array(ee_traj_df.iloc[:,1])
# z_new = np.array(ee_traj_df.iloc[:,2])

# start = df.iloc[0,:]
# end = df.iloc[3000,:]

# position_start = start[0:3]
# orientation_start = np.quaternion(start[6],start[3],start[4],start[5])

# position_end = end[0:3]
# orientation_end = np.quaternion(end[6],end[3],end[4],end[5])

# # Convert the quaternion orientation to a rotation matrix
# orientation_matrix_start = quaternion.as_rotation_matrix(orientation_start)
# orientation_matrix_end = quaternion.as_rotation_matrix(orientation_end)



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax = plt.axes(projection ='3d')
# ax.plot3D(x_ori, y_ori, z_ori, 'b',label = 'Ee trajectory')
# ax.plot3D(x_spoon, y_spoon, z_spoon, 'b--',label = 'Spoon trajectory')
# ax.plot3D(x_new, y_new, z_new, 'r',label = 'Ee_reconverted trajectory')

# colors = ['r', 'g', 'b']

# # Loop over the three axes and plot the corresponding arrow with a different color
# for i in range(3):
#     ax.quiver(position_start[0], position_start[1], position_start[2],
#               orientation_matrix_start[0][i], orientation_matrix_start[1][i], orientation_matrix_start[2][i],
#               length=0.14, color=colors[i])
    
# for i in range(3):
#     ax.quiver(position_end[0], position_end[1], position_end[2],
#               orientation_matrix_end[0][i], orientation_matrix_end[1][i], orientation_matrix_end[2][i],
#               length=0.14, color=colors[i])

# plt.show()



##########################################################################################################
# LENGTH VARIATION
############# for original traj #############
# longer
# dmp.y0   += np.array([0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
# dmp.goal += np.array([-0.02, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0])

# shorter
# dmp.y0   += np.array([-0.02, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
# dmp.goal += np.array([0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])

############# for modified 3 (12-6)#############
# # longer
# dmp.y0   += np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# dmp.goal += np.array([-0.02, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# shorter
# dmp.y0   += np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# dmp.goal += np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

############# for modified 5 (6-12)#############
# # longer
# dmp.y0   += np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# dmp.goal += np.array([0.02, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# shorter
# dmp.y0   += np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# dmp.goal += np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])





