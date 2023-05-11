import pandas as pd
import numpy as np
from get_fk_client import *
import csv
import os.path

if __name__ == '__main__':
    rospy.init_node('test_fk')
    gfk = GetFK('link_tcp', 'world')
    rospy.sleep(3)


    # original
    df = pd.read_csv('/home/chi-onn/fyp_ws/src/scooping/traj_files/scooping_bowl_left.traj',header=None)
    df = df.iloc[0:df.shape[0],0:6] # for draw_2.traj (before dmp), removed 7th column (gripper)

    print(df)

    save_path = '/home/chi-onn/fyp_ws/src/scooping/traj_files'
    file_name = 'scooping_bowl_left_pose.csv'
    completeName = os.path.join(save_path,file_name)
    f = open(completeName, 'w')
    writer = csv.writer(f)
    for row in range(df.shape[0]):
        joint_angles = df.iloc[row,:]
        joint_angles[6],joint_angles[7],joint_angles[8],joint_angles[9],joint_angles[10],joint_angles[11]=0,0,0,0,0,0
        resp = gfk.get_fk_list(joint_angles)
        posestamped = resp.pose_stamped[0]
        robot_pose = [posestamped.pose.position.x, posestamped.pose.position.y, posestamped.pose.position.z,
        posestamped.pose.orientation.x,posestamped.pose.orientation.y,posestamped.pose.orientation.z,posestamped.pose.orientation.w]

        # writing the data into the file
        writer.writerow(robot_pose)
    f.close()
