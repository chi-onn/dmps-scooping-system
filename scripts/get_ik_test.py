import pandas as pd
import numpy as np
from get_ik_client import *
import csv
import os.path
from geometry_msgs.msg import PoseStamped

if __name__ == '__main__':
    rospy.init_node('get_ik')
    gik = GetIK('xarm6')
    # df = pd.read_csv('/home/chionn/fyp_ws/src/movement_primitives/fyp/scooping2_pose_dmp_2.csv',header=None)
    # save_path = '/home/chionn/fyp_ws/src/movement_primitives/fyp'
    # file_name = 'scooping2_joint_dmp2.csv'

    # df = pd.read_csv('/home/chionn/fyp_ws/src/ros_dmp_moveit/scooping2_csv/scooping2_pose_dmp.csv',header=None)
    # save_path = '/home/chionn/fyp_ws/src/ros_dmp_moveit/scooping2_csv'
    # file_name = 'scooping2_joint_dmp1.csv'

    df = pd.read_csv('/home/chionn/fyp_ws/src/movement_primitives/fyp/test.csv',header=None)
    save_path = '/home/chionn/fyp_ws/src/movement_primitives/fyp'
    file_name = 'test_joint.csv'
    completeName = os.path.join(save_path,file_name)
    f = open(completeName, 'w')
    writer = csv.writer(f)
    for i in range(0,df.shape[0]):
        ps = PoseStamped()
        ps.pose.position.x = df.iloc[i,0]
        ps.pose.position.y = df.iloc[i,1]
        ps.pose.position.z = df.iloc[i,2]
        ps.pose.orientation.x = df.iloc[i,3]
        ps.pose.orientation.y = df.iloc[i,4]
        ps.pose.orientation.z = df.iloc[i,5]
        ps.pose.orientation.w = df.iloc[i,6]
        resp = gik.get_ik(ps)
        joint_angles = resp.solution.joint_state.position
        writer.writerow(joint_angles)

    f.close()


    # ps = PoseStamped()
    # ps.pose.position.x = df.iloc[0,0]
    # ps.pose.position.y = df.iloc[0,1]
    # ps.pose.position.z = df.iloc[0,2]
    # ps.pose.orientation.x = df.iloc[0,3]
    # ps.pose.orientation.y = df.iloc[0,4]
    # ps.pose.orientation.z = df.iloc[0,5]
    # ps.pose.orientation.w = df.iloc[0,6]
    # resp = gik.get_ik(ps)
    # #joint_angles = resp.solution.joint_state.position
    # print(resp)