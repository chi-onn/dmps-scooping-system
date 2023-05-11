#!/usr/bin/env python

# DMP IN CARTESIAN, EXECUTED IN JOINT DUE TO TRAJOPT PLANNER

# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import sin, cos, tan
import math

import pydmps
import pydmps.dmp_discrete

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from perception_msgs.srv import Get3dPointStringInput
from perception_msgs.srv import Get3dPointStringInputRequest
from perception_msgs.srv import Get3dPointStringInputResponse
from perception_msgs.srv import GetObjects, GetObjectsRequest, GetObjectsResponse
import tf2_ros, tf2_geometry_msgs

import actionlib
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectoryPoint

from get_fk_client import *
#from get_ik_client import *
from trajopt import ctrajoptpy as trajoptpy
from trajopt_test import trajopt_init
import json


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True

class initiate_robot(object):
    def __init__(self):
        super(initiate_robot, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "xarm6"
        move_group = moveit_commander.MoveGroupCommander(group_name, wait_for_servers=30.0)
        grasping_group = moveit_commander.MoveGroupCommander("xarm_gripper")
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,)

        planning_frame = move_group.get_planning_frame()
        # print("============ Planning frame: %s" % planning_frame)
        eef_link = move_group.get_end_effector_link()
        # print("============ End effector link: %s" % eef_link)
        group_names = robot.get_group_names()
        # print("============ Available Planning Groups:", robot.get_group_names())
        # print("============ Printing robot state")
        # print(robot.get_current_state())
        # print("")

        # Misc variables
        self.spoon_name = ""
        self.bowl_name = ""
        self.bowl_x, self.bowl_y = 0, 0
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.robot_state = robot.get_current_state()

        self.df = pd.read_csv('/home/chi-onn/fyp_ws/src/scooping/traj_files/scooping_bowl_left_pose.csv',header=None)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_objects(self):
        rospy.wait_for_service('get_objects')
        try:
            get_objects_srv = rospy.ServiceProxy('get_objects',GetObjects)
            req = GetObjectsRequest()
            resp = get_objects_srv(req)
            objects = resp.object_names
            rospy.loginfo(f"Obtained {objects} in scene.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        return objects

    def get_3d_point_string_input(self,obj_name):
        # tf_buffer = tf2_ros.Buffer()
        # tf_listener = tf2_ros.TransformListener(tf_buffer)
        rospy.wait_for_service('get_3d_point_string_input')

        try:
            get_obj_pos_srv = rospy.ServiceProxy('get_3d_point_string_input',Get3dPointStringInput)
            req = Get3dPointStringInputRequest()
            req.object_name.data = obj_name
            resp = get_obj_pos_srv(req)
            # print('BEFORE TRANSFORM\n')
            # print(resp)
            try:
                trans = self.tf_buffer.lookup_transform(self.planning_frame,resp.point.header.frame_id, rospy.Time())
                msg_obj_pose_transformed = tf2_geometry_msgs.do_transform_point(resp.point,trans)
                # print('AFTER TRANSFORM\n')
                # print(msg_obj_pose_transformed)        
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn(f"Failed to obtain transformation for object centers between {resp.point.header.frame_id} and {self.planning_frame}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        obj_pos = [msg_obj_pose_transformed.point.x, msg_obj_pose_transformed.point.y, msg_obj_pose_transformed.point.z]
        return obj_pos

    def go_to_detect_bowl(self):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()

        # position final:position: [0.29392415285110474, -1.0521132946014404, -1.9188756942749023, -0.444137305021286, 2.417985200881958, 0.004649879410862923
        joint_goal[0] = 0.29392415285110474
        joint_goal[1] = -1.0521132946014404
        joint_goal[2] = -1.9188756942749023
        joint_goal[3] = -0.444137305021286
        joint_goal[4] = 2.417985200881958
        joint_goal[5] = 0.004649879410862923
        
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.go(joint_goal, wait=True)
        move_group.stop()

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def produce_scoop_dmp(self,bowl_pos_x, bowl_pos_y, bowl_pos_z):
        df = self.df
        df_copy = copy.deepcopy(df)

        df_copy = np.transpose(df_copy)
        y_des = df_copy.to_numpy()

        # original dmp traj
        dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=7, n_bfs=500, dt = 0.025, ay=np.ones(7) * 10.0)
        y_track = []
        dy_track = []
        ddy_track = []

        dmp.imitate_path(y_des=y_des, plot=False)
        # dmp.reset_state()      
        y_track_ori, dy_track_ori, ddy_track_ori = dmp.rollout()

        scoop_traj_ori = y_track_ori

        # offset dmp traj
        dmp.reset_state() 
        # original bowl position = [0.6542916139117372, -0.43426200282766514, 0.2646180732965431]
        # original bowl position bowl_lef) = 0.5093849131582481, -0.28866651753570133, 0.20004996849552092
        bowl_ori_x = 0.5093849131582481
        bowl_ori_y = -0.28866651753570133
        bowl_ori_z = 0.20004996849552092

        # to increase room for error
        dmp.y0 -= np.array([0.03, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0])
        dmp.goal -= np.array([0.03, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0])

        # take into account offset of bowl
        dmp.y0 -= np.array([bowl_ori_x-bowl_pos_x, bowl_ori_y-bowl_pos_y, bowl_ori_z-bowl_pos_z, 0.0, 0.0, 0.0, 0.0])
        dmp.goal -= np.array([bowl_ori_x-bowl_pos_x, bowl_ori_y-bowl_pos_y, bowl_ori_z-bowl_pos_z, 0.0, 0.0, 0.0, 0.0])
        y_track, dy_track, ddy_track = dmp.rollout()

        raw_scoop = []
        for row in range(y_track.shape[0]):
            normalised_final = y_track[row,:]
            norm = math.sqrt(sum(np.square(y_track[row,3:7])))
            normalised = y_track[row,3:7]/norm
            normalised_final[3:7] = normalised
            raw_scoop.append(normalised_final)

        scoop_traj = raw_scoop

        return scoop_traj_ori, scoop_traj
    
    def go_to_scooping_init(self,input_traj): 
        # move_group = self.move_group
        # joint_goal = move_group.get_current_joint_values()
        
        # joint_goal = input_traj[0]
        # print(joint_goal)
        # move_group.go(joint_goal, wait=True)
        # move_group.set_max_velocity_scaling_factor(0.5)

        # move_group.stop()

        # current_joints = move_group.get_current_joint_values()
        # return all_close(joint_goal, current_joints, 0.01)
    
        move_group = self.move_group
        mouth_pos = input_traj[0]
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = mouth_pos[0]
        pose_goal.position.y = mouth_pos[1]
        pose_goal.position.z = mouth_pos[2]
        pose_goal.orientation.x = mouth_pos[3]
        pose_goal.orientation.y = mouth_pos[4]
        pose_goal.orientation.z = mouth_pos[5]
        pose_goal.orientation.w = mouth_pos[6] 

        move_group.set_pose_target(pose_goal)
        move_group.set_max_velocity_scaling_factor(0.2)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def go_to_mouth_pos(self): 
        move_group = self.move_group
        mouth_pos = [0.0719798999340879, -0.33155671822211386, 0.6727158985799516, 0.07847865880429095, 0.7326286727675786, -0.6659473695308138, 0.11666374315135897]
        #OLD, TOO SHORT, 0.03386095331095375,-0.40206855578709827,0.47907519103362095,
        #0.04236201355077926,0.7343593262296478,-0.6736895813756196,0.07116310654688636


        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = mouth_pos[0]
        pose_goal.position.y = mouth_pos[1]
        pose_goal.position.z = mouth_pos[2]
        pose_goal.orientation.x = mouth_pos[3]
        pose_goal.orientation.y = mouth_pos[4]
        pose_goal.orientation.z = mouth_pos[5]
        pose_goal.orientation.w = mouth_pos[6] 

        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    # def plan_cartesian_path(self, input_traj):
    #     move_group = self.move_group
    #     df = input_traj
    #     pose_list = []
    #     for rows in range(df.shape[0]): 
    #         pose_temp = Pose()
    #         pose_temp.position.x,pose_temp.position.y,pose_temp.position.z = df.iloc[rows,0],df.iloc[rows,1],df.iloc[rows,2]
    #         pose_temp.orientation.x,pose_temp.orientation.y,pose_temp.orientation.z,pose_temp.orientation.w = df.iloc[rows,3],df.iloc[rows,4],df.iloc[rows,5],df.iloc[rows,6]
    #         pose_list.append(pose_temp)

    #     (plan, fraction) = move_group.compute_cartesian_path(
    #         pose_list, 0.01, jump_threshold=0.0  # waypoints to follow  # eef_step
    #     ) 
    #     return plan, fraction

    def execute_plan(self, plan):
        move_group = self.move_group

        plan = self.move_group.retime_trajectory(self.robot_state,plan,0.3)
        move_group.execute(plan, wait=True)
    
    def wait_for_state_update(
        self, object_is_known=False, object_is_attached=False, timeout=4):
        spoon_name = self.spoon_name
        scene = self.scene

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the spoon is in attached objects
            attached_objects = scene.get_attached_objects([spoon_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the spoon is in the scene.
            # Note that attaching the spoon will remove it from known_objects
            is_known = spoon_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (object_is_attached == is_attached) and (object_is_known == is_known):
                return True

            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False

    def add_spoon(self, timeout=4):
        spoon_name = self.spoon_name
        scene = self.scene

        spoon_pose = geometry_msgs.msg.PoseStamped()
        spoon_pose.header.frame_id = 'link_tcp'
        angle = pi
        spoon_pose.pose.orientation.w = 1
        spoon_pose.pose.position.x = -0.03
        spoon_name = "spoon"
        file_name = '/home/chi-onn/fyp_ws/src/meshes/spoon.STL'
        scene.add_mesh(spoon_name, spoon_pose,file_name, size=(0.001, 0.001, 0.001))
        #add_mesh(self, name, pose, filename, size=(1, 1, 1))

        self.spoon_name = spoon_name
        return self.wait_for_state_update(object_is_known=True, timeout=timeout)

    def attach_spoon(self, timeout=4):
        spoon_name = self.spoon_name
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link
        group_names = self.group_names

        grasping_group = "xarm_gripper"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_mesh(eef_link, spoon_name, touch_links=touch_links)

        return self.wait_for_state_update(
            object_is_attached=True, object_is_known=False, timeout=timeout
        )

    def add_bowl(self, bowl_pos_x, bowl_pos_y, bowl_pos_z, timeout=4):
        bowl_name = self.bowl_name
        scene = self.scene

        bowl_pose = geometry_msgs.msg.PoseStamped()
        bowl_pose.header.frame_id = 'world'
        angle = pi
        bowl_pose.pose.orientation.w = 1

        # CALIBRATION FOR CAMERA
        bowl_pose.pose.position.x = bowl_pos_x - 0.025
        bowl_pose.pose.position.y = bowl_pos_y +0.03#+0.07 
        bowl_pose.pose.position.z = bowl_pos_z
        bowl_name = "bowl"
        file_name = '/home/chi-onn/fyp_ws/src/meshes/bowl.STL'
        scene.add_mesh(bowl_name, bowl_pose,file_name, size=(0.001, 0.001, 0.001))

        self.bowl_name = bowl_name
        self.bowl_x, self.bowl_y = bowl_pose.pose.position.x,bowl_pose.pose.position.y
        return self.wait_for_state_update(object_is_known=True, timeout=timeout)
    
    def plot_traj(self, scoop_traj_ori, scoop_traj):
        gfk = GetFK('link_tcp', 'world')
        rospy.sleep(3)
        scoop_traj_x, scoop_traj_y, scoop_traj_z = [],[],[]
        temp_df = pd.DataFrame(scoop_traj)
        for row in range(temp_df.shape[0]):
            joint_angles = temp_df.iloc[row,:]
            joint_angles[6],joint_angles[7],joint_angles[8],joint_angles[9],joint_angles[10],joint_angles[11]=0,0,0,0,0,0
            resp = gfk.get_fk_list(joint_angles)
            posestamped = resp.pose_stamped[0]
            scoop_traj_x.append(posestamped.pose.position.x)
            scoop_traj_y.append(posestamped.pose.position.y)
            scoop_traj_z.append(posestamped.pose.position.z)

        temp_df_ori = pd.DataFrame(scoop_traj_ori)
        scoop_traj_x_ori = temp_df_ori.iloc[:,0]
        scoop_traj_y_ori = temp_df_ori.iloc[:,1]
        scoop_traj_z_ori = temp_df_ori.iloc[:,2]
        
        fig = plt.figure()
        
        # defining all 3 axis
        x_ori = np.array(scoop_traj_x_ori)
        y_ori = np.array(scoop_traj_y_ori)
        z_ori = np.array(scoop_traj_z_ori)

        x = np.array(scoop_traj_x)
        y = np.array(scoop_traj_y)
        z = np.array(scoop_traj_z)

        # plotting
        # syntax for 3-D projection
        ax = plt.axes(projection ='3d')
        ax.plot3D(x_ori, y_ori, z_ori, 'b')
        ax.plot3D(x, y, z, 'b--')

        ax.scatter(x_ori[0], y_ori[0], z_ori[0],'rx') 
        ax.scatter(x[0], y[0], z[0],'rx') 

        ax.set_title('Scooping trajectory produced by DMPs')
        plt.legend(["Original trajectory", "Offset trajectory"])
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel('z', rotation=0)

        plt.show()
        return 
    
    def moveit_execute_traj_client(self, joint_traj):
        """
        Calls the moveit action server to execute the robot trajectory specified
        Args:
            joint_traj ([float]): Waypoints of trajectory in joint space
        """
        client = actionlib.SimpleActionClient("/execute_trajectory", ExecuteTrajectoryAction)
        client.wait_for_server()
        goal = ExecuteTrajectoryGoal()
        goal.trajectory.joint_trajectory.header.frame_id = self.planning_frame
        goal.trajectory.joint_trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        for idx, joint_poses in enumerate(joint_traj):
            #print(joint_poses)
            msg_traj_point = JointTrajectoryPoint()
            msg_traj_point.positions = joint_poses
            # Assumes velocity fixed
            msg_traj_point.time_from_start = rospy.Time(idx * 0.2)  # originally 0.5
            goal.trajectory.joint_trajectory.points.append(msg_traj_point)
        client.send_goal(goal)
        client.wait_for_result()
        result = client.get_result()
        if result.error_code.val != 1:
            moveit_error_dict = {}
            for name in MoveItErrorCodes.__dict__.keys():
                if not name[:1] == '_':
                    code = MoveItErrorCodes.__dict__[name]
                    moveit_error_dict[code] = name
            rospy.logwarn(f"Execution failed with error code {result.error_code.val}:  {moveit_error_dict[result.error_code.val]}")
        self.reached_goal =True
        return result

def main():
    try:
        # # Setting up MoveitCommander
        myrobot = initiate_robot()
        trajopt_obj = trajopt_init()
        myrobot.add_spoon()
        myrobot.attach_spoon()
        #myrobot.go_to_detect_bowl()



        # # # Object Detection
        # obj_names = myrobot.get_objects()
        # if 'bowl' in obj_names:
        #     print('Bowl detected')
        #     bowl_pos = myrobot.get_3d_point_string_input('bowl')
        #     print(bowl_pos)
        #     bowl_ori = [0.5093849131582481, -0.28866651753570133, 0.20004996849552092]
        #     print(bowl_ori)
        #     bowl_shift_x, bowl_shift_y, bowl_shift_z = bowl_pos[0]-bowl_ori[0], bowl_pos[1]-bowl_ori[1], bowl_pos[2]-bowl_ori[2]
        #     print('Bowl shifted by: ',bowl_shift_x,',',bowl_shift_y,',',bowl_shift_z)

        # else:
        #     print('Bowl not detected')
        #     # find a way to make program end
        
        # if no object detection code
        #old bowl_pos = [0.6542916139117372, -0.43426200282766514, 0.2646180732965431]
        # before transform - bowl_pos = [-0.09121962258291694, -0.03908291508838767, 0.7649999856948853]
        bowl_pos = [0.5093849131582481, -0.28866651753570133, 0.20004996849552092]
        # #offset to test different bowl pos
        # bowl_pos[0] -= 0.1
        # bowl_pos[1] += 0.4
        # # Insert bowl
        myrobot.add_bowl(bowl_pos[0],bowl_pos[1],bowl_pos[2])


        # Plan & execute trajectory
        scoop_traj_ori, scoop_traj = myrobot.produce_scoop_dmp(bowl_pos[0],bowl_pos[1],bowl_pos[2])
        myrobot.go_to_scooping_init(scoop_traj)

        # trajopt
        costs = trajopt_obj.get_cart_costs_and_constraints(scoop_traj, costs=True)
        request = trajopt_obj.make_request(costs)
        request_json = json.dumps(request)
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        prob = trajoptpy.ConstructProblem(request_json, trajopt_obj.get_current_joint_state(), joint_names)
        result = trajoptpy.OptimizeProblem(prob)
        trajopt_scoop = result.GetTraj()

        #input("Execute scoop by pressing 'Enter'")
        scoop_success = myrobot.moveit_execute_traj_client(trajopt_scoop)


        

        # # myrobot.go_to_mouth_pos() 
        myrobot.plot_traj(scoop_traj_ori, trajopt_scoop)

        

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()