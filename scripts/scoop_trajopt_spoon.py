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
#from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import quaternion
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
from pose_transform import *
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

        #position middle: [-0.022390367463231087, -0.7659856081008911, -2.083477735519409, -0.2575419545173645, 2.1557090282440186, -0.09019806981086731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # joint_goal[0] = -0.022390367463231087
        # joint_goal[1] = -0.7659856081008911
        # joint_goal[2] = -2.083477735519409
        # joint_goal[3] = -0.2575419545173645
        # joint_goal[4] = 2.1557090282440186
        # joint_goal[5] = -0.09019806981086731

        # see more right position: [-0.023055730387568474, -0.7659280896186829, -2.083491086959839, -0.6445001363754272, 2.0630545616149902, -0.09014438092708588, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # joint_goal[0] = -0.023055730387568474
        # joint_goal[1] = -0.7659280896186829
        # joint_goal[2] = -2.083491086959839
        # joint_goal[3] = -0.6445001363754272
        # joint_goal[4] = 2.0630545616149902
        # joint_goal[5] = -0.09014438092708588

        # see more left position: [-0.02302505075931549, -0.7098937034606934, -2.0834872722625732, -0.06657668203115463, 2.1776487827301025, -0.09022107720375061]
        # joint_goal[0] = -0.02302505075931549
        # joint_goal[1] = -0.7098937034606934
        # joint_goal[2] = -2.0834872722625732
        # joint_goal[3] = -0.06657668203115463
        # joint_goal[4] = 2.1776487827301025
        # joint_goal[5] = -0.09022107720375061
        move_group.set_max_velocity_scaling_factor(0.2)
        move_group.go(joint_goal, wait=True)
        move_group.stop()

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def produce_scoop_dmp(self,bowl_pos_x, bowl_pos_y, bowl_pos_z):
        df = self.df
        df_spoon = eef_to_spoon_traj(df)
        df_copy = copy.deepcopy(df_spoon)
        df_copy = np.transpose(df_copy)

        y_des = df_copy
        # y_des = df_copy.to_numpy()

        # original dmp traj
        dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=7, n_bfs=200, dt = 0.025, ay=np.ones(7) * 10.0)  # ORIGINAL N_BFS = 500
        y_track = []
        dy_track = []
        ddy_track = []

        dmp.imitate_path(y_des=y_des, plot=False)
        
        # # calibrate norm & short
        # dmp.y0 += np.array([-0.01, -0.01,-0.005,0.0,0.0,0.0,0.0])  
        # dmp.goal += np.array([-0.02, 0.0,-0.005,0.0,0.0,0.0,0.0])   
        # calibrate long
        dmp.y0 += np.array([-0.02, -0.01,0.01,0.0,0.0,0.0,0.0])  
        dmp.goal += np.array([-0.03, 0.0,0.01,0.0,0.0,0.0,0.0])  
        y_track_ori, dy_track_ori, ddy_track_ori = dmp.rollout()

        scoop_traj_ori_spoon = y_track_ori
        scoop_traj_ori = spoon_to_eef_traj(scoop_traj_ori_spoon)

        # offset dmp traj
        dmp.reset_state() 

        gfk = GetFK('link_tcp', 'world')
        rospy.sleep(3)
        ######################################################################################################
        # MODIFIED VARIATION
        # # modified position 1 (9-3, causes error afterwards)
        # joint_start = [-0.6134218080746506, -0.4524904519855002, -0.6550432606662434, 1.2421662042197232, 0.5067185788823427, -1.5335535961387377,0,0,0,0,0,0]
        # joint_end   = [-1.0632509343189922, -0.021655783931723387, -0.5698337643011968, -1.0559866666205182, -0.905622756773425, -1.0482338344258801,0,0,0,0,0,0]
        # dmp.w *= 0.2

        # # # modified position 2 (2-8, mirror of demonstrated)
        # # joint_start = [-1.0048515330168268, 0.273674019580252, -1.5750972954153344, 0.8941454708663932, 1.3504024526899248, 1.8876781312329007,0,0,0,0,0,0] # veli deep
        # joint_start = [-0.9833948772246884, 0.20562068650067436, -1.4886913895558853, 0.9158711399566625, 1.3176781975451644, 1.6057615739929665,0,0,0,0,0,0]
        # joint_end   = [-1.1001465826530232, 0.23078474032158572, -1.2932792124432546, 1.1591664142308138, 1.5768943083960714, 2.6406064496638812,0,0,0,0,0,0]
        # dmp.w *= 0.1

        # # modified position 3 upside down scoop (12-6)
        # joint_start = [-0.09824255203219395, 0.03831483164360754, -1.2465598230617978, -0.9902035382305344, 1.083933014563917, 1.3203242737116698,0,0,0,0,0,0]
        # joint_end   = [0.06517231597432391, 0.07244631333118749, -1.092539835008934, -1.208335924574612, 1.436126646715508, 0.5356434742291258,0,0,0,0,0,0]
        # # joint_end   = [0.08107458901417022, 0.06526235181866427, -1.0632322961610239, -1.21155413110561, 1.4025892350130094, 0.5153829894680502,0,0,0,0,0,0]  #upside down
        # dmp.w *= 0.2

        # # modified position 4 (5-11, causes error afterwards)
        # joint_start = [-0.8298035825813126, -0.23758293409295117, -1.0349360601134414, -2.5245556440426675, -0.4373985680033223, -0.4744129799790156,0,0,0,0,0,0]
        # joint_end   = [-0.30962071407039327, -0.2719604675881977, -0.34834093483394357, 0.33149601729244965, -0.6235331109489765, -1.7896270397601488,0,0,0,0,0,0]
        # dmp.w *= 0.1

        # modified position 5 (6-12)
        joint_start = [0.013168499916740952, -0.013555595629661875, -1.1778022368419538, -1.0006463276185842, 1.396489818554514, -0.6420838727923082,0,0,0,0,0,0]
        # joint_end   = [-0.00010321913854114081, 0.28952384502157963, -1.4072600639635775, -1.15923840473083, 1.3954657347990318, 0.39199808267344244,0,0,0,0,0,0] # upside down
        # joint_end   = [-0.006343522549030566, 0.31890850439316637, -1.4172869502202718, -1.165006389405802, 1.3826056549705088, 0.40785707664939147,0,0,0,0,0,0] # even larger upside down
        joint_end   = [0.04129501976222068, 0.25474079377395437, -1.389190587024844, -1.1414547057197924, 1.436259007955426, 0.3921076917955159,0,0,0,0,0,0] 
        dmp.w *= 0.6
        
        # only if using modified-from here
        resp_start = gfk.get_fk_list(joint_start)
        posestamped_start = resp_start.pose_stamped[0]
        joint_start = [posestamped_start.pose.position.x, 
            posestamped_start.pose.position.y, 
            posestamped_start.pose.position.z,
            posestamped_start.pose.orientation.x,
            posestamped_start.pose.orientation.y,
            posestamped_start.pose.orientation.z,
            posestamped_start.pose.orientation.w]
        
        
        resp_end = gfk.get_fk_list(joint_end)
        posestamped_end = resp_end.pose_stamped[0]
        joint_end = [posestamped_end.pose.position.x, 
            posestamped_end.pose.position.y, 
            posestamped_end.pose.position.z,
            posestamped_end.pose.orientation.x,
            posestamped_end.pose.orientation.y,
            posestamped_end.pose.orientation.z,
            posestamped_end.pose.orientation.w]
        
        dmp.y0 = eef_to_spoon_pose(joint_start)
        dmp.goal = eef_to_spoon_pose(joint_end)
        # only if using modified-until here

        # calibrate
        # # modpos1 ERROR
        # dmp.y0 += np.array([-0.05, 0.02,0.0,0.0,0.0,0.0,0.0])  
        # dmp.goal += np.array([-0.05, 0.04,0.0,0.0,0.0,0.0,0.0])  

        # # modpos2
        # dmp.y0 += np.array([-0.03, 0.0,0.01,0.0,0.0,0.0,0.0])  
        # dmp.goal += np.array([-0.03, 0.0,0.01,0.0,0.0,0.0,0.0])  

        # # modpos3
        # dmp.y0 += np.array([-0.02, 0.07,0.01,0.0,0.0,0.0,0.0])  
        # dmp.goal += np.array([-0.02, 0.07,0.01,0.0,0.0,0.0,0.0])  

        # # modpos4
        # dmp.y0 += np.array([-0.04, 0.02,0.018,0.0,0.0,0.0,0.0])  
        # dmp.goal += np.array([-0.05, 0.01,0.018,0.0,0.0,0.0,0.0])  

        # modpos5
        dmp.y0 += np.array([-0.03, 0.04,0.02,0.0,0.0,0.0,0.0])  
        dmp.goal += np.array([-0.03, 0.04,0.02,0.0,0.0,0.0,0.0])  

        ###########################################################################################################
        # # try rotating to change start/goal
        # dmp.y0 = rotate_pose(dmp.y0,-45)
        # dmp.goal = rotate_pose(dmp.goal,-45)

        # # -45 degree macam can
        # dmp.y0 += np.array([0.0,-0.07,0.0,0.0,0.0,0.0,0.0])
        # dmp.goal += np.array([-0.03,0.01,0.0,0.0,0.0,0.0,0.0])

        # # # # # 30 degree not bad, spoon handle almost hit
        # # # dmp.y0 += np.array([-0.01,0.025,0.0,0.0,0.0,0.0,0.0])
        # # # dmp.goal += np.array([0.03,0.01,0.0,0.0,0.0,0.0,0.0])
        # # # dmp.w *= 0.6

        # # # # # 70 degree DOES NOT WORK DO NOT ATTEMPT
        # # dmp.y0 += np.array([-0.05,0.04,0.0,0.0,0.0,0.0,0.0])
        # # dmp.goal += np.array([0.04,0.04,0.0,0.0,0.0,0.0,0.0])
        # # # dmp.w *= 0.6

        ###########################################################################################################
        # OFFSET VARIATION
        # ADAPT TO CHANGED BOWL POSITION
        # original bowl position: 0.5093849131582481, -0.28866651753570133, 0.20004996849552092
        bowl_ori_x = 0.5093849131582481
        bowl_ori_y = -0.28866651753570133
        bowl_ori_z = 0.20004996849552092
        dmp.y0 -= np.array([bowl_ori_x-bowl_pos_x, bowl_ori_y-bowl_pos_y, bowl_ori_z-bowl_pos_z, 0.0, 0.0, 0.0, 0.0])
        dmp.goal -= np.array([bowl_ori_x-bowl_pos_x, bowl_ori_y-bowl_pos_y, bowl_ori_z-bowl_pos_z, 0.0, 0.0, 0.0, 0.0])

        # LENGTH VARYING VARIATION
        dmp.y0, dmp.goal = change_scoop_length(dmp.y0, dmp.goal, 'long')


        y_track, dy_track, ddy_track = dmp.rollout()

        raw_scoop = []
        for row in range(y_track.shape[0]):
            normalised_final = y_track[row,:]
            norm = math.sqrt(sum(np.square(y_track[row,3:7])))
            normalised = y_track[row,3:7]/norm
            normalised_final[3:7] = normalised
            raw_scoop.append(normalised_final)
        raw_scoop = np.stack(raw_scoop)

        scoop_traj_spoon = raw_scoop
        scoop_traj = spoon_to_eef_traj(scoop_traj_spoon)

        return scoop_traj_ori, scoop_traj
    
    def go_to_scooping_init(self,input_traj):
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
        bowl_pose.pose.position.x = bowl_pos_x #- 0.025
        bowl_pose.pose.position.y = bowl_pos_y #+0.03 
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
        scoop_traj_x, scoop_traj_y, scoop_traj_z, scoop_traj_quart_x, scoop_traj_quart_y, scoop_traj_quart_z, scoop_traj_quart_w = [],[],[],[],[],[],[]
        temp_df = pd.DataFrame(scoop_traj)
        for row in range(temp_df.shape[0]):
            joint_angles = temp_df.iloc[row,:]
            joint_angles[6],joint_angles[7],joint_angles[8],joint_angles[9],joint_angles[10],joint_angles[11]=0,0,0,0,0,0
            resp = gfk.get_fk_list(joint_angles)
            posestamped = resp.pose_stamped[0]
            scoop_traj_x.append(posestamped.pose.position.x)
            scoop_traj_y.append(posestamped.pose.position.y)
            scoop_traj_z.append(posestamped.pose.position.z)
            scoop_traj_quart_x.append(posestamped.pose.orientation.x)
            scoop_traj_quart_y.append(posestamped.pose.orientation.y)
            scoop_traj_quart_z.append(posestamped.pose.orientation.z)
            scoop_traj_quart_w.append(posestamped.pose.orientation.w)

            if row == 0:
                start_mod = [posestamped.pose.position.x, 
                        posestamped.pose.position.y, 
                        posestamped.pose.position.z,
                        posestamped.pose.orientation.x,
                        posestamped.pose.orientation.y,
                        posestamped.pose.orientation.z,
                        posestamped.pose.orientation.w]
            elif row == 39:
                end_mod = [posestamped.pose.position.x, 
                        posestamped.pose.position.y, 
                        posestamped.pose.position.z,
                        posestamped.pose.orientation.x,
                        posestamped.pose.orientation.y,
                        posestamped.pose.orientation.z,
                        posestamped.pose.orientation.w]
            else:
                pass

        temp_df_ori = pd.DataFrame(scoop_traj_ori)
        scoop_traj_x_ori = temp_df_ori.iloc[:,0]
        scoop_traj_y_ori = temp_df_ori.iloc[:,1]
        scoop_traj_z_ori = temp_df_ori.iloc[:,2]
        
        fig = plt.figure(1)
        
        # defining all 3 axis
        x_ori = np.array(scoop_traj_x_ori)
        y_ori = np.array(scoop_traj_y_ori)
        z_ori = np.array(scoop_traj_z_ori)

        x = np.array(scoop_traj_x)
        y = np.array(scoop_traj_y)
        z = np.array(scoop_traj_z)

        # for orientation
        start_ori = scoop_traj_ori[0,:]
        end_ori = scoop_traj_ori[39,:]

        position_start_ori = start_ori[0:3]
        orientation_start_ori = np.quaternion(start_ori[6],start_ori[3],start_ori[4],start_ori[5])
        orientation_matrix_start_ori = quaternion.as_rotation_matrix(orientation_start_ori)

        position_end_ori = end_ori[0:3]
        orientation_end_ori = np.quaternion(end_ori[6],end_ori[3],end_ori[4],end_ori[5])
        orientation_matrix_end_ori = quaternion.as_rotation_matrix(orientation_end_ori)

        position_start_mod = start_mod[0:3]
        orientation_start_mod = np.quaternion(start_mod[6],start_mod[3],start_mod[4],start_mod[5])
        orientation_matrix_start_mod = quaternion.as_rotation_matrix(orientation_start_mod)

        position_end_mod = end_mod[0:3]
        orientation_end_mod = np.quaternion(end_mod[6],end_mod[3],end_mod[4],end_mod[5])
        orientation_matrix_end_mod = quaternion.as_rotation_matrix(orientation_end_mod)


        # PLOTTING TRAJECTORY
        # plot x,y,z of traj
        ax = plt.axes(projection ='3d')
        ax.plot3D(x_ori, y_ori, z_ori, 'b',label = 'Original trajectory')
        ax.plot3D(x, y, z, 'b--',label = 'Modified trajectory')

        # plot orientation of start and end
        colors = ['r', 'g', 'b']
        for i in range(3):
            ax.quiver(position_start_ori[0], position_start_ori[1], position_start_ori[2],
                      orientation_matrix_start_ori[0][i], orientation_matrix_start_ori[1][i], orientation_matrix_start_ori[2][i],
                      length=0.01, color=colors[i])
            
        for i in range(3):
            ax.quiver(position_end_ori[0], position_end_ori[1], position_end_ori[2],
                      orientation_matrix_end_ori[0][i], orientation_matrix_end_ori[1][i], orientation_matrix_end_ori[2][i],
                      length=0.01, color=colors[i])
            
        for i in range(3):
            ax.quiver(position_start_mod[0], position_start_mod[1], position_start_mod[2],
                      orientation_matrix_start_mod[0][i], orientation_matrix_start_mod[1][i], orientation_matrix_start_mod[2][i],
                      length=0.01, color=colors[i])
            
        for i in range(3):
            ax.quiver(position_end_mod[0], position_end_mod[1], position_end_mod[2],
                      orientation_matrix_end_mod[0][i], orientation_matrix_end_mod[1][i], orientation_matrix_end_mod[2][i],
                      length=0.01, color=colors[i])

        ax.scatter(position_start_ori[0], position_start_ori[1], position_start_ori[2], marker='o', color='r',label='Starting pose')
        ax.scatter(position_end_ori[0], position_end_ori[1], position_end_ori[2], marker='o', color='k',label = 'Goal pose')

        ax.scatter(position_start_mod[0], position_start_mod[1], position_start_mod[2], marker='o', color='r')
        ax.scatter(position_end_mod[0], position_end_mod[1], position_end_mod[2], marker='o', color='k')

        ax.set_title('Scooping trajectory produced by DMPs')
        # plt.legend(["Original trajectory", "Offset trajectory"])
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel('z', rotation=0)
        ax.set_box_aspect([1,1,1])

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
        myrobot.go_to_detect_bowl()

        # Object Detection
        obj_names = myrobot.get_objects()
        if 'bowl' in obj_names:
            print('Bowl detected')
            bowl_pos = myrobot.get_3d_point_string_input('bowl')
            print(bowl_pos)
            bowl_ori = [0.5093849131582481, -0.28866651753570133, 0.20004996849552092]
            print()
            bowl_shift_x, bowl_shift_y, bowl_shift_z = bowl_ori[0]-bowl_pos[0], bowl_ori[1]-bowl_pos[1], bowl_ori[2]-bowl_pos[2]
            print('Bowl shifted by: ',bowl_shift_x,',',bowl_shift_y,',',bowl_shift_z)

        else:
            # print('Bowl not detected')
            rospy.signal_shutdown("Bowl not detected")

        # # Insert bowl
        myrobot.add_bowl(bowl_pos[0],bowl_pos[1],bowl_pos[2])


        # Plan & execute trajectory
        scoop_traj_ori, scoop_traj = myrobot.produce_scoop_dmp(bowl_pos[0],bowl_pos[1],bowl_pos[2])
        scoop_df = pd.DataFrame(scoop_traj)
        scoop_df.to_csv('scoop_from_dmp',index=False)



        myrobot.go_to_scooping_init(scoop_traj)

        # trajopt
        costs = trajopt_obj.get_cart_costs_and_constraints(scoop_traj, costs=True)
        request = trajopt_obj.make_request(costs)
        request_json = json.dumps(request)
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        prob = trajoptpy.ConstructProblem(request_json, trajopt_obj.get_current_joint_state(), joint_names)
        result = trajoptpy.OptimizeProblem(prob)
        trajopt_scoop = result.GetTraj()

        scoop_trajopt_df = pd.DataFrame(scoop_traj)
        scoop_trajopt_df.to_csv('scoop_from_dmp_trajopt',index=False)

        input("Execute scoop by pressing 'Enter'")
        scoop_success = myrobot.moveit_execute_traj_client(trajopt_scoop)

        

        # # myrobot.go_to_mouth_pos() 
        myrobot.plot_traj(scoop_traj_ori, trajopt_scoop)

        

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()