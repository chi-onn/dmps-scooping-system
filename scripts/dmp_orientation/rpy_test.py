#!/usr/bin/env python

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
        #print("============ Planning frame: %s" % planning_frame)
        eef_link = move_group.get_end_effector_link()
        #print("============ End effector link: %s" % eef_link)
        group_names = robot.get_group_names()
        #print("============ Available Planning Groups:", robot.get_group_names())
        #print("============ Printing robot state")
        #print(robot.get_current_state())
        #print("")

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

        self.df = pd.read_csv('/home/chi-onn/fyp_ws/src/scooping/traj_files/scooping_final_pose.csv',header=None)
    
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

        # bowl_pose.pose.position.x = 0.67 -0.3  #original: 0.67
        # bowl_pose.pose.position.y = -0.16 -0.3 #original: -0.16
        # bowl_pose.pose.position.z = 0.16   #original: 0.16
        bowl_pose.pose.position.x = bowl_pos_x #- 0.025
        bowl_pose.pose.position.y = bowl_pos_y #+0.03 
        bowl_pose.pose.position.z = bowl_pos_z
        bowl_name = "bowl"
        file_name = '/home/chi-onn/fyp_ws/src/meshes/bowl.STL'
        scene.add_mesh(bowl_name, bowl_pose,file_name, size=(0.001, 0.001, 0.001))

        self.bowl_name = bowl_name
        self.bowl_x, self.bowl_y = bowl_pose.pose.position.x,bowl_pose.pose.position.y
        return self.wait_for_state_update(object_is_known=True, timeout=timeout)

    def move_to_pose(self):
        move_group = self.move_group
        # dmp.y0 = [0.47205505856882735, -0.2057627253679792, 0.32268336054600444, -0.060469034603161866, 0.3375821878038384, -0.19544419922357728, 0.9187944967845589]
        #dmp.goal = [0.4304503431720169, -0.24113287193670307, 0.3019438862323587, -0.6061252702823084, 0.3967394242520719, -0.14486325414451476, 0.6739618858422154]
        mouth_pos = [0.47205505856882735, -0.2057627253679792, 0.32268336054600444, -0.0604690346031619, 0.337582187803838, -0.195444199223577, 0.918794496784559]
        # mouth_pos = [0.47205505856882735, -0.2057627253679792, 0.32268336054600444, -0.281464718623584, 0.195948589784185, 0.511485900578677, 0.787885737807809]

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = mouth_pos[0]
        pose_goal.position.y = mouth_pos[1]
        pose_goal.position.z = mouth_pos[2]
        pose_goal.orientation.x = mouth_pos[3]
        pose_goal.orientation.y = mouth_pos[4]
        pose_goal.orientation.z = mouth_pos[5]
        pose_goal.orientation.w = mouth_pos[6] 

        move_group.set_pose_target(pose_goal)
        move_group.set_max_velocity_scaling_factor(0.5)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)
        
def main():
    try:
        # # Setting up MoveitCommander
        myrobot = initiate_robot()
        myrobot.add_spoon()
        myrobot.attach_spoon()

        # # # test without object detection
        bowl_pos = [0.5093849131582481, -0.28866651753570133, 0.20004996849552092]

        # # # Insert bowl
        myrobot.add_bowl(bowl_pos[0],bowl_pos[1],bowl_pos[2])
        myrobot.move_to_pose()
        # Adjust bowl position to match camera
        # input("Adjust bowl position to match camera and press 'Enter'")

        # # Plan & execute trajectory
        # scoop_traj = myrobot.produce_scoop_dmp(bowl_pos[0],bowl_pos[1],bowl_pos[2])
        # myrobot.go_to_scooping_init(scoop_traj) 
        # cartesian_plan, fraction = myrobot.plan_cartesian_path(scoop_traj)
        # myrobot.execute_plan(cartesian_plan)
        # myrobot.go_to_mouth_pos() 

        

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()