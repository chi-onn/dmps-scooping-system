#!/usr/bin/env python


# DMP IN JOINT SPACE, EXECUTED IN JOINT SPACE
# DMP IN JOINT SPACE NOT IDEAL BECAUSE IF 1 JOINT DOES NOT MOVE DURING RECORDING, IT WILL NOT MOVE IN GENERATED TRAJ EVEN IF GOAL POS IS CHANGED
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
from get_ik_client import *



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

        self.df = pd.read_csv('/home/chi-onn/fyp_ws/src/scooping/traj_files/scooping_final.traj',header=None)
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
        joint_goal[0] = -0.02302505075931549
        joint_goal[1] = -0.7098937034606934
        joint_goal[2] = -2.0834872722625732
        joint_goal[3] = -0.06657668203115463
        joint_goal[4] = 2.1776487827301025
        joint_goal[5] = -0.09022107720375061
        move_group.go(joint_goal, wait=True)
        move_group.stop()

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def produce_scoop_dmp(self,bowl_pos_x, bowl_pos_y, bowl_pos_z):
        df = self.df
        df_joint = df.iloc[:,0:6]
        df_copy = copy.deepcopy(df_joint)

        df_copy = np.transpose(df_copy)
        y_des = df_copy.to_numpy()

        # original dmp traj
        dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=6, n_bfs=500, ay=np.ones(6) * 10.0)
        y_track = []
        dy_track = []
        ddy_track = []

        dmp.imitate_path(y_des=y_des, plot=False)
        # dmp.reset_state()      
        y_track_ori, dy_track_ori, ddy_track_ori = dmp.rollout()

        scoop_traj_ori = y_track_ori
        
        print(dmp.y0)
        print(dmp.goal)
        # offset dmp traj
        dmp.reset_state() 
        # get fk to get start and goal position in cartesian space
        joint_start = dmp.y0
        joint_start = np.append(joint_start,[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        joint_end = dmp.goal
        joint_end = np.append(joint_end,[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        gfk = GetFK('link_tcp', 'world')
        rospy.sleep(3)
        resp1_fk = gfk.get_fk_list(joint_start)
        resp2_fk = gfk.get_fk_list(joint_end)
        ps1_fk = resp1_fk.pose_stamped[0]
        pose_start = [ps1_fk.pose.position.x, ps1_fk.pose.position.y, ps1_fk.pose.position.z,
                      ps1_fk.pose.orientation.x,
                      ps1_fk.pose.orientation.y,
                      ps1_fk.pose.orientation.z,
                      ps1_fk.pose.orientation.w]
        ps2_fk = resp2_fk.pose_stamped[0]
        pose_end = [ps2_fk.pose.position.x, ps2_fk.pose.position.y, ps2_fk.pose.position.z,
                      ps2_fk.pose.orientation.x,
                      ps2_fk.pose.orientation.y,
                      ps2_fk.pose.orientation.z,
                      ps2_fk.pose.orientation.w]

        # get new start and end position based on position of bowl
        # original bowl position = [0.6542916139117372, -0.43426200282766514, 0.2646180732965431]
        pose_start -= np.array([0.6542916139117372-bowl_pos_x, -0.43426200282766514-bowl_pos_y, 0.2646180732965431-bowl_pos_z, 0.0, 0.0, 0.0, 0.0])
        pose_end -= np.array([0.6542916139117372-bowl_pos_x, -0.43426200282766514-bowl_pos_y, 0.2646180732965431-bowl_pos_z, 0.0, 0.0, 0.0, 0.0])
        # print(pose_start)
        # print(pose_end)

        # get ik to get start and end position in terms of joint angles
        
        gik = GetIK('xarm6',ik_timeout=1.0,ik_attempts=50)
        ps1_ik,ps2_ik = PoseStamped() , PoseStamped()
        ps1_ik.pose.position.x = pose_start[0]
        ps1_ik.pose.position.y = pose_start[1]
        ps1_ik.pose.position.z = pose_start[2]
        ps1_ik.pose.orientation.x = pose_start[3]
        ps1_ik.pose.orientation.y = pose_start[4]
        ps1_ik.pose.orientation.z = pose_start[5]
        ps1_ik.pose.orientation.w = pose_start[6]

        ps2_ik.pose.position.x = pose_end[0]
        ps2_ik.pose.position.y = pose_end[1]
        ps2_ik.pose.position.z = pose_end[2]
        ps2_ik.pose.orientation.x = pose_end[3]
        ps2_ik.pose.orientation.y = pose_end[4]
        ps2_ik.pose.orientation.z = pose_end[5]
        ps2_ik.pose.orientation.w = pose_end[6]

        resp1_ik = gik.get_ik(ps1_ik)
        resp2_ik = gik.get_ik(ps2_ik)
        joint_start_final = resp1_ik.solution.joint_state.position
        joint_end_final = resp2_ik.solution.joint_state.position
        # print(joint_start_final)
        # print(joint_end_final)


        # change start and goal state of dmp
        dmp.y0 = np.array(joint_start_final[0:6])
        dmp.goal = np.array(joint_end_final[0:6])

        y_track, dy_track, ddy_track = dmp.rollout()

        scoop_traj = y_track
        print(dmp.y0)
        print(dmp.goal)

        print()
        print(scoop_traj)
        return scoop_traj_ori, scoop_traj
    
    def go_to_scooping_init(self,input_traj): 
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        
        joint_goal = input_traj[0]
        #print(joint_goal)
        move_group.go(joint_goal, wait=True)
        move_group.set_max_velocity_scaling_factor(0.5)

        move_group.stop()

        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

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

    def plan_cartesian_path(self, input_traj):
        move_group = self.move_group
        df = input_traj
        pose_list = []
        for rows in range(df.shape[0]): 
            pose_temp = Pose()
            pose_temp.position.x,pose_temp.position.y,pose_temp.position.z = df.iloc[rows,0],df.iloc[rows,1],df.iloc[rows,2]
            pose_temp.orientation.x,pose_temp.orientation.y,pose_temp.orientation.z,pose_temp.orientation.w = df.iloc[rows,3],df.iloc[rows,4],df.iloc[rows,5],df.iloc[rows,6]
            pose_list.append(pose_temp)

        (plan, fraction) = move_group.compute_cartesian_path(
            pose_list, 0.01, jump_threshold=0.0  # waypoints to follow  # eef_step
        ) 
        return plan, fraction

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
        bowl_pose.pose.position.y = bowl_pos_y +0.07 
        bowl_pose.pose.position.z = bowl_pos_z
        bowl_name = "bowl"
        file_name = '/home/chi-onn/fyp_ws/src/meshes/bowl.STL'
        scene.add_mesh(bowl_name, bowl_pose,file_name, size=(0.001, 0.001, 0.001))

        self.bowl_name = bowl_name
        self.bowl_x, self.bowl_y = bowl_pose.pose.position.x,bowl_pose.pose.position.y
        return self.wait_for_state_update(object_is_known=True, timeout=timeout)
    
    def plot_traj(self, scoop_traj_ori, scoop_traj):
        scoop_traj_x_ori = scoop_traj_ori.iloc[:,0]
        scoop_traj_y_ori = scoop_traj_ori.iloc[:,1]
        scoop_traj_z_ori = scoop_traj_ori.iloc[:,2]
        
        scoop_traj_x = scoop_traj.iloc[:,0]
        scoop_traj_y = scoop_traj.iloc[:,1]
        scoop_traj_z = scoop_traj.iloc[:,2]
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
        plt.legend(["Original trajectory", "New trajectory when bowl is moved -10cm in x-axis"])
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
            msg_traj_point.time_from_start = rospy.Time(idx * 0.1)  # originally 0.5
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
        myrobot.add_spoon()
        myrobot.attach_spoon()
        #myrobot.go_to_detect_bowl()

        bowl_pos = [0.6542916139117372, -0.43426200282766514, 0.2646180732965431]
        # #offset to test different bowl pos
        # very off
        # bowl_pos[0] -= 0.2
        # bowl_pos[1] -= 0.1

        # # very off
        # bowl_pos[0] -= 0.1
        # bowl_pos[1] += 0.5

        # # # Insert bowl
        #myrobot.add_bowl(bowl_pos[0]-0.02,bowl_pos[1]+0.01,bowl_pos[2])
        myrobot.add_bowl(bowl_pos[0],bowl_pos[1],bowl_pos[2])


        # Plan & execute trajectory
        scoop_traj_ori, scoop_traj = myrobot.produce_scoop_dmp(bowl_pos[0],bowl_pos[1],bowl_pos[2])
        myrobot.go_to_scooping_init(scoop_traj) 
        result = myrobot.moveit_execute_traj_client(scoop_traj)

        # # #input("Execute scoop by pressing 'Enter'")
        #myrobot.plot_traj(scoop_traj_ori, scoop_traj)

        myrobot.go_to_mouth_pos() 

        

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()