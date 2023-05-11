#!/usr/bin/env python

import moveit_msgs
import rospy

def moveit_execute_traj_client(self, joint_traj):
        """
        Calls the moveit action server to execute the robot trajectory specified
        Args:
            joint_traj ([float]): Waypoints of trajectory in joint space
        """
        client = actionlib.SimpleActionClient("/execute_trajectory", ExecuteTrajectoryAction)
        client.wait_for_server()
        goal = ExecuteTrajectoryGoal()
        goal.trajectory.joint_trajectory.header.frame_id = self.frame_id
        goal.trajectory.joint_trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_trajectory.joint_names = self.env.robot.actuated_arm_joint_names
        for idx, joint_poses in enumerate(joint_traj):
            msg_traj_point = JointTrajectoryPoint()
            msg_traj_point.positions = joint_poses
            # Assumes velocity fixed
            msg_traj_point.time_from_start = rospy.Time(idx * 0.5)
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


# J-ANNE CODE SNIPPET
# def moveit_execute_traj_client(self, joint_traj):
#         """
#         Calls the moveit action server to execute the robot trajectory specified
#         Args:
#             joint_traj ([float]): Waypoints of trajectory in joint space
#         """
#         client = actionlib.SimpleActionClient("/execute_trajectory", ExecuteTrajectoryAction)
#         client.wait_for_server()
#         goal = ExecuteTrajectoryGoal()
#         goal.trajectory.joint_trajectory.header.frame_id = self.frame_id
#         goal.trajectory.joint_trajectory.header.stamp = rospy.Time.now()
#         goal.trajectory.joint_trajectory.joint_names = self.env.robot.actuated_arm_joint_names
#         for idx, joint_poses in enumerate(joint_traj):
#             msg_traj_point = JointTrajectoryPoint()
#             msg_traj_point.positions = joint_poses
#             # Assumes velocity fixed
#             msg_traj_point.time_from_start = rospy.Time(idx * 0.5)
#             goal.trajectory.joint_trajectory.points.append(msg_traj_point)
#         client.send_goal(goal)
#         client.wait_for_result()
#         result = client.get_result()
#         if result.error_code.val != 1:
#             moveit_error_dict = {}
#             for name in MoveItErrorCodes.__dict__.keys():
#                 if not name[:1] == '_':
#                     code = MoveItErrorCodes.__dict__[name]
#                     moveit_error_dict[code] = name
#             rospy.logwarn(f"Execution failed with error code {result.error_code.val}:  {moveit_error_dict[result.error_code.val]}")
#         self.reached_goal =True
#         return result