#!/usr/bin/env python3

from trajopt import ctrajoptpy as trajoptpy
import json
import rospy
import pandas as pd
import numpy as np
import copy
import pydmps
import math
import quaternion
import moveit_commander
import sys

class trajopt_init(object):
    def __init__(self):
        #rospy.init_node("trajopt_test")
        #self.df = pd.read_csv('/home/chi-onn/fyp_ws/src/scooping/traj_files/scooping_final_pose.csv',header=None)
        self.num_waypts = 41


    def produce_scoop_dmp(self,bowl_pos_x, bowl_pos_y, bowl_pos_z):
        df = pd.read_csv('/home/chi-onn/fyp_ws/src/scooping/traj_files/scooping_final_pose.csv',header=None)
        df_copy = copy.deepcopy(df)

        df_copy = np.transpose(df_copy)
        y_des = df_copy.to_numpy()

        # test normal run
        dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=7, n_bfs=500, dt = 0.025, ay=np.ones(7) * 10.0)
        y_track = []
        dy_track = []
        ddy_track = []

        dmp.imitate_path(y_des=y_des, plot=False)
        dmp.reset_state() 
        # original bowl position = [0.6542916139117372, -0.43426200282766514, 0.2646180732965431]
        dmp.y0 -= np.array([0.6542916139117372-bowl_pos_x, -0.43426200282766514-bowl_pos_y, 0.2646180732965431-bowl_pos_z, 0.0, 0.0, 0.0, 0.0])
        dmp.goal -= np.array([0.6542916139117372-bowl_pos_x, -0.43426200282766514-bowl_pos_y, 0.2646180732965431-bowl_pos_z, 0.0, 0.0, 0.0, 0.0])
        y_track, dy_track, ddy_track = dmp.rollout()

        raw_scoop = []
        for row in range(y_track.shape[0]):
            normalised_final = y_track[row,:]
            norm = math.sqrt(sum(np.square(y_track[row,3:7])))
            normalised = y_track[row,3:7]/norm
            normalised_final[3:7] = normalised
            raw_scoop.append(normalised_final)

        scoop_traj = raw_scoop
        #scoop_traj = raw_scoop[:3]
        #print(scoop_traj)

        return scoop_traj
    
    def get_cart_costs_and_constraints(self, cart_traj, costs=False):
            """
            Converts the desiredtrajectory in cartesian space into cartesian pose costs or constraints
            in trajopt.
            Args:
                traj_seed ([float]): Trajectory in cartesian space to be executed
            Returns:
                constraint ([json]): Constraint or cost to be optimized
            """
            # Note this is termed as constraint, but can also be used as a cost
            constraint = []
            for idx, pt in enumerate(cart_traj):
                if idx >= self.num_waypts:
                    rospy.logwarn("Too many points in trajectory!")
                    print(len(cart_traj), self.num_waypts, "\n\n")
                    break
                # setting coeffs
                if costs:
                    pos_coeff = 10
                    rot_coeff = 5
                else:
                    # if constraints, set higher coeffs for last waypoint
                    if idx == self.num_waypts - 1:
                        pos_coeff = 5
                        rot_coeff = 1
                    else:
                        pos_coeff = 1
                        rot_coeff = 0
                # check that quaternions are normalized
                #q = np.quaternion(pt[3], pt[4], pt[5], pt[6])   
                norm = math.sqrt(sum(np.square([pt[3], pt[4], pt[5], pt[6]])))
                normalised = np.array([pt[3], pt[4], pt[5], pt[6]])/norm

                #normalised = [pt[3], pt[4], pt[5], pt[6]]/norm
                cnst_dict = {
                    "name": "waypt_cart_" + str(idx),
                    "type": "cart_pose",
                    "params": {
                        "timestep": idx,
                        "xyz" : [pt[0], pt[1], pt[2]],
                        # "wxyz" : [q.w, q.x, q.y, q.z],
                        #"wxyz" : [1., 0., 0., 0.],
                        "wxyz" : [normalised[3], normalised[0], normalised[1], normalised[2]],
                        "link": "link_tcp",
                        "rot_coeffs": [rot_coeff, rot_coeff, rot_coeff],    # ignore rotation for now
                        "pos_coeffs" : [pos_coeff, pos_coeff, pos_coeff]
                    }
                }
                constraint.append(cnst_dict)
                #constraint__json = json.dumps(constraint)
            return constraint
    
    def make_request(self,costs):
        # costs = self.get_cart_costs_and_constraints(cart_traj, costs=True)
        # Joint velocity cost may sometimes make the trajectory weird...
        joint_vel_cost = {
                            "type": "joint_vel",
                            "params": {
                                "targets": [0.2],
                                "coeffs": [2]
                            }
                        }
        costs.append(joint_vel_cost)
        request = {
            "basic_info": {
                "n_steps": self.num_waypts,
                "manip" : "xarm6",
                "start_fixed" : True,
                "max_iter" : 40
            },
            "costs": costs,
            "init_info": {
                "type": "stationary",
                #"data": init_waypts.tolist()
            }
        }
        
        return request
    
    def get_current_joint_state(self):
        moveit_commander.roscpp_initialize(sys.argv)
        move_group = moveit_commander.MoveGroupCommander('xarm6')
        joint_state = move_group.get_current_joint_values()
        return joint_state

def main():
    try:
        # initialising trajopt object
        print(trajoptpy.greet())
        trajopt_obj = trajopt_init()

        # generate scooping traj
        bowl_pos = [0.6542916139117372, -0.43426200282766514, 0.2646180732965431]
        scoop_traj = trajopt_obj.produce_scoop_dmp(bowl_pos[0], bowl_pos[1], bowl_pos[2])

        # produce cost from scooping traj
        costs = trajopt_obj.get_cart_costs_and_constraints(scoop_traj, costs=True)
        request = trajopt_obj.make_request(costs)
        request_json = json.dumps(request)
        #print(request)

        #initialise problem
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        prob = trajoptpy.ConstructProblem(request_json, trajopt_obj.get_current_joint_state(), joint_names)
        
        result = trajoptpy.OptimizeProblem(prob)
        #print(result)
        # print(result.GetTraj())

        test = result.GetTraj()
        print(test)
        #print(test[0])
        print("Finished")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()

        


# ###J-ANNE CODE ###
# costs = self.get_cart_costs_and_constraints(cart_traj, costs=True)
#             # Joint velocity cost may sometimes make the trajectory weird...
#             joint_vel_cost = {
#                                 "type": "joint_vel",
#                                 "params": {
#                                     "targets": [0.2],
#                                     "coeffs": [2]
#                                 }
#                             }
#             costs.append(joint_vel_cost)
#             request = {
#                 "basic_info": {
#                     "n_steps": self.num_waypts,
#                     "manip" : "xarm6",
#                     "start_fixed" : True,
#                     "max_iter" : self.MAX_ITER
#                 },
#                 "costs": costs,
#                 "init_info": {
#                     "type": "given_traj",
#                     "data": init_waypts.tolist()
#                 }
#             }

# def get_cart_costs_and_constraints(self, cart_traj_seed, costs=False):
#         """
#         Converts a (seed) trajectory in cartesian space into cartesian pose costs or constraints
#         in trajopt.
#         Args:
#             traj_seed ([float]): Trajectory in cartesian space to be executed
#         Returns:
#             constraint ([json]): Constraint or cost to be optimized
#         """
#         # Note this is termed as constraint, but can also be used as a cost
#         constraint = []
#         for idx, pt in enumerate(cart_traj_seed):
#             if idx >= self.num_waypts:
#                 rospy.logwarn("Too many points in trajectory!")
#                 print(len(cart_traj_seed), self.num_waypts, "\n\n")
#             # setting coeffs
#             if costs:
#                 pos_coeff = 10
#                 rot_coeff = 5
#             else:
#                 # if constraints, set higher coeffs for last waypoint
#                 if idx == self.num_waypts - 1:
#                     pos_coeff = 5
#                     rot_coeff = 1
#                 else:
#                     pos_coeff = 1
#                     rot_coeff = 0
#             # check that quaternions are normalized
#             q = np.quaternion(pt[3], pt[4], pt[5], pt[6])   
#             cnst_dict = {
#                 "name": "waypt_cart_" + str(idx),
#                 "type": "cart_pose",
#                 "params": {
#                     "timestep": idx,
#                     "xyz" : [pt[0], pt[1], pt[2]],
#                     # "wxyz" : [q.w, q.x, q.y, q.z],
#                     "wxyz" : [1., 0., 0., 0.],
#                     "link": "link_tcp",
#                     "rot_coeffs": [rot_coeff, rot_coeff, rot_coeff],    # ignore rotation for now
#                     "pos_coeffs" : [pos_coeff, pos_coeff, pos_coeff]
#                 }
#             }
#             constraint.append(cnst_dict)
#         return constraint