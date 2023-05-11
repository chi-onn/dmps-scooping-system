## Overview
DMP-based scooping system. 
Designed to adapt to the following situations:-
1. Different bowl position
2. Different bowl width
3. Scoop in different directions

## Installation

### Dependencies
- detectron2, pydmps, perception, trajopt

## Usage

### For simulation only (Rviz)
1. run 'roslaunch xarm6_gripper_moveit_config demo.launch' in terminal
2. run scoop_trajopt_spoon_sim.py python file

To test variations:

    Offset-place bowl in different position in rviz (line 608)

    Length-varying - use arg 'short' or 'long' in func change_scoop_length() (line 302)

    Modified - Use any modified variation available (line 217 onwards). Note that you also need to uncomment from lines 246-270.

### For actual robot arm
1. run 'roslaunch xarm6_gripper_moveit_config eye_in_hand.launch' in 1 terminal.
2. run 'conda activate fyp' in another terminal. This is for the terminal we use to run perception node.
3. in fyp environment terminal opened earlier, run 'roslaunch perception perception_node.launch'


### File descriptions (If file doesn't have the word spoon, its using the eef traj)
- scoop_trajopt_spoon_sim.py - Cartesian space dmps, uses trajopt, all variations.

- scoop_trajopt_spoon.py - Actual file ran during experiments

- pose_transform.py - Contains code to rotate start/goal position, eef to spoon traj conversion (& vice-versa), length-varying variation code

- get_fk_client - Contains function to call compute_fk service

- get_ik_client - Contains function to call compute_ik service

### Additional info
dmp_orientation folder contains attempts using 2nd dmp implementation (movement_primitives pkg)
