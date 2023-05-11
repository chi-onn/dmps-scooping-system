## Overview
---
DMP-based scooping system. 
Designed to adapt to the following situations:-
1. Different bowl position
2. Different bowl width
3. Scoop in different directions

## Installation
---

### Dependencies
- [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
- [pydmps](https://github.com/studywolf/pydmps)
    - `pip install pydmp`
- [perception_main](https://github.com/janneyow/perception_main)
- [trajopt](https://github.com/janneyow/trajopt)
- [xarm](https://github.com/janneyow/xarm)

## Usage
---

### For simulation only (Rviz)
```bash
roslaunch xarm6_gripper_moveit_config demo.launch

# in a new terminal
python scoop_trajopt_spoon_sim.py 
```

Configurations:

1. Offset-place bowl in different position in rviz (line 608)

2. Length-varying - use arg 'short' or 'long' in func change_scoop_length() (line 302)

3. Modified - Use any modified variation available (line 217 onwards). Note that you also need to uncomment from lines 246-270.

### For actual robot arm
```bash
roslaunch xarm6_gripper_moveit_config eye_in_hand.launch

# in a new terminal
conda activate fyp
roslaunch perception perception_node.launch

# in another new terminal
python scoop_trajop_spoon.py
```


### File descriptions (If file doesn't have the word spoon, its using the eef traj)
- scoop_trajopt_spoon_sim.py - Cartesian space dmps, uses trajopt, all variations.

- scoop_trajopt_spoon.py - Actual file ran during experiments

- pose_transform.py - Contains code to rotate start/goal position, eef to spoon traj conversion (& vice-versa), length-varying variation code

- get_fk_client - Contains function to call compute_fk service

- get_ik_client - Contains function to call compute_ik service

## Additional info
---
### Things to take note
1. Spoon direction
- Spoon should be facing right (towards wheelchair)
- Spoon mount should ideally be used for a fixed offset (CAD file - TODO get from Sam)
- Spoon currently scoops in a sweeping manner instead of a shovelling manner

2. DMPs
- Trajectory end effector should be of the spoon, NOT the robot (currently a fixed translation is used)
- Changes should be made to DMP's LEARNT trajectory's goal (`dmp.y`, `dmp.goal`)
    - Weights can also be changed if needed (`dmp.w`)

3. Trajectory optimization is done by Trajopt

4. Perception of bowl is not accurate
- Affects start position of trajectory

5. Food types tested
- Mashed potatoes (more watery better)
- Foam balls

6. Modified DMP
- Requires joint start and joint end positions

### Additional resources
- dmp_orientation folder contains attempts using 2nd dmp implementation ([movement_primitives](https://github.com/dfki-ric/movement_primitives)) add link
