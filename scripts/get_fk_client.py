#!/usr/bin/env python

import rospy
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse
from sensor_msgs.msg import JointState

class GetFK(object):
    def __init__(self, fk_link, frame_id):
        """
        A class to do FK calls thru the MoveIt!'s /compute_ik service.
        :param str fk_link: link to compute the forward kinematics
        :param str frame_id: frame_id to compute the forward kinematics
        into account collisions
        """
        rospy.loginfo("Initalizing GetFK...")
        self.fk_link = fk_link
        self.frame_id = frame_id
        rospy.loginfo("Asking forward kinematics for link: " + self.fk_link)
        rospy.loginfo("PoseStamped answers will be on frame: " + self.frame_id)
        self.fk_srv = rospy.ServiceProxy('/compute_fk',
                                         GetPositionFK)
        rospy.loginfo("Waiting for /compute_fk service...")
        self.fk_srv.wait_for_service()
        rospy.loginfo("Connected!")
        self.last_js = None
        self.js_sub = rospy.Subscriber('/joint_states',
                                       JointState,
                                       self.js_cb,
                                       queue_size=1)

    def js_cb(self, data):
        self.last_js = data

    def get_current_fk_pose(self):
        resp = self.get_current_fk()
        if len(resp.pose_stamped) >= 1:
            return resp.pose_stamped[0]
        return None

    def get_current_fk(self):
        while not rospy.is_shutdown() and self.last_js is None:
            rospy.logwarn("Waiting for a /joint_states message...")
            rospy.sleep(0.1)
        return self.get_fk(self.last_js)

    def get_fk(self, joint_state, fk_link=None, frame_id=None):
        """
        Do an FK call to with.
        :param sensor_msgs/JointState joint_state: JointState message
            containing the full state of the robot.
        :param str or None fk_link: link to compute the forward kinematics for.
        """
        if fk_link is None:
            fk_link = self.fk_link

        req = GetPositionFKRequest()
        req.header.frame_id = 'world'
        req.fk_link_names = [self.fk_link]
        req.robot_state.joint_state = joint_state
        try:
            resp = self.fk_srv.call(req)
            return resp
        except rospy.ServiceException as e:
            rospy.logerr("Service exception: " + str(e))
            resp = GetPositionFKResponse()
            resp.error_code = 99999  # Failure
            return resp

    def get_fk_list(self,joint_angles):
        self.last_js.position = joint_angles
        #print(self.last_js)
        return self.get_fk(self.last_js)


if __name__ == '__main__':
    rospy.init_node('test_fk')
    rospy.loginfo("Querying for FK")
    gfk = GetFK('link_tcp', 'world')
    rospy.sleep(3)
#start position: [-0.9089088311249498, 0.03410598107548042, -1.2525185671915444, 0.9727541635706638, 1.283624716998539, 1.7484555360801215, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# end position: [-0.9828529332799439, 0.13523809889799124, -1.1753221308831512, 1.1355620835818918, 1.6072786596129207, 2.559293508529663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 
    joint_angles1 = [-1.0480408162664134, 0.15543593365920078, -1.1250075390298548, 1.2816778111474771, 1.3947116701334321, 2.5166345467183246, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    resp = gfk.get_fk_list(joint_angles1)
    #from moveit_python_tools.friendly_error_codes import moveit_error_dict
    #rospy.loginfo(moveit_error_dict[resp.error_code.val])
    posestamped = resp.pose_stamped[0]
    position = (posestamped.pose.position.x, posestamped.pose.position.y, posestamped.pose.position.z)
    orientation = (posestamped.pose.orientation.x,
               posestamped.pose.orientation.y,
               posestamped.pose.orientation.z,
               posestamped.pose.orientation.w)
    print(position)
    print(orientation)
