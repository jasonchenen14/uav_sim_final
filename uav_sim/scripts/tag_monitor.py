#!/usr/bin/env python3

import rospy
import math
from apriltag_ros.msg import AprilTagDetectionArray
def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def cb(msg):
    if len(msg.detections) == 0:
        rospy.logwarn_throttle(1.0, "No tag detected")
        return

    det = msg.detections[0]
    pose = det.pose.pose.pose
    p = pose.position
    q = pose.orientation
    yaw = quat_to_yaw(q)
    yaw_deg = math.degrees(yaw)

    rospy.loginfo(
        "tag id: %d | x: %.3f, y: %.3f, z: %.3f | yaw: %.3f rad, %.2f deg"
        % (det.id[0], p.x, p.y, p.z, yaw, yaw_deg)
    )


rospy.init_node("tag_monitor")
rospy.Subscriber("/tag_detections", AprilTagDetectionArray, cb)
rospy.spin()