#!/usr/bin/env python3
import rospy
import math

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path


class UAVRvizTracker:
    def __init__(self):
        rospy.init_node("uav_rviz_tracker")

        self.uav_model_name = rospy.get_param("~uav_model_name", "uav")
        self.frame_id = rospy.get_param("~frame_id", "world")

        self.uav_marker_pub = rospy.Publisher(
            "/uav/current_marker",
            Marker,
            queue_size=1
        )

        self.uav_path_pub = rospy.Publisher(
            "/uav/actual_path",
            Path,
            queue_size=1
        )

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id

        rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_states_callback
        )

        rospy.loginfo("uav_rviz_tracker started.")
        rospy.spin()

    def model_states_callback(self, msg):
        if rospy.is_shutdown():
            return
        if self.uav_model_name not in msg.name:
            rospy.logwarn_throttle(
                1.0,
                "Cannot find UAV model name: %s" % self.uav_model_name
            )
            return

        idx = msg.name.index(self.uav_model_name)
        pose = msg.pose[idx]

        x = pose.position.x
        y = pose.position.y
        z = pose.position.z

        # Publish UAV marker
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()

        marker.ns = "uav"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z + 0.2
        marker.pose.orientation = pose.orientation

        marker.scale.x = 0.8   # arrow length
        marker.scale.y = 0.2   # arrow width
        marker.scale.z = 0.2   # arrow height

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # self.uav_marker_pub.publish(marker)

        # Publish actual path
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.frame_id
        pose_stamped.header.stamp = rospy.Time.now()

        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation = pose.orientation

        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.poses.append(pose_stamped)

        # 避免 path 無限變長
        if len(self.path_msg.poses) > 500:
            self.path_msg.poses.pop(0)

        try:
            if not rospy.is_shutdown():
                self.uav_marker_pub.publish(marker)

            if not rospy.is_shutdown():
                self.uav_path_pub.publish(self.path_msg)

        except rospy.ROSException:
            return


if __name__ == "__main__":
    UAVRvizTracker()