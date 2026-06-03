#!/usr/bin/env python3
import rospy
import math

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from std_msgs.msg import Int32


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
        # 降低 RViz 更新頻率，避免 Gazebo model_states 太高頻造成 RViz delay
        self.last_pub_time = rospy.Time(0)
        self.pub_period = rospy.Duration(0.1)  # 10 Hz
        # actual path 不要太密
        self.max_path_points = 500
        self.path_min_dist = 0.05
        self.last_path_x = None
        self.last_path_y = None
        self.last_path_z = None

        # 目前 flight mode，用來切 mode 時清 actual path
        self.active_mode = -1

        rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_states_callback
        )
        rospy.Subscriber(
            "/planner/active_mode",
            Int32,
            self.active_mode_callback
        )

        rospy.loginfo("uav_rviz_tracker started.")
        rospy.spin()

    def active_mode_callback(self, msg):
        new_mode = msg.data

        if new_mode != self.active_mode:
            self.active_mode = new_mode

            # 切 mode 時清掉 UAV 實際飛行路徑，避免 offline/online 混在一起
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.frame_id

            self.last_path_x = None
            self.last_path_y = None
            self.last_path_z = None

            rospy.loginfo("UAV RViz tracker: active_mode=%d, actual path cleared" % self.active_mode)

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
        now = rospy.Time.now()

        if now - self.last_pub_time < self.pub_period:
            return

        self.last_pub_time = now

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

        append_path = False

        if self.last_path_x is None:
            append_path = True
        else:
            dx = x - self.last_path_x
            dy = y - self.last_path_y
            dz = z - self.last_path_z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist > self.path_min_dist:
                append_path = True

        if append_path:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.frame_id
            pose_stamped.header.stamp = now

            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = z
            pose_stamped.pose.orientation = pose.orientation

            self.path_msg.header.stamp = now
            self.path_msg.poses.append(pose_stamped)

            self.last_path_x = x
            self.last_path_y = y
            self.last_path_z = z

            # 避免 path 無限變長
            if len(self.path_msg.poses) > self.max_path_points:
                self.path_msg.poses = self.path_msg.poses[-self.max_path_points:]

        try:
            if not rospy.is_shutdown():
                self.uav_marker_pub.publish(marker)

            if not rospy.is_shutdown():
                self.uav_path_pub.publish(self.path_msg)

        except rospy.ROSException:
            return


if __name__ == "__main__":
    UAVRvizTracker()