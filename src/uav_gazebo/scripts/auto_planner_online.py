#!/usr/bin/env python3
import rospy
import math
import numpy as np

from scipy.optimize import minimize

# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32MultiArray, Int32, Bool
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
# from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion


# =========================
# Planner parameters
# =========================
START_POS = (0.0, 0.0)
FLIGHT_Z = 1.5

SAFETY_MARGIN = 0.6
MAX_WAYPOINT_DIST = 2.0

CYL_LENGTH = 2.0


# =========================
# Original helper functions
# =========================
def subdivide_waypoints(waypoints, max_dist=2.0):
    new_wps = [waypoints[0]]

    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if dist > max_dist:
            num_segments = int(math.ceil(dist / max_dist))
            for j in range(1, num_segments):
                t = j / num_segments
                nx = p1[0] + t * (p2[0] - p1[0])
                ny = p1[1] + t * (p2[1] - p1[1])
                new_wps.append((nx, ny))

        new_wps.append(p2)

    return new_wps


def dist_point_to_segment(p, p1, p2):
    x, y = p
    x1, y1 = p1
    x2, y2 = p2

    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D

    if len_sq == 0:
        return math.hypot(x - x1, y - y1)

    param = dot / len_sq

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    return math.hypot(x - xx, y - yy)


def greedy_los_planner(start, target, obstacles, safety_margin=0.8):
    waypoints = [start]
    current = start
    visited = [start]

    while math.hypot(target[0] - current[0], target[1] - current[1]) > 0.1:
        blocking_obs = None
        min_dist_to_obs = float("inf")

        for obs in obstacles:
            dist_from_current = math.hypot(
                obs["x"] - current[0],
                obs["y"] - current[1]
            )

            if dist_from_current <= obs["r"] + safety_margin + 0.01:
                continue

            dist_to_line = dist_point_to_segment(
                (obs["x"], obs["y"]),
                current,
                target
            )

            if dist_to_line < obs["r"] + safety_margin:
                if dist_from_current < min_dist_to_obs:
                    min_dist_to_obs = dist_from_current
                    blocking_obs = obs

        if blocking_obs is None:
            waypoints.append(target)
            break

        cx = blocking_obs["x"]
        cy = blocking_obs["y"]

        vx = cx - current[0]
        vy = cy - current[1]
        v_len = math.hypot(vx, vy)

        if v_len == 0:
            vx, vy, v_len = 1.0, 0.0, 1.0

        nx = -vy / v_len
        ny = vx / v_len

        R = blocking_obs["r"] + safety_margin

        wp1 = (cx + nx * R, cy + ny * R)
        wp2 = (cx - nx * R, cy - ny * R)

        safe_cands = []

        for wp in [wp1, wp2]:
            is_safe = True

            for obs in obstacles:
                if obs != blocking_obs:
                    d = math.hypot(wp[0] - obs["x"], wp[1] - obs["y"])
                    if d < obs["r"] + 0.2:
                        is_safe = False

            for v in visited:
                if math.hypot(wp[0] - v[0], wp[1] - v[1]) < 0.2:
                    is_safe = False

            if is_safe:
                safe_cands.append(wp)

        if not safe_cands:
            rospy.logwarn("No safe waypoint found. Planner stopped.")
            break

        best_wp = min(
            safe_cands,
            key=lambda w: math.hypot(target[0] - w[0], target[1] - w[1])
        )

        waypoints.append(best_wp)
        visited.append(best_wp)
        current = best_wp

    return waypoints


# =========================
# Minimum snap
# =========================
def get_Q_matrix(T):
    Q = np.zeros((8, 8))

    Q[4, 4] = 576 * T
    Q[4, 5] = 1440 * T**2
    Q[4, 6] = 2880 * T**3
    Q[4, 7] = 5040 * T**4

    Q[5, 4] = 1440 * T**2
    Q[5, 5] = 4800 * T**3
    Q[5, 6] = 10800 * T**4
    Q[5, 7] = 20160 * T**5

    Q[6, 4] = 2880 * T**3
    Q[6, 5] = 10800 * T**4
    Q[6, 6] = 25920 * T**5
    Q[6, 7] = 50400 * T**6

    Q[7, 4] = 5040 * T**4
    Q[7, 5] = 20160 * T**5
    Q[7, 6] = 50400 * T**6
    Q[7, 7] = 100800 * T**7

    return Q


def evaluate_poly(c, t, derivative=0):
    if derivative == 0:
        return np.dot(c, [1, t, t**2, t**3, t**4, t**5, t**6, t**7])
    elif derivative == 1:
        return np.dot(c, [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6])
    elif derivative == 2:
        return np.dot(c, [0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5])
    elif derivative == 3:
        return np.dot(c, [0, 0, 0, 6, 24*t, 60*t**2, 120*t**3, 210*t**4])

    return 0


def optimize_1d(waypoints, times):
    n_seg = len(waypoints) - 1

    def obj(x):
        total = 0.0
        for i in range(n_seg):
            c = x[i * 8:(i + 1) * 8]
            total += c.T @ get_Q_matrix(times[i]) @ c
        return total

    cons = []

    cons.extend([
        {"type": "eq", "fun": lambda x: evaluate_poly(x[0:8], 0, 0) - waypoints[0]},
        {"type": "eq", "fun": lambda x: evaluate_poly(x[0:8], 0, 1)},
        {"type": "eq", "fun": lambda x: evaluate_poly(x[0:8], 0, 2)},

        {"type": "eq", "fun": lambda x: evaluate_poly(x[-8:], times[-1], 0) - waypoints[-1]},
        {"type": "eq", "fun": lambda x: evaluate_poly(x[-8:], times[-1], 1)},
        {"type": "eq", "fun": lambda x: evaluate_poly(x[-8:], times[-1], 2)}
    ])

    for i in range(n_seg - 1):
        T = times[i]
        wp = waypoints[i + 1]

        c_cur = lambda x, i=i: x[i * 8:(i + 1) * 8]
        c_next = lambda x, i=i: x[(i + 1) * 8:(i + 2) * 8]

        cons.extend([
            {"type": "eq", "fun": lambda x, c=c_cur, T=T, wp=wp:
                evaluate_poly(c(x), T, 0) - wp},

            {"type": "eq", "fun": lambda x, c=c_next, wp=wp:
                evaluate_poly(c(x), 0, 0) - wp},

            {"type": "eq", "fun": lambda x, c1=c_cur, c2=c_next, T=T:
                evaluate_poly(c1(x), T, 1) - evaluate_poly(c2(x), 0, 1)},

            {"type": "eq", "fun": lambda x, c1=c_cur, c2=c_next, T=T:
                evaluate_poly(c1(x), T, 2) - evaluate_poly(c2(x), 0, 2)},

            {"type": "eq", "fun": lambda x, c1=c_cur, c2=c_next, T=T:
                evaluate_poly(c1(x), T, 3) - evaluate_poly(c2(x), 0, 3)}
        ])

    result = minimize(
        obj,
        np.zeros(8 * n_seg),
        method="SLSQP",
        constraints=cons,
        options={"maxiter": 2000}
    )

    if not result.success:
        rospy.logwarn("Minimum snap optimization failed: %s", result.message)

    return result.x


def sample_min_snap_path(coeffs_x, coeffs_y, times, samples_per_seg=50):
    points = []

    for i in range(len(times)):
        cx = coeffs_x[i * 8:(i + 1) * 8]
        cy = coeffs_y[i * 8:(i + 1) * 8]

        for t in np.linspace(0.0, times[i], samples_per_seg):
            x = evaluate_poly(cx, t, 0)
            y = evaluate_poly(cy, t, 0)
            points.append((x, y))

    return points


# =========================
# ROS visualization
# =========================
def make_path_msg(points):
    path = Path()
    path.header.frame_id = "world"
    path.header.stamp = rospy.Time.now()

    for x, y in points:
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = rospy.Time.now()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = FLIGHT_Z
        pose.pose.orientation.w = 1.0

        path.poses.append(pose)

    return path


def make_line_marker(points, ns, marker_id, r, g, b, width):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()

    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.scale.x = width

    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = 1.0

    marker.pose.orientation.w = 1.0

    for x, y in points:
        p = Point()
        p.x = x
        p.y = y
        p.z = FLIGHT_Z
        marker.points.append(p)

    return marker


def make_waypoint_markers(waypoints):
    marker_array = MarkerArray()

    for i, (x, y) in enumerate(waypoints):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "waypoints"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = FLIGHT_Z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker_array.markers.append(marker)

    return marker_array


def make_obstacle_markers(obstacles):
    marker_array = MarkerArray()

    for i, obs in enumerate(obstacles):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "obstacles"
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = obs["x"]
        marker.pose.position.y = obs["y"]
        marker.pose.position.z = CYL_LENGTH / 2.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = obs["r"] * 2.0
        marker.scale.y = obs["r"] * 2.0
        marker.scale.z = CYL_LENGTH

        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.8

        marker_array.markers.append(marker)

    return marker_array


# =========================
# AutoPlanner Node
# =========================
class AutoPlannerNode:
    def __init__(self):
        self.obstacles = None
        self.target = None
        self.has_planned = False
        self.latest_scan = None
        self.uav_pose_ready = False

        self.uav_x = 0.0
        self.uav_y = 0.0
        self.uav_yaw = 0.0

        self.uav_model_name = "uav"

        self.last_plan_time = rospy.Time(0)
        self.plan_period = rospy.Duration(1.0)

        self.lidar_max_use_range = 6.0
        self.lidar_min_use_range = 0.25
        self.lidar_front_angle = math.radians(180.0)
        self.lidar_bin_angle = math.radians(5.0)

        self.lidar_obstacle_radius = 0.35
        self.has_online_plan = False
        # 只有前方這個距離內有障礙物才重規劃
        self.front_block_threshold = 1.75
        self.obstacle_memory = []
        self.obstacle_memory_merge_dist = 1.0
        self.obstacle_memory_max_age = rospy.Duration(10.0)

        # 判斷 blocked 用的前方角度，建議比 scan_to_obstacles 更窄
        self.front_block_angle = math.radians(90.0)

        self.obstacle_pub = rospy.Publisher(
            "/planner/online/obstacles_marker",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.waypoint_pub = rospy.Publisher(
            "/planner/online/waypoints_marker",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.los_marker_pub = rospy.Publisher(
            "/planner/online/los_marker",
            Marker,
            queue_size=1,
            latch=True
        )

        self.min_snap_marker_pub = rospy.Publisher(
            "/planner/online/min_snap_marker",
            Marker,
            queue_size=1,
            latch=True
        )

        self.los_path_pub = rospy.Publisher(
            "/planner/online/los_path",
            Path,
            queue_size=1,
            latch=True
        )

        self.min_snap_path_pub = rospy.Publisher(
            "/planner/online/min_snap_path",
            Path,
            queue_size=1,
            latch=True
        )

        rospy.Subscriber(
            "/lidar_360/scan",
            LaserScan,
            self.scan_callback
        )

        # rospy.Subscriber(
        #     "/gazebo/model_states",
        #     ModelStates,
        #     self.model_states_callback
        # )  0
        rospy.Subscriber(
            "/planner/uav_state_xyyaw",
            Float32MultiArray,
            self.uav_state_callback
        )#0

        rospy.Subscriber(
            "/planner/target_xy",
            Float32MultiArray,
            self.target_callback
        )
        rospy.Subscriber(
            "/planner/online/replan_start",
            Float32MultiArray,
            self.replan_start_callback
        )

        self.plan_timer = rospy.Timer(
            rospy.Duration(1.0),
            self.online_plan_timer
        )
        self.active_mode = -1
        self.online_enabled = False

        rospy.Subscriber(
            "/planner/active_mode",
            Int32,
            self.active_mode_callback
        )
        self.depth_topic = rospy.get_param(
            "~depth_topic",
            "/depth_camera/depth/image_raw"
        )

        self.depth_enabled = False
        self.depth_affects_planner = False
        self.depth_blocked = False
        self.depth_min = float("inf")
        self.depth_mean = float("inf")
        self.depth_close_ratio = 0.0
        self.depth_last_time = rospy.Time(0)
        self.depth_roi_w_ratio = 0.35
        self.depth_roi_h_ratio = 0.35
        self.depth_trigger_distance = 0.2
        self.depth_trigger_ratio = 0.8    
        self.depth_timeout = rospy.Duration(0.5)
        rospy.Subscriber(
            self.depth_topic,
            Image,
            self.depth_callback
        )

        self.depth_blocked_pub = rospy.Publisher(
            "/planner/online/depth_camera_blocked",
            Bool,
            queue_size=1
        )

    def active_mode_callback(self, msg):
        self.active_mode = msg.data
        self.online_enabled = (self.active_mode == 9)

        if not self.online_enabled:
            self.has_online_plan = False
            self.obstacle_memory = []

    def obstacle_callback(self, msg):
        data = list(msg.data)

        if len(data) % 3 != 0:
            rospy.logwarn("Invalid obstacle_list length: %d", len(data))
            return

        obstacles = []

        for i in range(0, len(data), 3):
            obstacles.append({
                "type": "cylinder",
                "x": float(data[i]),
                "y": float(data[i + 1]),
                "r": float(data[i + 2])
            })

        self.obstacles = obstacles
        rospy.loginfo("Received %d obstacles", len(self.obstacles))
        # self.try_plan()

    def target_callback(self, msg):
        data = list(msg.data)

        if len(data) < 2:
            rospy.logwarn("Invalid target_xy")
            return

        self.target = (float(data[0]), float(data[1]))
        rospy.loginfo("Received target: %s", str(self.target))
        # self.try_plan()

    def scan_callback(self, msg):
        self.latest_scan = msg

    def plan_from_start(self, start):
        if self.obstacles is None or self.target is None:
            rospy.logwarn("Cannot plan yet: obstacles or target not received.")
            return

        rospy.loginfo(
            "Start planning from current UAV position: start=(%.2f, %.2f), target=(%.2f, %.2f)"
            % (start[0], start[1], self.target[0], self.target[1])
        )

        waypoints = greedy_los_planner(
            start,
            self.target,
            self.obstacles,
            safety_margin=SAFETY_MARGIN
        )

        waypoints = subdivide_waypoints(
            waypoints,
            max_dist=MAX_WAYPOINT_DIST
        )

        if len(waypoints) < 2:
            rospy.logerr("Planning failed: waypoint number < 2")
            return

        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]

        times = []
        for i in range(len(waypoints) - 1):
            dist = math.hypot(
                wp_x[i + 1] - wp_x[i],
                wp_y[i + 1] - wp_y[i]
            )
            times.append(max(1.5, dist / 1.5))

        coeffs_x = optimize_1d(wp_x, times)
        coeffs_y = optimize_1d(wp_y, times)

        min_snap_points = sample_min_snap_path(
            coeffs_x,
            coeffs_y,
            times,
            samples_per_seg=50
        )

        self.obstacle_pub.publish(make_obstacle_markers(self.obstacles))
        self.waypoint_pub.publish(make_waypoint_markers(waypoints))

        self.los_marker_pub.publish(
            make_line_marker(
                waypoints,
                ns="greedy_los",
                marker_id=0,
                r=1.0,
                g=1.0,
                b=0.0,
                width=0.04
            )
        )

        self.min_snap_marker_pub.publish(
            make_line_marker(
                min_snap_points,
                ns="minimum_snap",
                marker_id=0,
                r=0.0,
                g=0.2,
                b=1.0,
                width=0.06
            )
        )

        self.los_path_pub.publish(make_path_msg(waypoints))
        self.min_snap_path_pub.publish(make_path_msg(min_snap_points))

        rospy.loginfo("Replanning done.")
        rospy.loginfo("Waypoints: %s", str(waypoints))
        rospy.loginfo("Published new /planner/min_snap_path")
    
    def replan_start_callback(self, msg):
        if not self.online_enabled:
            rospy.loginfo_throttle(
                1.0,
                "Ignoring online replan request because active_mode=%d"
                % self.active_mode
            )
            return
        data = list(msg.data)

        if len(data) < 2:
            rospy.logwarn("Invalid /planner/replan_start")
            return

        if self.target is None:
            rospy.logwarn("Cannot force replan: target not received.")
            return

        start = (float(data[0]), float(data[1]))

        if self.latest_scan is not None:
            new_obstacles = self.scan_to_obstacles(self.latest_scan)

            # 如果你有 obstacle memory，就用 memory
            if hasattr(self, "update_obstacle_memory"):
                self.update_obstacle_memory(new_obstacles)
                obstacles = self.get_memory_obstacles_for_planning()
            else:
                obstacles = new_obstacles
        else:
            obstacles = []

        rospy.loginfo(
            "Force online replan requested: start=(%.2f, %.2f), target=(%.2f, %.2f), obstacles=%d"
            % (start[0], start[1], self.target[0], self.target[1], len(obstacles))
        )

        self.plan_from_start_and_obstacles(start, obstacles)
        self.has_online_plan = True
        self.last_plan_time = rospy.Time.now()

    # def model_states_callback(self, msg): #0
    #     if self.uav_model_name not in msg.name:
    #         rospy.logwarn_throttle(
    #             1.0,
    #             "Cannot find UAV model: %s" % self.uav_model_name
    #         )
    #         return

    #     idx = msg.name.index(self.uav_model_name)
    #     pose = msg.pose[idx]

    #     self.uav_x = pose.position.x
    #     self.uav_y = pose.position.y

    #     q = pose.orientation
    #     quat = [q.x, q.y, q.z, q.w]
    #     roll, pitch, yaw = euler_from_quaternion(quat)

    #     self.uav_yaw = yaw
    #     self.uav_pose_ready = True
    def uav_state_callback(self, msg): #0
        """
        Receive estimated/controller UAV pose from trajectory.py.
        Data format:
        [x_world, y_world, yaw_world]
        """
        data = list(msg.data)

        if len(data) < 3:
            rospy.logwarn_throttle(
                1.0,
                "Invalid /planner/uav_state_xyyaw length: %d" % len(data)
            )
            return

        self.uav_x = float(data[0])
        self.uav_y = float(data[1])
        self.uav_yaw = float(data[2])
        self.uav_pose_ready = True

    def scan_to_obstacles(self, scan):
        obstacles = []

        if scan is None:
            return obstacles

        angle = scan.angle_min
        bins = {}

        for r in scan.ranges:
            if math.isfinite(r):
                if self.lidar_min_use_range < r < self.lidar_max_use_range:
                    wx, wy = self.beam_to_world(r, angle)

                    a_norm = self.normalize_angle(angle)
                    bin_id = int(math.floor(a_norm / self.lidar_bin_angle))

                    if bin_id not in bins:
                        bins[bin_id] = (wx, wy)
                    else:
                        old_wx, old_wy = bins[bin_id]
                        old_d = math.hypot(old_wx - self.uav_x, old_wy - self.uav_y)
                        new_d = math.hypot(wx - self.uav_x, wy - self.uav_y)

                        if new_d < old_d:
                            bins[bin_id] = (wx, wy)

            angle += scan.angle_increment

        for _, (wx, wy) in bins.items():
            obstacles.append({
                "type": "cylinder",
                "x": wx,
                "y": wy,
                "r": self.lidar_obstacle_radius
            })

        return obstacles
    
    def front_blocked(self, scan):
        if scan is None or self.target is None:
            return False

        angle = scan.angle_min
        min_forward = float("inf")
        blocked = False

        for r in scan.ranges:
            if math.isfinite(r):
                if self.lidar_min_use_range < r < self.lidar_max_use_range:
                    wx, wy = self.beam_to_world(r, angle)
                    if self.is_in_goal_corridor(
                        wx,
                        wy,
                        max_forward=self.front_block_threshold,
                        corridor_width=1.5
                    ):
                        dist = math.hypot(wx - self.uav_x, wy - self.uav_y)
                        min_forward = min(min_forward, dist)
                        blocked = True

            angle += scan.angle_increment

        # blocked = min_front < self.front_block_threshold

        rospy.loginfo_throttle(
            1.0,
            "Front check: min_forward=%.2f, threshold=%.2f, blocked=%s"
            % (min_forward, self.front_block_threshold, str(blocked))
        )

        return blocked
    
    def depth_callback(self, msg):
        if not self.online_enabled:
            return

        if not self.depth_enabled:
            return

        ok, depth = self.depth_msg_to_meters(msg)

        if not ok:
            rospy.logwarn_throttle(
                1.0,
                "Depth camera: unsupported encoding=%s"
                % msg.encoding
            )
            return

        h, w = depth.shape

        roi_w = int(w * self.depth_roi_w_ratio)
        roi_h = int(h * self.depth_roi_h_ratio)

        x0 = int((w - roi_w) / 2)
        y0 = int((h - roi_h) / 2)
        x1 = x0 + roi_w
        y1 = y0 + roi_h

        roi = depth[y0:y1, x0:x1]

        valid = np.isfinite(roi)
        valid = valid & (roi > 0.05) & (roi < 20.0)

        if np.count_nonzero(valid) < 10:
            self.depth_blocked = False
            self.depth_min = float("inf")
            self.depth_mean = float("inf")
            self.depth_close_ratio = 0.0
            return

        valid_depth = roi[valid]

        self.depth_min = float(np.min(valid_depth))
        self.depth_mean = float(np.mean(valid_depth))

        close = valid_depth < self.depth_trigger_distance
        self.depth_close_ratio = float(np.mean(close))

        self.depth_blocked = (
            self.depth_min < self.depth_trigger_distance and
            self.depth_close_ratio > self.depth_trigger_ratio
        )

        self.depth_last_time = rospy.Time.now()
        self.depth_blocked_pub.publish(Bool(self.depth_blocked))

        rospy.loginfo_throttle(
            1.0,
            "Depth camera check | blocked=%s | min=%.2f m | mean=%.2f m | close_ratio=%.2f | affects_planner=%s"
            % (
                str(self.depth_blocked),
                self.depth_min,
                self.depth_mean,
                self.depth_close_ratio,
                str(self.depth_affects_planner)
            )
        )
    def depth_msg_to_meters(self, msg):
        try:
            h = int(msg.height)
            w = int(msg.width)
            enc = msg.encoding.upper()

            if enc == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w))
                return True, depth

            elif enc == "16UC1":
                depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))
                depth_m = depth_mm.astype(np.float32) / 1000.0
                return True, depth_m

            else:
                return False, None

        except Exception as e:
            rospy.logwarn_throttle(
                1.0,
                "Depth conversion failed: %s" % str(e)
            )
            return False, None
    
    def online_plan_timer(self, event):
        if not self.online_enabled:
            return
        if self.latest_scan is None:
            rospy.logwarn_throttle(1.0, "Waiting for /lidar_360/scan ...")
            return

        if not self.uav_pose_ready:
            rospy.logwarn_throttle(1.0, "Waiting for UAV pose ...")
            return

        if self.target is None:
            rospy.logwarn_throttle(1.0, "Waiting for /planner/target_xy ...")
            return

        now = rospy.Time.now()
        if now - self.last_plan_time < self.plan_period:
            return

        # 第一次要先產生一條路徑，否則 trajectory.py 會一直等 /planner/min_snap_path
        need_initial_plan = not self.has_online_plan

        # 後續只有前方真的有障礙物才重規劃
        lidar_blocked = self.front_blocked(self.latest_scan)
        depth_recent = (
            rospy.Time.now() - self.depth_last_time
        ) < self.depth_timeout
        if self.depth_affects_planner and depth_recent:
            blocked = lidar_blocked or self.depth_blocked
        else:
            blocked = lidar_blocked

        rospy.loginfo_throttle(
            1.0,
            "Obstacle decision | lidar=%s | depth=%s | depth_recent=%s | final=%s"
            % (
                str(lidar_blocked),
                str(self.depth_blocked),
                str(depth_recent),
                str(blocked)
            )
        )


        if (not need_initial_plan) and (not blocked):
            rospy.loginfo_throttle(
                1.0,
                "Front is clear. Keep current path, no replanning."
            )
            return

        self.last_plan_time = now

        start = (self.uav_x, self.uav_y)

        if blocked:
            new_obstacles = self.scan_to_obstacles(self.latest_scan)
            self.update_obstacle_memory(new_obstacles)
            obstacles = self.get_memory_obstacles_for_planning()
            rospy.loginfo(
                "Front blocked. Online replanning with LiDAR obstacles=%d, memory=%d"
                % (len(new_obstacles), len(obstacles))
            )
        else:
            # 第一次規劃但前方沒障礙物：先給一條直線路徑
            obstacles = []
            rospy.loginfo(
                "Initial online plan. No front obstacle, planning straight path."
            )

        rospy.loginfo(
            "Online planning: start=(%.2f, %.2f), target=(%.2f, %.2f), obstacles=%d"
            % (start[0], start[1], self.target[0], self.target[1], len(obstacles))
        )

        self.plan_from_start_and_obstacles(start, obstacles)
        self.has_online_plan = True

    def plan_from_start_and_obstacles(self, start, obstacles):
        if self.target is None:
            rospy.logwarn("Cannot plan: target not received.")
            return

        if len(obstacles) == 0:
            rospy.logwarn_throttle(1.0, "No LiDAR obstacles detected. Planning straight path.")

        waypoints = greedy_los_planner(
            start,
            self.target,
            obstacles,
            safety_margin=SAFETY_MARGIN
        )

        waypoints = subdivide_waypoints(
            waypoints,
            max_dist=MAX_WAYPOINT_DIST
        )

        if len(waypoints) < 2:
            rospy.logerr("Planning failed: waypoint number < 2")
            return

        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]

        times = []
        for i in range(len(waypoints) - 1):
            dist = math.hypot(
                wp_x[i + 1] - wp_x[i],
                wp_y[i + 1] - wp_y[i]
            )
            times.append(max(1.5, dist / 1.5))

        coeffs_x = optimize_1d(wp_x, times)
        coeffs_y = optimize_1d(wp_y, times)

        min_snap_points = sample_min_snap_path(
            coeffs_x,
            coeffs_y,
            times,
            samples_per_seg=50
        )

        self.obstacle_pub.publish(make_obstacle_markers(obstacles))
        self.waypoint_pub.publish(make_waypoint_markers(waypoints))

        self.los_marker_pub.publish(
            make_line_marker(
                waypoints,
                ns="greedy_los",
                marker_id=0,
                r=1.0,
                g=1.0,
                b=0.0,
                width=0.04
            )
        )

        self.min_snap_marker_pub.publish(
            make_line_marker(
                min_snap_points,
                ns="minimum_snap",
                marker_id=0,
                r=0.0,
                g=0.2,
                b=1.0,
                width=0.06
            )
        )


        self.los_path_pub.publish(make_path_msg(waypoints))
        self.min_snap_path_pub.publish(make_path_msg(min_snap_points))

        rospy.loginfo("Online planning done. Published /planner/online/min_snap_path")

    def normalize_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))


    def get_goal_direction(self):
        if self.target is None:
            return None

        dx = self.target[0] - self.uav_x
        dy = self.target[1] - self.uav_y
        norm = math.hypot(dx, dy)

        if norm < 1e-6:
            return None

        return (dx / norm, dy / norm)


    def beam_to_world(self, r, angle):
        """
        LiDAR beam -> Gazebo world point
        LiDAR frame: x forward, y left
        """
        a = self.normalize_angle(angle)

        lx = r * math.cos(a)
        ly = r * math.sin(a)

        wx = self.uav_x + math.cos(self.uav_yaw) * lx - math.sin(self.uav_yaw) * ly
        wy = self.uav_y + math.sin(self.uav_yaw) * lx + math.cos(self.uav_yaw) * ly

        return wx, wy


    def is_in_goal_corridor(self, wx, wy, max_forward=None, corridor_width=None):
        """
        判斷障礙物是否位於 UAV -> target 的前方走廊內
        """
        goal_dir = self.get_goal_direction()
        if goal_dir is None:
            return False

        if max_forward is None:
            max_forward = self.front_block_threshold

        if corridor_width is None:
            corridor_width = 1.5

        gx, gy = goal_dir

        vx = wx - self.uav_x
        vy = wy - self.uav_y

        forward = vx * gx + vy * gy
        lateral = abs(vx * gy - vy * gx)

        return (0.0 < forward < max_forward) and (lateral < corridor_width)
    def update_obstacle_memory(self, new_obstacles):
        now = rospy.Time.now()

        for obs in new_obstacles:
            merged = False

            for old in self.obstacle_memory:
                d = math.hypot(obs["x"] - old["x"], obs["y"] - old["y"])

                if d < self.obstacle_memory_merge_dist:
                    # 平滑更新位置
                    old["x"] = 0.7 * old["x"] + 0.3 * obs["x"]
                    old["y"] = 0.7 * old["y"] + 0.3 * obs["y"]
                    old["r"] = max(old["r"], obs["r"])
                    old["stamp"] = now
                    merged = True
                    break

            if not merged:
                self.obstacle_memory.append({
                    "type": "cylinder",
                    "x": obs["x"],
                    "y": obs["y"],
                    "r": obs["r"],
                    "stamp": now
                })

        # 移除太舊的障礙物
        self.obstacle_memory = [
            obs for obs in self.obstacle_memory
            if now - obs["stamp"] < self.obstacle_memory_max_age
        ]


    def get_memory_obstacles_for_planning(self):
        clean = []

        for obs in self.obstacle_memory:
            clean.append({
                "type": "cylinder",
                "x": obs["x"],
                "y": obs["y"],
                "r": obs["r"]
            })

        return clean


def main():
    rospy.init_node("auto_planner_online")
    AutoPlannerNode()
    rospy.spin()


if __name__ == "__main__":
    main()