#!/usr/bin/env python3
import rospy
import math
import numpy as np

from scipy.optimize import minimize

# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32MultiArray, Int32
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path


# =========================
# Planner parameters
# =========================
START_POS = (0.0, 0.0)
FLIGHT_Z = 1.5

SAFETY_MARGIN = 0.8
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

        self.obstacle_pub = rospy.Publisher(
            "/planner/offline/obstacles_marker",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.waypoint_pub = rospy.Publisher(
            "/planner/offline/waypoints_marker",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.los_marker_pub = rospy.Publisher(
            "/planner/offline/los_marker",
            Marker,
            queue_size=1,
            latch=True
        )

        self.min_snap_marker_pub = rospy.Publisher(
            "/planner/offline/min_snap_marker",
            Marker,
            queue_size=1,
            latch=True
        )

        self.los_path_pub = rospy.Publisher(
            "/planner/offline/los_path",
            Path,
            queue_size=1,
            latch=True
        )

        self.min_snap_path_pub = rospy.Publisher(
            "/planner/offline/min_snap_path",
            Path,
            queue_size=1,
            latch=True
        )

        rospy.Subscriber(
            "/planner/obstacle_list",
            Float32MultiArray,
            self.obstacle_callback
        )

        rospy.Subscriber(
            "/planner/target_xy",
            Float32MultiArray,
            self.target_callback
        )
        rospy.Subscriber(
            "/planner/offline/replan_start",
            Float32MultiArray,
            self.replan_start_callback
        )
        self.active_mode = -1
        self.offline_enabled = False

        rospy.Subscriber(
            "/planner/active_mode",
            Int32,
            self.active_mode_callback
        )

        rospy.loginfo("auto_planner waiting for /planner/obstacle_list and /planner/target_xy")

    def active_mode_callback(self, msg):
        self.active_mode = msg.data
        self.offline_enabled = (self.active_mode == 8)

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
        if not self.offline_enabled:
            rospy.loginfo_throttle(
                1.0,
                "Ignoring offline replan request because active_mode=%d"
                % self.active_mode
            )
            return
        data = list(msg.data)

        if len(data) < 2:
            rospy.logwarn("Invalid /planner/replan_start")
            return

        start = (float(data[0]), float(data[1]))

        rospy.loginfo(
            "Received replan request. start=(%.2f, %.2f)"
            % (start[0], start[1])
        )

        self.plan_from_start(start)


def main():
    rospy.init_node("auto_planner_offline")
    AutoPlannerNode()
    rospy.spin()


if __name__ == "__main__":
    main()