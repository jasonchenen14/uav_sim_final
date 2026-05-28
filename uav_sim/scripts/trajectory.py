import datetime
import numpy as np
import pdb
import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from gazebo_msgs.srv import GetModelState
from nav_msgs.msg import Path
# from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32MultiArray, Int32


class Trajectory:
    def __init__(self):
        self.mode = 0
        self.is_mode_changed = False
        self.is_landed = False

        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_traj = 0.0

        self.xd = np.zeros(3)
        self.xd_dot = np.zeros(3)
        self.xd_2dot = np.zeros(3)
        self.xd_3dot = np.zeros(3)
        self.xd_4dot = np.zeros(3)

        self.b1d = np.zeros(3)
        self.b1d[0] = 1.0
        self.b1d_dot = np.zeros(3)
        self.b1d_2dot = np.zeros(3)

        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.R = np.identity(3)
        self.W = np.zeros(3)

        self.x_init = np.zeros(3)
        self.v_init = np.zeros(3)
        self.a_init = np.zeros(3)
        self.R_init = np.identity(3)
        self.W_init = np.zeros(3)
        self.b1_init = np.zeros(3)
        self.theta_init = 0.0

        self.trajectory_started = False
        self.trajectory_complete = False
        
        # Manual mode
        self.manual_mode = False
        self.manual_mode_init = False
        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.0

        # Take-off
        self.takeoff_end_height = -1.5  # (m)
        self.takeoff_velocity = -1.0  # (m/s)

        # Landing
        self.landing_velocity = 1.0  # (m/s)
        self.landing_motor_cutoff_height = -0.25  # (m)

        # Circle
        self.circle_center = np.zeros(3)
        self.circle_linear_v = 1.0 
        self.circle_W = 1.2
        self.circle_radius = 1.2

        self.waypoint_speed = 2.0  # (m/s)

        self.e1 = np.array([1.0, 0.0, 0.0])
        # AprilTag landing
        self.tag_sub_initialized = False
        self.tag_detected = False
        self.tag_pos = np.zeros(3)
        self.tag_last_time = rospy.Time(0)

        self.tag_timeout = 0.3      # 超過 0.3 秒沒收到 tag，就視為沒偵測到
        self.tag_kx = 0.35          # x 修正增益
        self.tag_ky = 0.35          # y 修正增益
        self.tag_center_threshold = 0.05  # tag 偏差小於 5 cm 才下降
        self.tag_landing_step = 0.004     # 每次下降量，NED z 越大越往下
        self.tag_landing_vz = 0.12        # 下降速度
        self.tag_final_descent_height = -0.6   # 低於約 0.6 m 後，即使看不到 tag 也繼續降落
        self.tag_last_good_xy = np.zeros(2)     # 最後一次有看到 tag 時的目標 x/y
        self.tag_has_ever_seen = False          # 是否曾經看過 tag
        # Search direction from Gazebo ground truth
        self.search_direction_from_gazebo = True
        self.search_direction_initialized = False

        self.uav_model_name = "uav"
        self.tag_model_name = "apriltag_0"

        self.search_direction = np.array([1.0, 0.0, 0.0])
        self.search_start_pos = np.zeros(3)

        self.search_height = -1.5
        self.search_step = 0.02
        self.search_speed = 0.15
        self.search_max_distance = 3.0
        self.search_land_phase = 0
        self.tag_yaw_threshold = 0.10    # rad，大約 5.7 度
        self.tag_yaw_k = 0.4
        self.tag_yaw_step_limit = 0.08
        self.tag_q = None

         # Planner path tracking
        # self.planner_path_sub_initialized = False
        self.offline_path_sub_initialized = False
        self.online_path_sub_initialized = False
        self.planner_path_received = False

        self.planner_path_points = []
        self.planner_path_idx = 0

        # 注意：你的 UAV 座標是 NED，z 往上是負的
        self.planner_follow_height = -1.5

        self.planner_reach_radius = 0.25
        self.planner_tracking_speed = 0.5

        self.planner_hold_final = True

                # Planner mission: takeoff -> planner path -> search-and-land
        self.planner_mission_phase = 0
        self.planner_mission_started = False

        # phase 0: follow planner path
        # phase 1: search-and-land
        self.return_to_search_if_tag_lost = True
        self.tag_lost_return_timeout = 1.0
        self.tag_lost_since = None
        self.offline_replan_pub = rospy.Publisher(
            "/planner/offline/replan_start",
            Float32MultiArray,
            queue_size=1
        )
        self.online_replan_pub = rospy.Publisher(
            "/planner/online/replan_start",
            Float32MultiArray,
            queue_size=1
        )

        self.planner_replan_sent = False
        self.planner_replan_request_time = rospy.Time(0)
        self.online_path_request_time = rospy.Time(0)
        self.online_goal_xy = np.array([15.0, -0.0])
        self.online_goal_reach_radius = 0.6
        self.planner_mode_pub = rospy.Publisher(
            "/planner/active_mode",
            Int32,
            queue_size=1,
            latch=True
        )
        


    def get_desired(self, mode, states, x_offset, yaw_offset):
        self.x, self.v, self.a, self.R, self.W = states
        self.x_offset = x_offset
        self.yaw_offset = yaw_offset

        if mode == self.mode:
            self.is_mode_changed = False
        else:
            self.is_mode_changed = True
            self.mode = mode
            self.mark_traj_start()
            self.planner_mode_pub.publish(Int32(self.mode))

        self.calculate_desired()
        self.planner_mode_pub.publish(Int32(self.mode))

        desired = (self.xd, self.xd_dot, self.xd_2dot, self.xd_3dot, \
            self.xd_4dot, self.b1d, self.b1d_dot, self.b1d_2dot, self.is_landed)
        return desired

    
    def calculate_desired(self):
        if self.manual_mode:
            self.manual()
            return
        
        if self.mode == 0 or self.mode == 1:  # idle and warm-up
            self.set_desired_states_to_zero()
            self.mark_traj_start()
        elif self.mode == 2:  # take-off
            self.takeoff()
        elif self.mode == 3:  # land
            self.land()
        elif self.mode == 4:  # stay
            self.stay()
        elif self.mode == 5:  # circle
            self.circle()
        elif self.mode == 6:  # tag-land
            self.tag_land()
        elif self.mode == 7:  # search-and-land
            self.search_and_land()
        elif self.mode == 8:  # planner-path-follow
            # self.planner_path_follow()
            self.planner_mission()
        elif self.mode == 9:  # planner-path-follow
            self.online_planner_mission()




    def mark_traj_start(self):
        self.trajectory_started = False
        self.trajectory_complete = False

        self.manual_mode = False
        self.manual_mode_init = False
        self.is_landed = False

        self.t = 0.0
        self.t_traj = 0.0
        self.t0 = datetime.datetime.now()

        self.x_offset = np.zeros(3)
        self.yaw_offset = 0.0

        self.planner_mission_started = False
        self.planner_mission_phase = 0

        self.update_initial_state()


    def mark_traj_end(self, switch_to_manual=False):
        self.trajectory_complete = True

        if switch_to_manual:
            self.manual_mode = True


    def set_desired_states_to_zero(self):
        self.xd = np.zeros(3)
        self.xd_dot = np.zeros(3)
        self.xd_2dot = np.zeros(3)
        self.xd_3dot = np.zeros(3)
        self.xd_4dot = np.zeros(3)

        self.b1d = np.array([1.0, 0.0, 0.0])
        self.b1d_dot = np.zeros(3)
        self.b1d_2dot = np.zeros(3)

    
    def set_desired_states_to_current(self):
        self.xd = np.copy(self.x)
        self.xd_dot = np.copy(self.v)
        self.xd_2dot = np.zeros(3)
        self.xd_3dot = np.zeros(3)
        self.xd_4dot = np.zeros(3)

        self.b1d = self.get_current_b1()
        self.b1d_dot = np.zeros(3)
        self.b1d_2dot = np.zeros(3)


    def update_initial_state(self):
        self.x_init = np.copy(self.x)
        self.v_init = np.copy(self.v)
        self.a_init = np.copy(self.a)
        self.R_init = np.copy(self.R)
        self.W_init = np.copy(self.W)

        self.b1_init = self.get_current_b1()
        self.theta_init = np.arctan2(self.b1_init[1], self.b1_init[0])

    
    def get_current_b1(self):
        b1 = self.R.dot(self.e1)
        theta = np.arctan2(b1[1], b1[0])
        return np.array([np.cos(theta), np.sin(theta), 0.0])


    def waypoint_reached(self, waypoint, current, radius):
        delta = waypoint - current
        
        if abs(np.linalg.norm(delta) < radius):
            return True
        else:
            return False


    def update_current_time(self):
        t_now = datetime.datetime.now()
        self.t = (t_now - self.t0).total_seconds()


    def manual(self):
        if not self.manual_mode_init:
            self.set_desired_states_to_current()
            self.update_initial_state()

            self.manual_mode_init = True
            self.x_offset = np.zeros(3)
            self.yaw_offset = 0.0

            print('Switched to manual mode')
        
        self.xd = self.x_init + self.x_offset
        self.xd_dot = (self.xd - self.x) / 1.0

        theta = self.theta_init + self.yaw_offset
        self.b1d = np.array([np.cos(theta), np.sin(theta), 0.0])


    def takeoff(self):
        if not self.trajectory_started:
            self.set_desired_states_to_zero()

            # Take-off starts from the current horizontal position.
            self.xd[0] = self.x[0]
            self.xd[1] = self.x[1]
            self.x_init = self.x

            self.t_traj = (self.takeoff_end_height - self.x[2]) / \
                self.takeoff_velocity

            # Set the takeoff attitude to the current attitude.
            self.b1d = self.get_current_b1()

            self.trajectory_started = True

        self.update_current_time()

        if self.t < self.t_traj:
            self.xd[2] = self.x_init[2] + self.takeoff_velocity * self.t
            self.xd_2dot[2] = self.takeoff_velocity
        else:
            if self.waypoint_reached(self.xd, self.x, 0.04):
                self.xd[2] = self.takeoff_end_height
                self.xd_dot[2] = 0.0

                if not self.trajectory_complete:
                    print('Takeoff complete\nSwitching to manual mode')
                
                self.mark_traj_end(True)


    def land(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.t_traj = (self.landing_motor_cutoff_height - self.x[2]) / \
                self.landing_velocity

            # Set the landing attitude to the current attitude.
            self.b1d = self.get_current_b1()

            self.trajectory_started = True

        self.update_current_time()

        if self.t < self.t_traj:
            self.xd[2] = self.x_init[2] + self.landing_velocity * self.t
            self.xd_2dot[2] = self.landing_velocity
        else:
            if self.x[2] > self.landing_motor_cutoff_height:
                self.xd[2] = self.landing_motor_cutoff_height
                self.xd_dot[2] = 0.0

                if not self.trajectory_complete:
                    print('Landing complete')

                self.mark_traj_end(False)
                self.is_landed = True
            else:
                self.xd[2] = self.landing_motor_cutoff_height
                self.xd_dot[2] = self.landing_velocity

            
    def stay(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True
        
        self.mark_traj_end(True)


    def circle(self):
        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.trajectory_started = True

            self.circle_center = np.copy(self.x)
            self.circle_W = 2 * np.pi / 8

            num_circles = 2
            self.t_traj = self.circle_radius / self.circle_linear_v \
                + num_circles * 2 * np.pi / self.circle_W

        self.update_current_time()

        if self.t < (self.circle_radius / self.circle_linear_v):
            self.xd[0] = self.circle_center[0] + self.circle_linear_v * self.t
            self.xd_dot[0] = self.circle_linear_v

        elif self.t < self.t_traj:
            circle_W = self.circle_W
            circle_radius = self.circle_radius

            t = self.t - circle_radius / self.circle_linear_v
            th = circle_W * t

            circle_W2 = circle_W * circle_W
            circle_W3 = circle_W2 * circle_W
            circle_W4 = circle_W3 * circle_W

            # axis 1
            self.xd[0] = circle_radius * np.cos(th) + self.circle_center[0]
            self.xd_dot[0] = - circle_radius * circle_W * np.sin(th)
            self.xd_2dot[0] = - circle_radius * circle_W2 * np.cos(th)
            self.xd_3dot[0] = circle_radius * circle_W3 * np.sin(th)
            self.xd_4dot[0] = circle_radius * circle_W4 * np.cos(th)

            # axis 2
            self.xd[1] = circle_radius * np.sin(th) + self.circle_center[1]
            self.xd_dot[1] = circle_radius * circle_W * np.cos(th)
            self.xd_2dot[1] = - circle_radius * circle_W2 * np.sin(th)
            self.xd_3dot[1] = - circle_radius * circle_W3 * np.cos(th)
            self.xd_4dot[1] = circle_radius * circle_W4 * np.sin(th)

            w_b1d = 2.0 * np.pi / 10.0
            th_b1d = w_b1d * t

            self.b1d = np.array([np.cos(th_b1d), np.sin(th_b1d), 0])
            self.b1d_dot = np.array([- w_b1d * np.sin(th_b1d), \
                w_b1d * np.cos(th_b1d), 0.0])
            self.b1d_2dot = np.array([- w_b1d * w_b1d * np.cos(th_b1d),
                w_b1d * w_b1d * np.sin(th_b1d), 0.0])
        else:
            self.mark_traj_end(True)
        
    def init_tag_subscriber(self):
        if not self.tag_sub_initialized:
            rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.tag_callback)
            self.tag_sub_initialized = True
            print("AprilTag subscriber initialized")


    def tag_callback(self, msg):
        if len(msg.detections) == 0:
            self.tag_detected = False
            return

        det = msg.detections[0]
        pose = det.pose.pose.pose
        p = pose.position
        q = pose.orientation

        self.tag_pos = np.array([p.x, p.y, p.z])
        self.tag_q = q
        self.tag_detected = True
        self.tag_has_ever_seen = True
        self.tag_last_time = rospy.Time.now()
        self.tag_lost_since = None

    def tag_land(self):
        # 第一次進入 Tag-Land 時才建立 subscriber
        self.init_tag_subscriber()

        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.update_initial_state()
            self.trajectory_started = True
            print('Tag landing mode started')

        # 預設：清掉高階導數
        self.xd_dot = np.zeros(3)
        self.xd_2dot = np.zeros(3)
        self.xd_3dot = np.zeros(3)
        self.xd_4dot = np.zeros(3)

        self.b1d = self.get_current_b1()
        self.b1d_dot = np.zeros(3)
        self.b1d_2dot = np.zeros(3)

        now = rospy.Time.now()
        tag_valid = self.tag_detected and ((now - self.tag_last_time).to_sec() < self.tag_timeout)

        # NED 座標：z 越接近 0 代表越低
        low_enough_for_blind_landing = self.x[2] > self.tag_final_descent_height

        # ============================================================
        # 情況 A：目前有看到 AprilTag
        # ============================================================
        if tag_valid:
            tag_x = self.tag_pos[0]
            tag_y = self.tag_pos[1]

            err_norm = np.sqrt(tag_x**2 + tag_y**2)

            # 注意：正負號可能要依你的相機方向調整
            body_dx = -self.tag_ky * tag_y
            body_dy = self.tag_kx * tag_x

            body_dx = np.clip(body_dx, -0.03, 0.03)
            body_dy = np.clip(body_dy, -0.03, 0.03)
            body_correction = np.array([body_dx, body_dy, 0.0])
            world_correction = self.R.dot(body_correction)
            dx_cmd = world_correction[0]
            dy_cmd = world_correction[1]

            self.xd[0] = self.x[0] + dx_cmd
            self.xd[1] = self.x[1] + dy_cmd

            # 記住最後一次有看到 tag 時的目標水平位置
            self.tag_last_good_xy[0] = self.xd[0]
            self.tag_last_good_xy[1] = self.xd[1]

            # 還沒對準，而且還沒進入低高度盲降區：只修正 x/y，不下降
            if err_norm > self.tag_center_threshold and not low_enough_for_blind_landing:
                self.xd[2] = self.x[2]
                self.xd_dot = np.array([dx_cmd, dy_cmd, 0.0])
                return
            if self.tag_q is not None and not low_enough_for_blind_landing:
                tag_yaw = self.quat_to_yaw(self.tag_q)

                # 目標：讓 tag_yaw 接近 0
                # 如果實測轉反方向，把 -tag_yaw 改成 tag_yaw
                yaw_err = self.wrap_angle(tag_yaw)
                if abs(yaw_err) > self.tag_yaw_threshold:
                    current_b1 = self.get_current_b1()
                    current_yaw = np.arctan2(current_b1[1], current_b1[0])

                    yaw_cmd = self.tag_yaw_k * yaw_err
                    yaw_cmd = np.clip(
                        yaw_cmd,
                        -self.tag_yaw_step_limit,
                        self.tag_yaw_step_limit
                    )

                    desired_yaw = current_yaw + yaw_cmd

                    self.b1d = np.array([
                        np.cos(desired_yaw),
                        np.sin(desired_yaw),
                        0.0
                    ])

                    # 鎖住目前 XY，不下降
                    self.xd[0] = self.x[0]
                    self.xd[1] = self.x[1]
                    self.xd[2] = self.x[2]
                    self.xd_dot[:] = 0.0

                    rospy.loginfo_throttle(1.0, 'XY aligned, aligning yaw')
                    return

            # 已對準，或已經足夠低：開始下降
            self.xd[2] = self.x[2] + self.tag_landing_step
            self.xd_dot = np.array([dx_cmd, dy_cmd, self.tag_landing_vz])

        # ============================================================
        # 情況 B：目前沒看到 AprilTag
        # ============================================================
        else:
            # 沒看過 tag，或還太高：保持原地，不下降
            if (not self.tag_has_ever_seen) or (not low_enough_for_blind_landing):
                # 如果是從 search-and-land 切進來的 tag_land，
                # tag 掉太久就回到 search phase 繼續找。
                if self.return_to_search_if_tag_lost and self.search_land_phase == 1:
                    if self.tag_lost_since is None:
                        self.tag_lost_since = rospy.Time.now()

                    lost_duration = (rospy.Time.now() - self.tag_lost_since).to_sec()

                    if lost_duration > self.tag_lost_return_timeout:
                        rospy.logwarn(
                            "Tag lost for %.2f sec. Returning to Search-and-Land."
                            % lost_duration
                        )

                        # 回到 search phase
                        self.search_land_phase = 0
                        self.search_direction_initialized = False

                        # 讓 search_and_land 重新初始化搜尋方向
                        self.trajectory_started = False
                        self.trajectory_complete = False
                        self.manual_mode = False
                        self.manual_mode_init = False

                        self.tag_detected = False
                        return
                self.xd[0] = self.x[0]
                self.xd[1] = self.x[1]
                self.xd[2] = self.x[2]
                self.xd_dot[:] = 0.0
                rospy.logwarn_throttle(1.0, 'Tag lost: holding position')
                return

            # 已經低於指定高度，而且之前看過 tag：盲降
            self.xd[0] = self.tag_last_good_xy[0]
            self.xd[1] = self.tag_last_good_xy[1]
            self.xd[2] = self.x[2] + self.tag_landing_step
            self.xd_dot = np.array([0.0, 0.0, self.tag_landing_vz])
            rospy.logwarn_throttle(1.0, 'Tag lost but low enough: continuing final descent')

        # ============================================================
        # 完成降落判斷
        # ============================================================
        if self.x[2] > self.landing_motor_cutoff_height:
            rospy.logwarn_throttle(1.0,'Tag landing complete')
            self.is_landed = True
            self.mark_traj_end(False)
    def read_search_direction_from_gazebo(self):
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=1.0)
            get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

            uav_resp = get_model_state(self.uav_model_name, 'world')
            tag_resp = get_model_state(self.tag_model_name, 'world')

            if not uav_resp.success:
                rospy.logwarn("Failed to read UAV model state: %s" % uav_resp.status_message)
                return False

            if not tag_resp.success:
                rospy.logwarn("Failed to read AprilTag model state: %s" % tag_resp.status_message)
                return False

            uav_xy = np.array([
                uav_resp.pose.position.x,
                uav_resp.pose.position.y
            ])

            tag_xy = np.array([
                tag_resp.pose.position.x,
                tag_resp.pose.position.y
            ])

            direction_xy = tag_xy - uav_xy
            direction_xy[1] = -direction_xy[1]
            norm = np.linalg.norm(direction_xy)
            self.search_distance_to_tag = norm
            self.search_max_distance = norm + 1.0
            if norm < 1e-6:
                rospy.logwarn("UAV is already very close to tag position. Using default search direction.")
                self.search_direction = np.array([1.0, 0.0, 0.0])
                return True

            direction_xy = direction_xy / norm

            self.search_direction = np.array([
                direction_xy[0],
                direction_xy[1],
                0.0
            ])

            rospy.loginfo(
                "Read rough direction from Gazebo: uav=(%.2f, %.2f), tag=(%.2f, %.2f), dir=(%.2f, %.2f)"
                % (
                    uav_xy[0], uav_xy[1],
                    tag_xy[0], tag_xy[1],
                    self.search_direction[0], self.search_direction[1]
                )
            )

            return True

        except Exception as e:
            rospy.logwarn("Could not read search direction from Gazebo: %s" % str(e))
            return False
    def search_and_land(self):
        """
        從 Gazebo 讀取 UAV 與 AprilTag 的相對方向。
        UAV 只沿著大致方向搜尋，不直接使用精準 tag 座標降落。
        一旦 AprilTag 被下視相機辨識，就切換到 tag_land()。
        """

        self.init_tag_subscriber()

        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.update_initial_state()
            self.trajectory_started = True

            self.search_start_pos = np.copy(self.x)
            self.search_direction_initialized = False
            self.search_land_phase = 0
            self.search_direction = np.array([1.0, 0.0, 0.0])

            print('Search-and-Land mode started')

        # ============================================================
        # Phase 1：已經辨識到 tag，直接交給 tag_land()
        # ============================================================
        if self.search_land_phase == 1:
            self.tag_land()
            return
        # 第一次進入時，從 Gazebo 讀取大致方向
        if not self.search_direction_initialized:
            ok = self.read_search_direction_from_gazebo()

            if not ok:
                self.xd[0] = self.x[0]
                self.xd[1] = self.x[1]
                self.xd[2] = self.x[2]
                self.xd_dot = np.zeros(3)

                rospy.logwarn_throttle(
                    1.0,
                    "Waiting for rough search direction from Gazebo..."
                )
                return

            self.search_direction_initialized = True
            self.search_start_pos = np.copy(self.x)

            rospy.loginfo(
                "New search direction initialized: dir=(%.3f, %.3f, %.3f)"
                % (
                    self.search_direction[0],
                    self.search_direction[1],
                    self.search_direction[2]
                )
            )

            # if not ok:
            #     self.xd[0] = self.x[0]
            #     self.xd[1] = self.x[1]
            #     self.xd[2] = self.x[2]
            #     self.xd_dot = np.zeros(3)

            #     rospy.logwarn_throttle(
            #         1.0,
            #         "Waiting for rough search direction from Gazebo..."
            #     )
            #     return

            self.search_direction_initialized = True
            self.search_start_pos = np.copy(self.x)

        # 預設清掉高階導數
        self.xd_dot = np.zeros(3)
        self.xd_2dot = np.zeros(3)
        self.xd_3dot = np.zeros(3)
        self.xd_4dot = np.zeros(3)

        self.b1d = self.get_current_b1()
        self.b1d_dot = np.zeros(3)
        self.b1d_2dot = np.zeros(3)

        now = rospy.Time.now()
        tag_valid = self.tag_detected and ((now - self.tag_last_time).to_sec() < self.tag_timeout)

        # 看到 tag，切換到 visual landing
        if tag_valid:
            rospy.loginfo('AprilTag detected. Switching to visual tag landing...')
            self.search_land_phase = 1
            self.trajectory_started = False
            self.tag_land()
            return

        # 沒看到 tag，就沿著 Gazebo 提供的大致方向搜尋
        direction = np.copy(self.search_direction)
        direction[2] = 0.0

        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm_dir

        traveled_xy = np.linalg.norm(self.x[0:2] - self.search_start_pos[0:2])

        if traveled_xy > self.search_max_distance:
            self.xd[0] = self.x[0]
            self.xd[1] = self.x[1]
            self.xd[2] = self.search_height
            self.xd_dot[:] = 0.0

            rospy.logwarn_throttle(
                1.0,
                "Search-and-Land: max search distance reached, holding position"
            )
            return

        self.xd[0] = self.x[0] + direction[0] * self.search_step
        self.xd[1] = self.x[1] + direction[1] * self.search_step
        self.xd[2] = self.search_height

        self.xd_dot = np.array([
            direction[0] * self.search_speed,
            direction[1] * self.search_speed,
            0.0
        ])

        rospy.loginfo_throttle(
            1.0,
            "Search-and-Land | searching from Gazebo direction | current=(%.2f, %.2f, %.2f), dir=(%.2f, %.2f), traveled=%.2f, tag_valid=%s"
            % (
                self.x[0], self.x[1], self.x[2],
                direction[0], direction[1],
                traveled_xy,
                # self.search_max_distance,
                tag_valid
            )
        )

    def wrap_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))


    def quat_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def init_planner_path_subscriber(self):
        if not self.offline_path_sub_initialized:
            rospy.Subscriber(
                "/planner/offline/min_snap_path",
                Path,
                self.planner_path_callback
            )
            self.offline_path_sub_initialized = True
            print("Offline Planner path subscriber initialized")
    def planner_path_callback(self, msg):
        if self.mode != 8:
            return
        if msg.header.stamp < self.planner_replan_request_time:
            rospy.logwarn_throttle(1.0, "Ignoring old planner path")
            return
        points = []

        for pose_stamped in msg.poses:
            p = pose_stamped.pose.position

            # x, y 使用 planner 給的
            # z 不直接用 RViz 的 z，因為你的控制器是 NED 座標
            points.append(np.array([
                p.x,
                -p.y,
                self.planner_follow_height
            ]))

        if len(points) == 0:
            rospy.logwarn("Received empty planner path")
            return

        self.planner_path_points = points
        self.planner_path_idx = 0
        self.planner_path_received = True

        rospy.loginfo("Received planner path with %d points", len(points))
    def planner_path_follow(self):
        self.init_planner_path_subscriber()

        if not self.trajectory_started:
            self.init_planner_path_subscriber()
            self.set_desired_states_to_current()
            self.update_initial_state()

            self.planner_path_idx = 0
            self.planner_path_points = []
            self.planner_path_received = False
            self.planner_replan_sent = False
            self.trajectory_started = True

            print("Planner path tracking mode started")

        if not self.planner_replan_sent:
            msg = Float32MultiArray()

            # controller/NED -> planner/Gazebo world
            # x 保持一樣，y 反號
            start_x = float(self.x[0])
            start_y = float(-self.x[1])

            msg.data = [start_x, start_y]

            self.planner_replan_request_time = rospy.Time.now()
            self.offline_replan_pub.publish(msg)
            self.planner_replan_sent = True

            rospy.loginfo(
                "Requested planner replan from current UAV position: world_start=(%.2f, %.2f)"
                % (start_x, start_y)
            )

        # 沒收到 planner path，就先停在原地
        if not self.planner_path_received or len(self.planner_path_points) == 0:
            self.xd[0] = self.x[0]
            self.xd[1] = self.x[1]
            self.xd[2] = self.takeoff_end_height

            self.xd_dot = np.zeros(3)
            self.xd_2dot = np.zeros(3)
            self.xd_3dot = np.zeros(3)
            self.xd_4dot = np.zeros(3)

            self.b1d = self.get_current_b1()
            self.b1d_dot = np.zeros(3)
            self.b1d_2dot = np.zeros(3)

            rospy.logwarn_throttle(1.0, "Waiting for /planner/min_snap_path ...")
            return

        # 避免 index 超出
        if self.planner_path_idx >= len(self.planner_path_points):
            self.planner_path_idx = len(self.planner_path_points) - 1

        target = self.planner_path_points[self.planner_path_idx]

        # 判斷是否到達目前追蹤點
        dist = np.linalg.norm(target - self.x)

        if dist < self.planner_reach_radius:
            if self.planner_path_idx < len(self.planner_path_points) - 1:
                self.planner_path_idx += 1
                target = self.planner_path_points[self.planner_path_idx]
            else:
                # 到達最後一點
                self.xd = np.copy(target)
                self.xd_dot = np.zeros(3)
                self.xd_2dot = np.zeros(3)
                self.xd_3dot = np.zeros(3)
                self.xd_4dot = np.zeros(3)

                self.b1d = self.get_current_b1()
                self.b1d_dot = np.zeros(3)
                self.b1d_2dot = np.zeros(3)

                if not self.trajectory_complete:
                    print("Planner path tracking complete")

                self.mark_traj_end(True)
                return

        # 追蹤目前 path point
        direction = target - self.x
        norm_dir = np.linalg.norm(direction)

        if norm_dir > 1e-6:
            direction = direction / norm_dir
        else:
            direction = np.zeros(3)

        self.xd = np.copy(target)

        # 給一個簡單速度前饋，不要太大
        self.xd_dot = direction * self.planner_tracking_speed

        # 高階導數先設 0，讓控制器只追位置/速度
        self.xd_2dot = np.zeros(3)
        self.xd_3dot = np.zeros(3)
        self.xd_4dot = np.zeros(3)

        # yaw 先維持目前方向
        self.b1d = self.get_current_b1()
        self.b1d_dot = np.zeros(3)
        self.b1d_2dot = np.zeros(3)

        rospy.loginfo_throttle(
            1.0,
            "Planner tracking | idx=%d/%d | target=(%.2f, %.2f, %.2f) | current=(%.2f, %.2f, %.2f)"
            % (
                self.planner_path_idx,
                len(self.planner_path_points) - 1,
                target[0],
                target[1],
                target[2],
                self.x[0],
                self.x[1],
                self.x[2]
            )
        )

    def planner_mission(self):
        """
        Planner mode full mission:
        phase 0: follow /planner/min_snap_path
        phase 1: search-and-land
        """

        # 第一次進入 Planner mode
        if not self.planner_mission_started:
            self.planner_mission_phase = 0
            self.planner_mission_started = True

            self.manual_mode = False
            self.manual_mode_init = False
            self.trajectory_started = False
            self.trajectory_complete = False
            # 重新規劃用
            self.planner_path_received = False
            self.planner_path_points = []
            self.planner_path_idx = 0
            self.planner_replan_sent = False

            print("Planner mission started: Planner Path -> Search-and-Land")

        # # ============================================================
        # # Phase 0: Takeoff
        # # ============================================================
        # if self.planner_mission_phase == 0:
        #     self.takeoff()

        #     # takeoff() 完成時會 mark_traj_end(True)，所以這裡要把 manual 關掉
        #     if self.trajectory_complete:
        #         print("Planner mission: takeoff complete, switching to planner path")

        #         self.manual_mode = False
        #         self.manual_mode_init = False
        #         self.trajectory_started = False
        #         self.trajectory_complete = False

        #         self.planner_mission_phase = 1

        #     return

        # ============================================================
        # Phase 0: Follow planner path
        # ============================================================
        if self.planner_mission_phase == 0:
            self.planner_path_follow()

            # planner_path_follow() 結束時如果 mark_traj_end(True)，這裡也要關掉 manual
            if self.trajectory_complete:
                print("Planner mission: planner path complete, switching to Search-and-Land")

                self.manual_mode = False
                self.manual_mode_init = False
                self.trajectory_started = False
                self.trajectory_complete = False

                # 重置 search-and-land 狀態
                self.search_direction_initialized = False
                self.search_land_phase = 0
                self.tag_detected = False
                self.tag_has_ever_seen = False
                self.tag_lost_since = None

                self.planner_mission_phase = 1

            return

        # ============================================================
        # Phase 1: Search-and-Land
        # ============================================================
        if self.planner_mission_phase == 1:
            self.search_and_land()
            return
        
    def online_planner_path_callback(self, msg):
        if self.mode != 9:
            return
        # 忽略重新進入 online planner 前的舊 path
        if msg.header.stamp < self.online_path_request_time:
            rospy.logwarn_throttle(1.0, "Ignoring old ONLINE planner path")
            return
        points = []

        for pose_stamped in msg.poses:
            p = pose_stamped.pose.position

            # auto_planner_online.py 發的是 Gazebo/RViz world 座標
            # trajectory/controller 使用 NED-like 座標，所以 y 要反號
            # z 不用 path 的 z，固定使用 planner_follow_height
            points.append(np.array([
                p.x,
                -p.y,
                self.planner_follow_height
            ]))

        if len(points) == 0:
            rospy.logwarn("Received empty online planner path")
            return

        self.planner_path_points = points
        # self.planner_path_idx = 0
        self.planner_path_idx = self.find_nearest_path_index(
            self.planner_path_points,
            lookahead=5
        )
        self.planner_path_received = True

        rospy.loginfo("Received ONLINE planner path with %d points", len(points))

    def init_online_planner_path_subscriber(self):
        if not self.online_path_sub_initialized:
            rospy.Subscriber(
                "/planner/online/min_snap_path",
                Path,
                self.online_planner_path_callback
            )
            self.online_path_sub_initialized = True
            print("Online planner path subscriber initialized")

    def online_planner_path_follow(self):
        self.init_online_planner_path_subscriber()

        if not self.trajectory_started:
            self.set_desired_states_to_current()
            self.update_initial_state()

            # Online planner 是 2D 避障，固定維持 takeoff 高度
            self.planner_follow_height = self.takeoff_end_height

            # 不清空 planner_path_points，因為 auto_planner_online.py
            # 可能已經先 publish 過 path
            if self.planner_path_received and len(self.planner_path_points) > 0:
                self.planner_path_idx = self.find_nearest_path_index(
                self.planner_path_points,
                lookahead=5
            )
            else:
                self.planner_path_idx = 0

            self.trajectory_started = True

            print("Online planner path tracking mode started")
        if not self.planner_replan_sent:
            msg = Float32MultiArray()

            # trajectory/controller 是 NED-like，planner/Gazebo world 的 y 要反號
            start_x = float(self.x[0])
            start_y = float(-self.x[1])

            msg.data = [start_x, start_y]

            self.online_path_request_time = rospy.Time.now()
            self.planner_replan_request_time = self.online_path_request_time
            self.online_replan_pub.publish(msg)
            self.planner_replan_sent = True

            rospy.loginfo(
                "Requested ONLINE planner replan from current UAV position: world_start=(%.2f, %.2f)"
                % (start_x, start_y)
            )

        # 還沒收到 online planner path，先固定在目前 XY + takeoff 高度
        if not self.planner_path_received or len(self.planner_path_points) == 0:
            self.xd[0] = self.x[0]
            self.xd[1] = self.x[1]
            self.xd[2] = self.planner_follow_height

            self.xd_dot = np.zeros(3)
            self.xd_2dot = np.zeros(3)
            self.xd_3dot = np.zeros(3)
            self.xd_4dot = np.zeros(3)

            self.b1d = self.get_current_b1()
            self.b1d_dot = np.zeros(3)
            self.b1d_2dot = np.zeros(3)

            rospy.logwarn_throttle(
                1.0,
                "Waiting for ONLINE /planner/min_snap_path ..."
            )
            return

        # 避免 index 超出
        if self.planner_path_idx >= len(self.planner_path_points):
            self.planner_path_idx = len(self.planner_path_points) - 1

        target = np.copy(self.planner_path_points[self.planner_path_idx])
        target[2] = self.planner_follow_height

        # Online planner 是 2D path，只用 XY 判斷是否抵達
        dist_xy = np.linalg.norm(target[0:2] - self.x[0:2])

        if dist_xy < self.planner_reach_radius:
            if self.planner_path_idx < len(self.planner_path_points) - 1:
                self.planner_path_idx += 1
                target = np.copy(self.planner_path_points[self.planner_path_idx])
                target[2] = self.planner_follow_height
            else:
                self.xd = np.copy(target)
                self.xd[2] = self.planner_follow_height

                self.xd_dot = np.zeros(3)
                self.xd_2dot = np.zeros(3)
                self.xd_3dot = np.zeros(3)
                self.xd_4dot = np.zeros(3)

                self.b1d = self.get_current_b1()
                self.b1d_dot = np.zeros(3)
                self.b1d_2dot = np.zeros(3)

                goal_dist = np.linalg.norm(self.x[0:2] - self.online_goal_xy)
                if goal_dist < self.online_goal_reach_radius:
                    if not self.trajectory_complete:
                        print("Online planner path tracking complete: global target reached")

                    self.mark_traj_end(True)
                    return
                else:
                    rospy.logwarn_throttle(
                        1.0,
                        "Online path ended but global target not reached. Holding and requesting replan. dist=%.2f"
                        % goal_dist
                    )
                # 清掉舊 path，要求重新規劃，不進 search mode
                self.planner_path_received = False
                self.planner_path_points = []
                self.planner_path_idx = 0
                self.planner_replan_sent = False
                self.trajectory_complete = False
                self.manual_mode = False
                return

        # 只追 XY，Z 不參與速度
        direction = np.zeros(3)
        direction[0:2] = target[0:2] - self.x[0:2]

        norm_xy = np.linalg.norm(direction[0:2])

        if norm_xy > 1e-6:
            direction[0:2] = direction[0:2] / norm_xy
        else:
            direction[0:2] = 0.0

        self.xd = np.copy(target)
        self.xd[2] = self.planner_follow_height

        self.xd_dot = np.zeros(3)
        self.xd_dot[0:2] = direction[0:2] * self.planner_tracking_speed
        self.xd_dot[2] = 0.0

        self.xd_2dot = np.zeros(3)
        self.xd_3dot = np.zeros(3)
        self.xd_4dot = np.zeros(3)

        if np.linalg.norm(direction[0:2]) > 1e-6:
            yaw_des = np.arctan2(direction[1], direction[0])

            self.b1d = np.array([
                np.cos(yaw_des),
                np.sin(yaw_des),
                0.0
            ])
        else:
            self.b1d = self.get_current_b1()
        self.b1d_dot = np.zeros(3)
        self.b1d_2dot = np.zeros(3)

        rospy.loginfo_throttle(
            1.0,
            "Online planner tracking | idx=%d/%d | target=(%.2f, %.2f, %.2f) | current=(%.2f, %.2f, %.2f)"
            % (
                self.planner_path_idx,
                len(self.planner_path_points) - 1,
                target[0],
                target[1],
                target[2],
                self.x[0],
                self.x[1],
                self.x[2]
            )
        )
    def online_planner_mission(self):
        """
        Online planner mission:
        phase 0: follow online /planner/min_snap_path
        phase 1: search-and-land
        """

        if not self.planner_mission_started:
            self.planner_mission_phase = 0
            self.planner_mission_started = True

            self.manual_mode = False
            self.manual_mode_init = False
            self.trajectory_started = False
            self.trajectory_complete = False

            # 每次進入 online planner，都清掉舊路徑
            self.planner_path_received = False
            self.planner_path_points = []
            self.planner_path_idx = 0

            # online 不使用 /planner/replan_start
            self.planner_replan_sent = False

            print("Online planner mission started: Online Path -> Search-and-Land")

        # ============================================================
        # Phase 0: Follow online planner path
        # ============================================================
        if self.planner_mission_phase == 0:
            self.online_planner_path_follow()

            if self.trajectory_complete:
                print("Online planner mission: path complete, switching to Search-and-Land")

                self.manual_mode = False
                self.manual_mode_init = False
                self.trajectory_started = False
                self.trajectory_complete = False

                self.search_direction_initialized = False
                self.search_land_phase = 0
                self.tag_detected = False
                self.tag_has_ever_seen = False
                self.tag_lost_since = None

                self.planner_mission_phase = 1

            return

        # ============================================================
        # Phase 1: Search-and-Land
        # ============================================================
        if self.planner_mission_phase == 1:
            self.search_and_land()
            return
        
    def find_nearest_path_index(self, points, lookahead=5):
        if len(points) == 0:
            return 0

        dists = [
            np.linalg.norm(p[0:2] - self.x[0:2])
            for p in points
        ]

        nearest_idx = int(np.argmin(dists))

        # 稍微往前看幾個點，避免追到身後的點
        nearest_idx = min(nearest_idx + lookahead, len(points) - 1)

        return nearest_idx
