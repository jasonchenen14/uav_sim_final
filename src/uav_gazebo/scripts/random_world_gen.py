#!/usr/bin/env python3
import rospy
import random
import math

from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray

NUM_OBSTACLES = 8

START_POS = (0.0, 0.0)
TARGET_POS = (15.0, 0.0)

MIN_X, MAX_X = 2.0, 13.0
MIN_Y, MAX_Y = -3.5, 3.5

CYL_RADIUS = 0.2
CYL_LENGTH = 3.0


def make_cylinder_sdf(radius, length):
    return f"""
<sdf version="1.6">
  <model name="random_cylinder">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def make_target_box_sdf():
    return """
<sdf version="1.6">
  <model name="target_box">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def generate_obstacle_positions():
    obstacles = []

    while len(obstacles) < NUM_OBSTACLES:
        rx = round(random.uniform(MIN_X, MAX_X), 2)
        ry = round(random.uniform(MIN_Y, MAX_Y), 2)

        dist_to_start = math.hypot(rx - START_POS[0], ry - START_POS[1])
        dist_to_target = math.hypot(rx - TARGET_POS[0], ry - TARGET_POS[1])

        too_close = False
        for ox, oy in obstacles:
            if math.hypot(rx - ox, ry - oy) < 1.5:
                too_close = True
                break

        if dist_to_start > 2.0 and dist_to_target > 2.0 and not too_close:
            obstacles.append((rx, ry))

    return obstacles


def spawn_model(spawn_srv, model_name, model_xml, x, y, z):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    try:
        spawn_srv(model_name, model_xml, "", pose, "world")
        rospy.loginfo(f"Spawned {model_name} at x={x}, y={y}, z={z}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to spawn {model_name}: {e}")


def main():
    rospy.init_node("random_world_gen")
    obstacle_pub = rospy.Publisher("/planner/obstacle_list", Float32MultiArray, queue_size=1, latch=True)

    target_pub = rospy.Publisher("/planner/target_xy", Float32MultiArray, queue_size=1, latch=True)

    rospy.loginfo("Waiting for /gazebo/spawn_sdf_model...")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    # 等 Gazebo world 完全起來
    rospy.sleep(2.0)

    # 生成目標 box
    # spawn_model(
    #     spawn_srv,
    #     "target_box",
    #     make_target_box_sdf(),
    #     TARGET_POS[0],
    #     TARGET_POS[1],
    #     0.5
    # )

    # 生成隨機障礙物
    obstacles = generate_obstacle_positions()
    cyl_sdf = make_cylinder_sdf(CYL_RADIUS, CYL_LENGTH)
    # publish target position
    target_msg = Float32MultiArray()
    target_msg.data = [TARGET_POS[0], TARGET_POS[1]]
    target_pub.publish(target_msg)

    # publish obstacle list: [x1, y1, r1, x2, y2, r2, ...]
    obs_msg = Float32MultiArray()
    obs_data = []

    for x, y in obstacles:
        obs_data.extend([x, y, CYL_RADIUS])

    obs_msg.data = obs_data
    obstacle_pub.publish(obs_msg)

    rospy.loginfo(f"Published obstacle list: {obs_data}")

    for i, (x, y) in enumerate(obstacles):
        spawn_model(
            spawn_srv,
            f"rand_cyl_{i+1}",
            cyl_sdf,
            x,
            y,
            CYL_LENGTH / 2.0
        )

    rospy.loginfo("Random obstacle generation finished.")
    rospy.spin()


if __name__ == "__main__":
    main()