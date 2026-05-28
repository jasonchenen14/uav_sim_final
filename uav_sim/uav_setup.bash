#!/bin/bash

# ===== UAV simulator ROS/Gazebo environment =====

source /opt/ros/noetic/setup.bash

if [ -f /catkin_ws/uav_simulator_rtx/devel/setup.bash ]; then
    source /catkin_ws/uav_simulator_rtx/devel/setup.bash
fi

export GAZEBO_MODEL_PATH=/catkin_ws/uav_simulator_rtx/src/uav_gazebo/models:$GAZEBO_MODEL_PATH

# GUI fixes for Docker / X11
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_SOFTWARE=1
