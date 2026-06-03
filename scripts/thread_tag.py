from rover import rover

import datetime
import rospy

from apriltag_ros.msg import AprilTagDetectionArray


def thread_tag():
    print('TAG: thread starting ..')

    rospy.Subscriber('/tag_detections', AprilTagDetectionArray, rover.ros_tag_detection_callback)

    rate = rospy.Rate(30)

    freq = 30.0
    t_pre = datetime.datetime.now()
    avg_number = 30

    while not rospy.is_shutdown() and rover.on:
        t = datetime.datetime.now()
        dt = (t - t_pre).total_seconds()
        if dt < 1e-6:
            continue

        freq = (freq * (avg_number - 1) + (1 / dt)) / avg_number
        t_pre = t
        rover.freq_tag = freq

        rate.sleep()

    print('TAG: thread closed!')