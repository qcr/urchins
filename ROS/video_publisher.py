#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml
import threading
import os 

def read_config(config_file):
    """Open and read the configuration file.
    Args:
        config_file (str): Path to the configuration file.
    Returns:
        dict: Configuration parameters read from the file.
    """
    if not os.path.isfile(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        return None
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file '{config_file}': {e}")
        return None
    return config

def video_publisher(video, camera_side, common_time):
    bridge = CvBridge()
    image_pub = rospy.Publisher(camera_side+"/image_raw", Image, queue_size=10)
    camera_info_pub = rospy.Publisher(camera_side+'/camera_info', CameraInfo, queue_size=10)

    # Open the video file
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        rospy.logerr("Error opening the " + camera_side +" video file")
        return

    # Set the video frame rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Get video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ROS rate based on video frame rate
    rate = rospy.Rate(frame_rate)

    while not rospy.is_shutdown():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Convert the frame to a ROS Image message
            image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_msg.header.stamp = common_time
            # Publish the ROS Image message
            image_pub.publish(image_msg)

            # Publish camera information
            camera_info_msg = CameraInfo()
            camera_info_msg.width = width
            camera_info_msg.height = height
            camera_info_pub.publish(camera_info_msg)

        # Sleep to match the video frame rate
        rate.sleep()

    # Release the video capture object
    cap.release()

def stero_video_publisher(video_left, video_right):
    rospy.init_node('video_publisher', anonymous=True)
    common_time = rospy.Time.now()
    left = threading.Thread(target=video_publisher, args=(video_left, "left", common_time))
    right = threading.Thread(target=video_publisher, args=(video_right, "right", common_time))
    left.start()
    right.start()
    left.join()
    right.join()

if __name__ == '__main__':
    try:
        config = read_config("video_publisher_config.yaml")
        video_left = config["video_path_left_in"]
        video_right = config["video_path_right_in"]
        stero_video_publisher(video_left, video_right)
    except rospy.ROSInterruptException:
        pass
