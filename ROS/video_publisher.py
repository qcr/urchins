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

def video_publisher(video, camera_side, barrier):
    """Publishes a video as a ROS Image message.
    Args:
        video (str): Path to the video file.
        camera_side (str): Camera side (left or right).
        barrier (threading.Barrier): Barrier to synchronize frame processing.
    Publishes:
        Image: ROS Image message.
        CameraInfo: Camera information message.
    """
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
            image_msg.header.stamp = rospy.Time.now()  # Use the current timestamp

            # Publish the ROS Image message
            image_pub.publish(image_msg)

            # Publish camera information
            camera_info_msg = CameraInfo()
            camera_info_msg.width = width
            camera_info_msg.height = height
            camera_info_msg.header.stamp = image_msg.header.stamp  # Use the same timestamp
            camera_info_pub.publish(camera_info_msg)

        # Wait for the other thread
        barrier.wait()

        # Sleep to match the video frame rate
        rate.sleep()

    # Release the video capture object
    cap.release()

def stereo_video_publisher(video_left, video_right):
    rospy.init_node('video_publisher', anonymous=True)

    # Barrier to synchronize the two threads
    barrier = threading.Barrier(2)

    left_thread = threading.Thread(target=video_publisher, args=(video_left, "left", barrier))
    right_thread = threading.Thread(target=video_publisher, args=(video_right, "right", barrier))

    left_thread.start()
    right_thread.start()

    left_thread.join()
    right_thread.join()

if __name__ == '__main__':
    try:
        config = read_config("video_publisher_config.yaml")
        video_left = config["video_path_left_in"]
        video_right = config["video_path_right_in"]
        stereo_video_publisher(video_left, video_right)
    except rospy.ROSInterruptException:
        pass
