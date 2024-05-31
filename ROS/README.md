# ROS
## Video_Publisher.py
A script that takes a takes two video files (A left and right camera input) then publishes them into four nodes:
- [left/imageraw](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html)
- [left/camera_info](http://docs.ros.org/en/api/sensor_msgs/html/msg/CameraInfo.html)
- [right/imageraw](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html)
- [right/camera_info](http://docs.ros.org/en/api/sensor_msgs/html/msg/CameraInfo.html)

The file is configed using the video_publisher_config.yaml

### TODO 
- Should be run using a bash script that also starts a bag to auto record the ros published nodes