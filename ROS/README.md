# ROS
## Video_Publisher.py
A script that takes a takes two video files (A left and right camera input) then publishes them into four nodes:
- left/imageraw
- left/camera_info
- right/imageraw
- right/camera_info

The file is configed using the video_publisher_config.yaml

### TODO 
- Currently the videos are out of sync, needs to fixed 
- Should be run using a bash script that also starts a bag to auto record the ros published nodes