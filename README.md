# Urchin Monitoring - Stage 1

Welcome to the urchin monitoring project for giant kelp reforestation in Tasmania. This is a feasibility study using marine robots to automatically detect sea urchins that are devastating Tasmania’s critically endangered giant kelp forests. Giant kelp forests have immense ecological, economic, and cultural value. Ecologically, giant kelp are a key-stone species that drastically change the natural landscape to provide a habitat for many other species. Economically, giant kelp is the fastest growing plant and is an excellent carbon storage option. Culturally, kelp forests have played a significant role in Aboriginal cultures and history for thousands of years. The goal of this study is to leverage state-of-the-art object detection and tracking techniques to automatically count urchins, which can be extended to differentiate invasive long-spined urchins from native short-spined urchins. 

Since it is currently difficult to get in-situ data in Tasmania, the current dataset comprises of video captured from the tanks of the Australian Instittute for Marine Science (AIMS) National Sea Simulator.  Serving as proxies for the long-spined and short-spined urchins, the two species of urchins in the current AIMS dataset are:

  - Orange = tripneustes gratilla = Tripy
  - black/purple/green = Echinometra mathaei = Rings

The goal is to build a tool (software pipeline) that takes in a video and outputs a count of the two different species of urchins and provides size metrics that can be used to inform control methods for the urchins. 

## Minimum Viable Product (MVP):
- Python script that takes in video, outputs urchin counts into text (does not differentiate between species)

## Stretch Goals (SG):
- Algorithm  can differentiate between two species
- Provides size characteristics
- We want to make the software modular, so that future extensions of this project can still use this code (eg, put this code on a robot as a module). 

# Whiteboard:
Shared edit link to Kelp Conservation Crew:
https://connectqutedu-my.sharepoint.com/:wb:/g/personal/tsaid_qut_edu_au/EeTC8KNiBLFEpLtww-Zt6KIBPSwAP5p0iPyUS8k3GmRYOQ?e=UHOsiR

# Software Environment
- Python 3.9+
- PyTorch
- Yolov8
- OpenCV
- Yolov8 (https://github.com/ultralytics/ultralytics)

# Installation
- Operating system should be Ubuntu 20.04..6 LTS
- run `make_urchins_venv.sh`
- `conda activate urchins`
- run `pip install ultralytics` 
- navigate to urchins folder
- install CVAT locally: https://opencv.github.io/cvat/docs/administration/basics/installation/#ubuntu-1804-x86_64amd64 or create an account: https://www.cvat.ai/

# Data
- location of the data (folders/directories/links)
- location of the data log
- data split, key parameters
- location of annotation tools: CVAT

# Image Processing
- scripts to rename the videos
- scripts to sort/organise the video data into images for labelling

# Urchin Detector
- scripts to sort/organise the data 
- training scripts, yaml files for training 

# Urchin Tracker
- BotSORT



