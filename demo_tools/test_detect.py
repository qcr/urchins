#! /usr/bin/env python3

"""
test_detect.py - Test the detection of the YOLOv5 model on a images file
"""

import os
import torch
import torchvision
import glob
# import numpy as np
# from PIL import Image as PILImage
import cv2 as cv
import time
import pickle
from ultralytics import YOLO

weights_dir = 'weights/X_Urchin_Detector_2024_03_12.pt'
img_dir = '/home/wardlewo/Reggie/data/20231201_urchin/images/test'
save_dir = '/home/wardlewo/Reggie/data/20231201_urchin/test_results/detect'
os.makedirs(save_dir, exist_ok=True)
imglist = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
max_no = 10
model = YOLO(weights_dir)

for i, img_name in enumerate(imglist):
    if i > max_no:
        break
    img = cv.imread(img_name)
    img_filename = os.path.basename(img_name)

    # Perform detection on the image
    results = model(img)  # Pass image data directly
    
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
    
        # Construct the save path
        save_path = os.path.join(save_dir, img_filename + '_detect.jpg')
    
        result.save(filename=save_path)  # save to disk
