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

weights_dir = 'weights/best.pt'
img_dir = '/home/java/Java/data/20231201_urchin/images/test'
save_dir = '/home/java/Java/data/20231201_urchin/images/test/detect'
os.makedirs(save_dir, exist_ok=True)
imglist = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
max_no = 10
model = YOLO(weights_dir)

for i, img_name in enumerate(imglist):
    if i > max_no:
        break
    img = cv.imread(img_name)
    
    results = model(img_name)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        result.save(filename = os.path.join(save_dir, img_name + '_detect.jpg'))  # save to disk
        import code
        code.interact(local=dict(globals(), **locals()))

