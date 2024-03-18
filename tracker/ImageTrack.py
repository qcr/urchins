#! /usr/bin/env python3

""" track class
the class for an individual tracked turtle
"""

import os
import numpy as np
from tracker.DetectionWithID import DetectionWithID


class ImageTrack:
    
    def __init__(self, id: int, detection: DetectionWithID):
        self.id = id
        
        
        # placeholders to add on as we iterate through image_list
        self.image_names = []
        self.detections = []
        self.boxes = []
        self.class_labels = []
        self.detection_confidences = []
        self.ids = [] # this should not fluctuate, but we include it for debugging purposes
        
        self.classifications = []
        self.classification_confidences = []
        
        self.add_detection(detection)

        self.classification_overall = []
        
        
    def add_detection(self, detection: DetectionWithID):
        self.detections.append(detection)
        self.boxes.append(detection.box)
        self.class_labels.append(detection.class_label)
        self.detection_confidences.append(detection.detection_confidence)
        self.ids.append(detection.id)
        self.image_names.append(detection.image_name)
        self.classifications.append(detection.classification)
        self.classification_confidences.append(detection.classification_confidence)


    def add_classification(self, classification, classification_confidence):
        self.classifications.append(classification)
        self.classification_confidences.append(classification_confidence)


    def print_boxes(self):
        print('Boxes: [x1 y1 x2 y2] normalised')
        for i, box in enumerate(self.boxes):
            print(f'{i}: [{box[0]} {box[1]} {box[2]} {box[3]}]')
            
            
    def print_names(self):
        print('Image Names:')
        for i, name in enumerate(self.image_names):
            print(f'{i}: {os.path.basename(name)}')
            
            
    def print_classifications(self):
        print('Track Classification:')
        print(f'{self.classifications}')
        # for i, classification in enumerate(self.classifications):
        #     print(f'{i}: ')
        
        
    def print_track(self):
        print(f'Track ID: {self.id}')
        print(f'Class labels: {self.class_labels}')
        self.print_names()
        self.print_boxes()
        
        
    