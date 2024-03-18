#! /usr/bin/env python3

"""
class for images that have detections [class, x1 y1 x2 y2 conf id], where boxes xyxy are normalised to image size
and name properties
"""
import os
import glob
import numpy as np

from tracker.DetectionWithID import DetectionWithID


class ImageWithDetection():
    
    def __init__(self,
                 txt_file: str, 
                 image_name: str, 
                 detection_data, 
                 image_width: int = None, 
                 image_height: int = None):
        
        self.txt_file = txt_file
        self.image_name = image_name
        self.image_width = image_width
        self.image_height = image_height
        
        self.classes = []
        self.boxes = []
        self.detection_confidences = [] # detection confidence
        self.ids = []
        self.detections = []
        self.classifications = []
        self.classification_confidences = [] # classification confidence
        
        if isinstance(detection_data, np.ndarray):
            self.detection_data = np.array(detection_data) # [class x1 y1 x2 y2 conf id] for each row of detections, numpy array
            self.set_ids()
            self.set_classes()
            self.set_boxes()
            self.set_detection_confidences()
            self.get_detections_from_array()
        elif isinstance(detection_data, list):
            # assume is a list, with each element [cls, x1 y1 x2 y2 conf, track_id]
            self.detection_data = detection_data
            self.get_detections_from_list()
        elif isinstance(detection_data, DetectionWithID):
            # assume is already a DetectionWithID object
            self.append_detection_from_obj(detection_data)
        else:
            TypeError(detection_data, 'Uknown type for detection data')
    
    
    def append_classification(self, classification, classification_confidence):
        self.classifications.append(classification)
        self.classification_confidences.append(classification_confidence)
        
        
    def append_detection_from_obj(self, det):
        self.classes.append(det.class_label)
        self.boxes.append(det.box)
        self.detection_confidences.append(det.detection_confidence)
        self.ids.append(det.id)
        # self.image_name = detection_data.image_name # ?
        self.detections.append(det)
        
    
    def append_detection_from_array(self, det):
        """ add detection to image """
        self.classes.append(det[0])
        self.boxes.append(det[1:5])
        self.detection_confidences.append(det[5])
        self.ids.append(det[6])
        self.classifications.append(det[7])
        self.classification_confidences.append(det[8])
        self.detections.append(DetectionWithID(det[0], det[1:5], det[5], det[6], self.image_name, det[7], det[8]))
        
        
    def get_detections_from_list(self):
        """ get detections from list """
        self.classes = []
        self.boxes = []
        self.detection_confidences = []
        self.ids = []
        self.detections = []
        self.classifications = []
        self.classification_confidences = []
        for det in self.detection_data:
            self.append_detection_from_array(det)
            # self.classes.append(det[0])
            # self.boxes.append(det[1:5])
            # self.confidences.append(det[5])
            # self.ids.append(det[6])
            # detections.append(DetectionWithID(det[0], det[1:5], det[5], det[6], self.image_name))
        # return detections
    
    
    def set_classes(self):
        self.classes = self.detection_data[:,0]
        
    def set_boxes(self):
        self.boxes = self.detection_data[:,1:5]
        
    def set_detection_confidences(self):
        self.detection_confidences = self.detection_data[:,5]
    
    def set_ids(self):
        self.ids = self.detection_data[:, 6]

    
    def get_detections_from_array(self):
        detections = []
        for i, id in enumerate(self.ids):
            class_label = self.classes[i]
            box = self.boxes[i, :]
            confidence = self.detection_confidences[i]
            id = self.ids[i]
            detections.append(DetectionWithID(class_label, box, confidence, id, self.image_name))
        # return detections
        self.detections = detections
        

if __name__ == "__main__":
    
    # [cls, x1 y1 x2 y2 conf, track_id, predicted class, classification_confidence]
    no_detection_case = [np.array([0, 0, 0.1, 0, 0.1, 0, -1, 0, 0.0])]
    ImgDet = ImageWithDetection('txt', 'img', no_detection_case, 1920, 1080)
    
    import code
    code.interact(local=dict(globals(), **locals()))