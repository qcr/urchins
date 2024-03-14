import cv2 as cv
import os
import glob
import code
import numpy as np
from ultralytics import YOLO
import time
import yaml
from PIL import Image as PILImage
import csv
from datetime import datetime
import pandas as pd

import rospy
import std_msgs.msg
from std_msgs.msg import Int8, Float64, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from tracker.ImageTrack import ImageTrack
from tracker.DetectionWithID import DetectionWithID
from tracker.ImageWithDetectionTrack import ImageWithDetectionTrack

from classifier.Classifierv8 import Classifier

from plotter.Plotter import Plotter
from tracker.ImageWithDetection import ImageWithDetection
# from tracker.Tracker import Tracker

'''Class that takes a video and outputs a video file with tracks and
classifications and a text file with final turtle counts'''



class Pipeline:

    default_config_file = 'pipeline_config.yaml' # configuration file for video/model/output
    default_output_file = 'urchin_counts.csv'
    default_image_suffix = '.jpg'
    img_scale_factor = 0.3 # for display-purposes only
    max_time_min = 6 # max 6 minutes/video
    default_max_frames = 1000
    
    def __init__(self,
                 config_file: str = default_config_file,
                 img_suffix: str = default_image_suffix,
                 output_file: str = default_output_file,
                 max_frames = None):
        
        self.WRITE_VID = 0

        # ROS subscribers
        self.sub_start_stop = rospy.Subscriber('/start', Int8, self.callback, queue_size=5)

        # ROS publishers
        self.pub_status = rospy.Publisher('/status', String, queue_size=5)
        self.pub_clean_img = rospy.Publisher('/clean_img', Image, queue_size=5)
        self.pub_output_img = rospy.Publisher('/output_img', Image, queue_size=5)
        self.pub_counts = rospy.Publisher('/counts', Int8, queue_size=5)

        self.bridge = CvBridge()
        
        self.config_file = config_file
        config = self.read_config(config_file)
        
        self.video_path = config['video_path_in']
        self.video_name = os.path.basename(self.video_path).rsplit('.', 1)[0]
        
        self.save_dir = config['save_dir']
        
        # make output file default to video name, not default_output_file
        output_name = self.video_name + '.csv'
        self.output_file = os.path.join(self.save_dir, output_name)
        self.output_tracks = os.path.join(self.save_dir,'tracks.csv')
        self.frame_skip = config['frame_skip']
        
        self.get_video_info()
        if max_frames is None or max_frames <= 0:
            self.set_max_count(self.max_time_min) # setup max count from defaults
        else:
            self.max_count = max_frames
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.image_suffix = img_suffix
        
        self.model_track = YOLO(config['detection_model_path'])
        self.model_track.fuse()
        self.classifier_weights = config['classification_model_path']
        self.yolo_path = config['YOLOv8_install_path']
        
        self.overall_class_confidence_threshold = config['overall_class_confidence_threshold']
        self.overall_class_track_threshold = config['overall_class_track_threshold']

        self.detection_confidence_threshold = config['detection_confidence_threshold']
        self.detection_iou_threshold = config['detection_iou_threshold']
        
        self.detector_image_size = config['detector_image_size']
        self.image_scale_factor = self.detector_image_size / self.image_width
        
        self.TurtleClassifier = Classifier(weights_file = self.classifier_weights,
                                      yolo_dir = self.yolo_path)
        
        self.datestr = datetime.now()

    # Start/stop callback
    def callback(self, start):
        global if_run
        if_run = start.data
        print(if_run)

    
    def set_max_count(self, max_time_min = 6):
        # arbitrarily large number for very long videos (5 minutes, fps)
        self.max_count = int(max_time_min * 60 * self.fps)
    

    def get_tracks_from_frame(self, frame):
        '''Given an image in a numpy array, find and track a turtle.
        Returns an array of numbers with class,x1,y1,x2,y2,conf,track_id with
        x1,y1,x2,y2 all resized for the image'''
        # [cls, x1 y1 x2 y2 conf, track_id, predicted class, classification_confidence]
        no_detection_case = [np.array([0, 0, 0.1, 0, 0.1, 0, -1, 0, 0.0])]
        box_array = []
        results = self.model_track.track(source=frame, 
                                         stream=True, 
                                         persist=True, 
                                         boxes=True,
                                         verbose=False,
                                         conf=self.detection_confidence_threshold, # test for detection thresholds
                                         iou=self.detection_iou_threshold,
                                         tracker='botsorttracker_config.yaml')
        
        
        # code.interact(local=dict(globals(), **locals()))
        # if len(results) == 0:
        #     return no_detection_case
        # if len(results) > 0:
        for r in results:
            boxes = r.boxes
            print(boxes)
            input()
            annotated_frame = r.plot()
            # no detection case
            if boxes.id is None:
                return box_array.append(no_detection_case) # assign turtle with -1 id for no detections

            for i, id in enumerate(boxes.id):
                xyxyn = np.array(boxes.xyxyn[i, :])
                box_array.append([int(boxes.cls[i]),            # class
                                float(xyxyn[0]),    # x1
                                float(xyxyn[1]),    # y1
                                float(xyxyn[2]),    # x2
                                float(xyxyn[3]),    # y2
                                float(boxes.conf[i]),         # conf
                                int(boxes.id[i])])            # track id
        
        self.pub_output_img.publish(self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8'))
        return box_array
        

    def get_video_info(self):
        """ get video info """
        
        print(f'video name: {self.video_name}')
        print(f'video location: {self.video_path}')
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()
            
        # get fps of video
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        print(f'Video FPS: {self.fps}')
        
        # get total number of frames of video
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(f'Video frame count: {total_frames}')
        
        self.image_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        print(f'image width: {self.image_width}')
        print(f'image height: {self.image_height}')
        

    def get_tracks_from_video(self, SHOW=False): 
        ''' Given video file, get tracks across entire video
        Returns list of image tracks (ImageTrack object)
        MAX_COUNT = maximum number of frames before video closes
        UPDATE: also write frames at the same time, don't worry about overall setup (will have flickering)
        '''
        # create video writing object
        video_out_name = os.path.join(self.save_dir, self.video_name + '_tracked.mp4')
        video_out = cv.VideoWriter(video_out_name, 
                                   cv.VideoWriter_fourcc(*'mp4v'), 
                                   int(np.ceil(self.fps / self.frame_skip)), 
                                   (self.image_width, self.image_height), 
                                   isColor=True)
        # create plotting object (to draw bboxes onto frame)
        plotter = Plotter(self.image_width, self.image_height)
        
        # create video reading object
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f'Error opening video file: {self.video_path}')
            exit()
        
        print(f'Frame skip interval: {self.frame_skip}')
        
        start_read_time = time.time()
        
        count = 0
        image_detection_list = []
        
        while cap.isOpened() and count <= self.max_count:
            success, frame = cap.read()
            if not success:
                cap.release() # release object
                break
        
            # skip frames based on FRAME_SKIP
            if count % self.frame_skip == 0:
                print(f'frame: {count}')
                
                self.pub_clean_img.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

                # generate unique image name in case of writing frame to file
                count_str = '{:06d}'.format(count)
                image_name = self.video_name + '_frame_' + count_str + self.image_suffix
                save_path = os.path.join(self.save_dir, image_name)
                
                # TODO downsize frame to 640
                frame_resize = cv.resize(frame, dsize=None, fx=self.image_scale_factor, fy=self.image_scale_factor)
                
                # track and detect single frame
                # [class,x1,y1,x2,y2,conf,track_id, classification, classification_conf] with x1,y1,x2,y2 all resized for the image
                box_list = self.get_tracks_from_frame(frame_resize)
                
                
                # for each detection, run classifier
                box_array_with_classification = []
                if type(box_list) == type(None):
                    no_detection_case = [np.array([0, 0, 0.1, 0, 0.1, 0, -1, 0, 0.0])]
                    box_array_with_classification = no_detection_case
                    
                else: 
                    for box in box_list:
                        # classifer works on PIL images currently due to image transforms
                        # TODO change to yolov8 so no longer require PIL image - just operate on numpy arrays
                        
                        # TODO grab classification/image crops from original frame size
                        frame_rgb = PILImage.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                        # image_rgb.show()
                        
                        
                        
                        image_crop = self.TurtleClassifier.crop_image(frame_rgb, box[1:5], self.image_width, self.image_height)
                        
                        predicted_class, predictions = self.TurtleClassifier.classify_image(image_crop)
                        # append classifications to the det object
                        # det.append_classification(predicted_class, 1-predictions[predicted_class].item())
                        box.append(predicted_class)
                        box.append(1-predictions[predicted_class].item())
                        box_array_with_classification.append(box)
                        
                det = ImageWithDetection(txt_file='empty', 
                                        image_name=save_path,
                                        detection_data=box_array_with_classification,
                                        image_width=self.image_width,
                                        image_height=self.image_height)
                 
                # else detections, classifications are empty 
                image_detection_list.append(det)
                
                # make plots
                plotter.boxwithid(det.detection_data, frame)
                # write to frame
                if self.WRITE_VID:
                    video_out.write(frame)
                
                # TODO output tracks to some ROS topic or something
                # may be able to just call convert_images_to_tracks on current image_detection_list
                tracks_current = self.convert_images_to_tracks(image_detection_list)
                print(f'num tracks currently: {len(tracks_current)}') # tracks that aren't -1 ID
                
            if SHOW:
                img = cv.resize(frame, None, fx=self.img_scale_factor,
                                fy=self.img_scale_factor, interpolation=cv.INTER_AREA)
                cv.imshow('images', img)
                cv.waitKey(0)   
            
            count += 1
            
        # release the video capture object
        cap.release()
        
        end_read_time = time.time()
        sec = end_read_time - start_read_time
        print('video read time: {} sec'.format(sec))
        print('video read time: {} min'.format(sec / 60.0))
        
        return image_detection_list



    def count_painted_turtles_overall(self, tracks):
        """ count painted turtle tracks, tracks must be classified overall """
        painted_count = 0
        unpainted_count = 0
        for i, track in enumerate(tracks):
            if track.classification_overall:
                painted_count += 1
            else:
                unpainted_count += 1
        return painted_count, unpainted_count
    
    
    def print_tracks_overall(self, tracks):
        """ print tracks overall"""
        with open(self.output_tracks, mode='w', newline='') as csv_file:
            f = csv.writer(csv_file)
            f.writerow(['track id', 'len', 'avg', 'classification'])
            for i, track in enumerate(tracks):
                write_str = [track.id,len(track.classifications),self.calculate_mean_classification(track.classifications),track.classification_overall]
                f.writerow(write_str)
            

    
    def convert_images_to_tracks(self, image_list):
        """ convert image detections to tracks"""
        
        # iterate through image_list and add/append onto tracks
        tracks = []
        track_ids = []
        # for each image_list, add a new track whenever we have new ID
        # when going through each detection, if old ID, then append detection info
        for image in image_list:
            for detection in image.detections:
                if detection.id not in track_ids:
                    # new id, thus add to track_ids and create new track and append it to list of tracks
                    tracks.append(ImageTrack(detection.id, detection))
                    track_ids.append(detection.id) # not sure if better to maintain a list of track_ids in parallel, or simply make list when needed
                    
                else:
                    # detection id is in track_ids, so we add to existing track
                    # first, find corresponding track index
                    track_index = track_ids.index(detection.id)
                    # then we add in the new detection to the same ID
                    tracks[track_index].add_detection(detection)
        
        return tracks
    

    
    def classify_tracks_overall(self, tracks):
        """ classify trackers overall after per-image classification has been done """
        
        # each track has a classification and a classification_confidence
        
        # what defines the overall classification of painted (1) vs not painted (0)
        # arbitrarily:
        # if over 50% of the tracks are ID'd as painted, then the overall track is painted
        # else default to not painted
        
        # overall_class_confidence_threshold = 0.5 # all class confidences must be greater than this
        # overall_class_track_threshold = 0.5 # half of the track must be painted
        for track in tracks:
            if self.check_overall_class_tracks(track.classifications, self.overall_class_track_threshold): # and \
                # self.check_overall_class_confidences(track.classification_confidences, self.overall_class_confidence_threshold):
                track.classification_overall = 1 # painted turtle
            else:
                track.classification_overall = 0 # unpainted turtle
                
        return tracks
    
    
    def calculate_mean_classification(self, class_track):
        classes_per_image = np.array(class_track)
        return np.sum(classes_per_image) / len(classes_per_image)
        
    def check_overall_class_tracks(self, class_track, overall_threshold=0.5):
        if  self.calculate_mean_classification(class_track) > overall_threshold:
            return True
        else:
            return False


    def read_config(self, config_file):
        """_summary_

        Args:
            config_file (_type_): _description_
        """
        
        with open(config_file, 'r') as file:
            yaml_data = yaml.safe_load(file)
            
        # Extract the variables
        config = {'video_path_in': yaml_data['video_path_in'],
                  'save_dir': yaml_data['save_dir'],
                  'detection_model_path': yaml_data['detection_model_path'],
                  'classification_model_path': yaml_data['classification_model_path'],
                  'YOLOv8_install_path': yaml_data['YOLOv8_install_path'],
                  'frame_skip': yaml_data['frame_skip'],
                  'detection_confidence_threshold': yaml_data['detection_confidence_threshold'],
                  'detection_iou_threshold': yaml_data['detection_iou_threshold'],
                  'overall_class_confidence_threshold': yaml_data['overall_class_confidence_threshold'],
                  'overall_class_track_threshold': yaml_data['overall_class_track_threshold'],
                  'detector_image_size': yaml_data['detector_image_size']}
        
        return config


    def write_counts_to_file(self, output_file, count_painted, count_unpainted, count_total):
        """write_counts_to_file

        Args:
            output_file (str): absolute filepath to where we want to save the file
        """
        
        title_row = ['Raine AI Turtle Counts']
        label_vid = ['Video name']
        label_date = ['Date counted']
        
        date_counted = [self.datestr.strftime("%Y-%m-%d")]
        label_counts = ['painted', 'unpainted', 'total']
        counts = [count_painted, count_unpainted, count_total]
        
        # also output yaml file (configuration parameters to the csv)
        with open(self.config_file, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        
        with open(output_file, mode='w', newline='') as csv_file:
            f = csv.writer(csv_file)
            f.writerow(title_row)
            f.writerow([label_vid, self.video_name])
            f.writerow([label_date, date_counted])
            
            for i in range(len(counts)):
                f.writerow([label_counts[i], counts[i]])
            
            
            header = ['pipeline_config.yaml']
            f.writerow(header)
            for key, value in yaml_data.items():
                f.writerow([key, value])
            
        print(f'Counts written to {output_file}')

        
    def run(self, SHOW=False):

        start_time = time.time()

        # get detection list for each image
        image_detection_list = self.get_tracks_from_video(SHOW)
        # NOTE: we don't need to save each frame, because each frame is already 
        # just want to save the detections/metadata to a file for replotting
        # and we re-open the video when it's time to make the video with detections/plots
        
        # should be input, also slight misnomer perhaps because Yolov8 tracking actually happens in GetTracksFromVideo function
        # TODO merge into tracking pipeline
        # tracker_obj = Tracker(self.video_path, 
        #                       self.save_dir,
        #                       classifier_weights=self.classifier_weights, 
        #                       yolo_dir=self.yolo_path,
        #                       image_width=image_width, 
        #                       image_height=image_height,
        #                       overall_class_confidence_threshold=self.overall_class_confidence_threshold,
        #                       overall_class_track_threshold=self.overall_class_track_threshold)
        
        # code.interact(local=dict(globals(), **locals()))
        
        # convert from image list detections to tracks
        tracks = self.convert_images_to_tracks(image_detection_list)
        
        # run classifier on tracks
        # NOTE: requires the actual image!
        # tracks_classified = self.classify_tracks(tracks)
        
        # run classification overall on classified tracks
        tracks_overall = self.classify_tracks_overall(tracks)        
        
        # convert tracks back into image detections!
        # image_detection_track_list = self.convert_tracks_to_images(tracks_overall)
        
        # plot classified tracks to file by re-opening the video and applying our tracks back to the images
        # self.make_video_after_tracks(image_detection_track_list)
        
        # print tracks overall - to get an idea of how many there are (overall)
        self.print_tracks_overall(tracks_overall)
        
        # for overall counts of painted turtles:
        painted, unpainted = self.count_painted_turtles_overall(tracks_overall)
        print("Overal counts")
        print(f'painted count: {painted}')
        print(f'unpainted count: {unpainted}')
        print(f'total turtles: {len(tracks)}') # TODO length of tracks that are not -1!
        
        # painted, unpainted = self.count_painted_turtles(tracks)
        # print("Count along the way")
        # print(f'painted count: {painted}')
        # print(f'unpainted count: {unpainted}')
        
        print('counting complete')
        end_time = time.time()
        sec = end_time - start_time
        print('compute time: {} sec'.format(sec))
        print('compute time: {} min'.format(sec / 60.0))
        print('compute time: {} hrs'.format(sec / 3600.0))
        
        print(f'Number of frames processed: {len(image_detection_list)}')
        print(f'Seconds/frame: {sec  / len(image_detection_list)}')
                
        self.write_counts_to_file(os.path.join(self.save_dir, self.output_file), painted, unpainted, len(tracks))
        
        # to check for different thresholds of painted overall:
        # self.overall_painted_count(tracks_overall, 0.1)
        
        code.interact(local=dict(globals(), **locals()))
        
        
        return tracks_overall   
        
        
    def overall_painted_count(self, tracks, threshold_overall=0.5):
        t = [tr for tr in tracks if self.calculate_mean_classification(tr.classifications) > threshold_overall]
        return len(t)

def main():
    rospy.init_node("track_pipeline", disable_signals=True)
    config_file = 'pipeline_config_sm.yaml' # locally-referenced from cd: tracker folder
    p = Pipeline(config_file=config_file, max_frames=0)
    # p = Pipeline(config_file=config_file)
    if 1:
        results = p.run()

    try:
        rospy.spin()
    except (KeyboardInterrupt, SystemExit):
        print("shutting down serial/ROS node")

    cv.destroyAllWindows()



if __name__ == "__main__":
    
    main()