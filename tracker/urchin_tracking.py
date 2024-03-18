
#! /usr/bin/env python3
import os
import cv2 as cv
import yaml
import numpy as np
import time
import csv
from ultralytics import YOLO
from PIL import Image as PILImage
from tracker.ImageWithDetection import ImageWithDetection
from tracker.ImageTrack import ImageTrack


def read_config(config_file):
    """_summary_

    Args:
        config_file (_type_): _description_
    """
        
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config




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
        ###plotter = Plotter(self.image_width, self.image_height)
        
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
                
                self.pub_clean_img.publish(self.bridge.cv_to_imgmsg(frame, encoding='bgr8'))

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
                ###plotter.boxwithid(det.detection_data, frame)
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

def convert_images_to_tracks(image_list):
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

def print_tracks_overall(self, tracks):
    """ print tracks overall"""
    with open(self.output_tracks, mode='w', newline='') as csv_file:
        f = csv.writer(csv_file)
        f.writerow(['track id', 'len', 'avg', 'classification'])
        for i, track in enumerate(tracks):
            write_str = [track.id,len(track.classifications),self.calculate_mean_classification(track.classifications),track.classification_overall]
            f.writerow(write_str)

    def count_urchin(self, tracks):
        """ count painted turtle tracks, tracks must be classified overall """
        tripy_count = 0
        rings_count = 0
        for i, track in enumerate(tracks):
            #TODO Find out what class_labels outputs and correct for counting
            if track.class_labels:
                tripy_count += 1
            else:
                rings_count += 1
        return tripy_count, rings_count




def main():
    # Read configuration from YAML file
    config = read_config("tracker/pipeline_config.yaml")
    # Extract configuration parameters
    video_path = config["video_path_in"]
    output_dir = config["save_dir"]
    weights_path = config["detection_model_path"]
    conf_threshold = config.get("detection_confidence_threshold")

    # Load the YOLOv8 model
    model = YOLO(weights_path)

    # get detection list for each image
    image_detection_list = get_tracks_from_video(False)
    tracks = convert_images_to_tracks(image_detection_list)
    ###total_urchin_count = count_urchin(tracks)
    print('counting complete')
    end_time = time.time()
    sec = end_time - start_time
    print('compute time: {} sec'.format(sec))
    print('compute time: {} min'.format(sec / 60.0))
    print('compute time: {} hrs'.format(sec / 3600.0))
        
    print(f'Number of frames processed: {len(image_detection_list)}')
    print(f'Seconds/frame: {sec  / len(image_detection_list)}')


if __name__ == "__main__":
    main()
