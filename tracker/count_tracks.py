import os
import cv2 as cv
import yaml
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

classes = ['Tripys', 'Rings']
green = [0, 255, 00] #Rings
red = [0, 0, 255] #Tripys
class_colours = [green, red]

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

def save_track_img(results, track_history, track_class, frame, save_path, SHOW=False):
    """Save the frame with tracking lines and bounding boxes.
    Args:
        results (list): List of detection results from model.track().
        track_history (dict): Dictionary of track histories.
        track_class (dict): Dictionary of track classes.
        frame (np.ndarray): Frame image.
        save_path (str): Path to save the image.
        SHOW (bool, optional): Whether to display the image. Defaults to False.
    Returns: True"""
    #try:
    if results[0].boxes.id == None:
        print("No track ids found")
        return track_class
    track_ids = results[0].boxes.id.int().cpu().tolist()
    boxes = results[0].boxes
    for i, track_id in enumerate(track_ids):
        box = boxes[i]
        x, y, w, h = box.xywh[0]
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        class_id = int(box.cls.item())
        conf = box.conf.item()
        track = track_history[track_id]
        track.append((float(x), float(y)))
        tr_cls = track_class[track_id]
        tr_cls.append(class_id)
        if len(track) > 90:  # Retain 90 tracks for 90 frames
            track.pop(0)
        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv.polylines(frame, [points], isClosed=False, color=class_colours[class_id], thickness=10)  # Draw tracking lines
        cv.rectangle(frame, (x1, y1), (int(x+w/2), int(y+h/2)), class_colours[class_id], 2)  # Rectangle around object
        cv.putText(frame, f"{track_id} {classes[class_id]}: {conf:.2f}", (int(x1-5), int(y1-5)), 
                    cv.FONT_HERSHEY_SIMPLEX, fontScale=3.5, color=class_colours[class_id], thickness=3)
    if SHOW:
        cv.imshow("Tracking", frame)
        cv.waitKey(1)
    cv.imwrite(save_path, frame)

    return track_class
    # except Exception as e:
    #     print(f"Error saving image: {e}")
    #     return tr
    
def class_counts(track_class, class_count_list):
    for key in track_class.keys():
        length = len(track_class[key])
        total = sum(track_class[key])
        if total/length > 0.5:
            class_count_list[1] += 1
        else:
            class_count_list[0] += 1
    return class_count_list



def track(video_path, save_dir, model, track_config, conf_threshold, frame_skip, max_count=9999999):
    """Track objects in a video using the YOLOv8 model.
    Args:
        video_path (str): Path to the video file.
        save_dir (str): Directory to save the output images.
        model (YOLO): YOLOv8 model object.
        track_config (str): Path to the tracking configuration file.
        conf_threshold (float): Detection confidence threshold.
        frame_skip (int): Number of frames to skip between detections.
        max_count (int, optional): Maximum number of frames to process. Defaults to 9999999.
    Returns:
        int: Number of tracks detected.
    """
    # Open the video file
    cap = cv.VideoCapture(video_path)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(save_dir, video_filename+'_detect')
    os.makedirs(output_dir, exist_ok=True)
    # Define the codec and create VideoWriter object
    codec = cv.VideoWriter_fourcc(*'mp4v')
    output_file = os.path.join(output_dir, f"{video_filename}_CT_trac_demo.mp4")
    output_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    out = cv.VideoWriter(output_file, codec, fps, (output_width, output_height))

    # Track history
    track_history = defaultdict(lambda: [])
    track_class = defaultdict(lambda: [])
    class_count_list = [0, 0]

    # Loop through the video frames
    count = 0

    while cap.isOpened():
        if count > max_count:
            print(f"Max count reached: {max_count}")
            break
        count += 1
        success, frame = cap.read()
        if not success:
            cap.release() # release object
            print("Video capture object not successful")
            break
        # skip frames based on FRAME_SKIP
        if count % frame_skip == 0:
            print(f'frame: {count}')
            count_str = '{:06d}'.format(count)
            image_name = video_filename + '_frame_' + count_str + '.jpg'
            save_path = os.path.join(output_dir, image_name)
            # TODO downsize frame to 640
            
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=conf_threshold, tracker=track_config)
            boxes = results[0].boxes.xywh.cpu()

            if len(boxes) != 0:
                track_class = save_track_img(results, track_history, track_class, frame, save_path)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # Write the frame with annotations to the output video
            out.write(annotated_frame)

            # Break the loop if 'q' is pressed
            #if cv.waitKey(1) & 0xFF == ord("q"):
            #  break

    # Release the video capture object and close the output video writer
    cap.release()
    out.release()
    class_count_list = class_counts(track_class, class_count_list)
    #import code
    #code.interact(local=dict(globals(), **locals()))
    #cv.destroyAllWindows()

    return len(track_history), class_count_list

def main():
    config = read_config("tracker/pipeline_config.yaml")
    video_path = config["video_path_in"]
    output_dir = config["save_dir"]
    weights_path = config["detection_model_path"]
    conf_threshold = config.get("detection_confidence_threshold")
    frame_skip = config.get("frame_skip")
    track_config = config.get("custom_tracker")
    model = YOLO(weights_path)
    track_counts, class_count_list = track(video_path, output_dir, model, track_config, conf_threshold, frame_skip)
    print(f"Number of tracks: {track_counts}")
    print(f"Ring count: {class_count_list[0]} Tripys count: {class_count_list[1]}")

if __name__ == "__main__":
    main()
    print("Tracking complete!")
