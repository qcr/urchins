import os
import cv2
import yaml
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

def main():
    # Read configuration from YAML file
    with open("trac/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    video_path = config["video_path_in"]
    output_dir = config["save_dir"]
    weights_path = config["detection_model_path"]
    conf_threshold = config.get("detection_confidence_threshold")

    # Load the YOLOv8 model
    model = YOLO(weights_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # Define the codec and create VideoWriter object
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = os.path.join(output_dir, f"{video_filename}_BS_trac_demo.mp4")
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_file, codec, fps, (output_width, output_height))

    # Track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=conf_threshold, tracker="botsort.yaml")
            boxes = results[0].boxes.xywh.cpu()

            if len(boxes) != 0:
                print(f"confidence score: {results[0].boxes.conf}")
                if results[0].boxes.id == None:
                    print("No track ids found")
                else: 
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    print("Track ids found")
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                        out.write(annotated_frame)
                        #cv2.imshow("YOLOv8 Tracking lines", resized_frame)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Write the frame with annotations to the output video
            out.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
               break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the output video writer
    cap.release()
    out.release()

    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print("Tracking complete!")
