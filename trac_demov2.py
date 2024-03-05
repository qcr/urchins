import cv2

import time

from ultralytics import YOLO
 
# Load the YOLOv8 model

model = YOLO('weights/best.pt')
 
# Open the video file

video_path = "/home/java/Java/data/20231201_urchin/sorted_videos/unseen_test/GX010296.MP4"

cap = cv2.VideoCapture(video_path)
 
# Initialize variables to track confidence and time

last_confidence = {}

last_time = {}
 
# Loop through the video frames

while cap.isOpened():

    # Read a frame from the video

    success, frame = cap.read()
 
    if success:

        # Run YOLOv8 tracking on the frame, persisting tracks between frames

        results = model.track(frame)
 
        # Iterate over each result in the current frame
        # import code
        # code.interact(local=dict(globals(), **locals()))
        for i, boxes in enumerate(results[0].boxes):

            class_index = boxes.cls[0].item()

            confidence = boxes.conf[0].item()
 
            # Check if confidence for this class exceeds 0.7

            if confidence > 0.7:

                # Update last_confidence and last_time for the class

                if class_index not in last_confidence:

                    last_confidence[class_index] = confidence

                    last_time[class_index] = time.time()

                else:

                    # If the time elapsed since the last update is greater than 0.5 seconds, update the confidence and time

                    if time.time() - last_time[class_index] > 0.5:

                        last_confidence[class_index] = confidence

                        last_time[class_index] = time.time()
 
                # Display the annotation only if it meets the conditions

                if last_confidence[class_index] > 0.7:

                    annotated_frame = results[0].plot()

                    cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

                    cv2.resizeWindow("YOLOv8 Tracking", 600, 600)

                    cv2.imshow("YOLOv8 Tracking", annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):

                        break
 
    else:

        # Break the loop if the end of the video is reached

        break
 
# Release the video capture object and close the display window

cap.release()

cv2.destroyAllWindows()
