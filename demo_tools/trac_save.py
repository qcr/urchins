import os
import cv2 as cv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('weights/X_Urchin_Detector_2024_03_12.pt')

# Open the video file
video_path = "/home/wardlewo/Reggie/data/20240222_urchin_videos_aims/unseen_test/20240222_aims_GX010899.MP4"
cap = cv.VideoCapture(video_path)
video_filename = os.path.splitext(os.path.basename(video_path))[0]
# Define the codec and create VideoWriter object
codec = cv.VideoWriter_fourcc(*'mp4v')
output_file = f"{video_filename}_detection_demo.mp4"
output_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
out = cv.VideoWriter(output_file, codec, fps, (output_width, output_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.9)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the frame with annotations to the output video
        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        #if cv.waitKey(1) & 0xFF == ord("q"):
           # break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the output video writer
cap.release()
out.release()
cv.destroyAllWindows()