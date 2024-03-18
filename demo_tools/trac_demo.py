import cv2 as cv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('weights/X_Urchin_Detector_2024_03_12.pt')

# Open the video file
video_path = "/home/wardlewo/Reggie/data/20231201_urchin/unseen_test/GX010300.MP4"
cap = cv.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        cv.namedWindow("YOLOv8 Tracking",cv.WINDOW_NORMAL)
        cv.resizeWindow("YOLOv8 Tracking", 600, 600)
        # Display the annotated frame
        cv.imshow("YOLOv8 Tracking", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()