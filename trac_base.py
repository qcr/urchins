from ultralytics import YOLO

# Load an official or custom model
model = YOLO('weights/X_Urchin_Detector_2024_03_12.pt')  # Load a custom trained model

# Perform tracking with the model
results = model.track(source="/home/wardlewo/Reggie/data/20240222_urchin_videos_aims/unseen_test/20240222_aims_GX010887.MP4")  # Tracking with default tracker
#results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker

