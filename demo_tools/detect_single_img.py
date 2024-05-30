from ultralytics import YOLO

# Load a model
model = YOLO('X_Urchin_Detector_2024_03_12.pt')  # pretrained YOLOv8n model

# Run inference on 'bus.jpg' with arguments
model.predict('/home/wardlewo/Reggie/data/weird_test/Hair.png', save=True, imgsz=320, conf=0.5)