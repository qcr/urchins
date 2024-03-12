from ultralytics import YOLO


# Yolo's Model 
# model = YOLO('weights/yolov8l.pt')

# load pretrained model
model = YOLO('weights/X_Urchin_Detector_2024_03_12.pt')

# train the model
model.train(data='data/urchin.yaml', 
            epochs=2500, 
            imgsz=640,
            workers=10,
            cache=True,
            amp=False,
            batch=-1,
            patience = 250,
            pretrained = True
            )

print('done')