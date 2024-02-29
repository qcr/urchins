from ultralytics import YOLO


# Yolo's Model 
model = YOLO('weights/yolov8l.pt')

# load pretrained model
model = YOLO('weights/best.pt')

# train the model
model.train(data='data/urchin.yaml', 
            epochs=50, 
            imgsz=640,
            workers=10,
            cache=True,
            amp=False,
            batch=-1
            )

print('done')