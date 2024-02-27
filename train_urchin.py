from ultralytics import YOLO

# load pretrained model
model = YOLO('weights/yolov8n.pt')

# train the model
model.train(data='data/urchin.yaml', 
            epochs=15, 
            imgsz=640,
            workers=10,
            cache=True,
            amp=False,
            batch=-1
            )

print('done')