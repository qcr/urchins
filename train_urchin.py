from ultralytics import YOLO

# load pretrained model
model = YOLO('weights/yolov8x.pt')

# train the model
model.train(data='data/cslics_surface.yaml', 
            epochs=300, 
            imgsz=640,
            workers=10,
            cache=True,
            amp=False,
            batch=-1
            )

print('done')
