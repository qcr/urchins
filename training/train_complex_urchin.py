from ultralytics import YOLO


# Yolo's Model 
model = YOLO('X_TempName_2024_03_01.pt')

# load pretrained model
# model = YOLO('weights/best.pt')

# train the model
model.train(data='data/urchin.yaml', 
            epochs=2500, 
            imgsz=640,
            workers=12,
            cache=True,
            amp=False,
            batch=-1,
            patience = 300,
            pretrained = True,
            hsv_h = 0.015,
            hsv_s = 0.7,
            hsv_v = 0.4,
            perspective = 0.0
            )

print('done')