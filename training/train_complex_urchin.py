from ultralytics import YOLO


# Yolo's Model 
model = YOLO('weights/20240312_yolov8x_urchinDetector_best.pt')

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
            plots=True,
            patience = 300,
            pretrained = True,
            hsv_h = 0.015,
            hsv_s = 0.7,
            hsv_v = 0.4,
            flipud=0.10,
            fliplr=0.10,
            scale = 0.1,
            project = "/home/wardlewo/Reggie/ultralytics_output/",
            name = "20240222_urchin_videos_aims"        
            #output_dir="/home/wardlewo/Reggie/ultralytics/runs/detection/"
            )

print('done')