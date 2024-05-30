from ultralytics import YOLO


# Our Model 
#model = YOLO('weights/20240312_yolov8x_urchinDetector_best.pt')

# load pretrained model
model = YOLO('yolov8x.pt')

# train the model
model.train(data='data/urchin.yaml', 
            epochs=2500, 
            imgsz=640,
            workers=12,
            cache=True,
            amp=False,
            batch=-1,
            plots=True,
            patience = 150,
            pretrained = False,
            hsv_h = 0.015,
            hsv_s = 0.7,
            hsv_v = 0.4,
            flipud=0.10,
            fliplr=0.10,
            #scale = 0.1,
            project = "/home/wardlewo/Reggie/ultralytics_output/",
            name = "20240402_urchin_model"        
            #output_dir="/home/wardlewo/Reggie/ultralytics/runs/detection/"
            )

print('done')