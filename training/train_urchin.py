from ultralytics import YOLO


# Yolo's Model 
# model = YOLO('weights/yolov8l.pt')



# load pretrained model
model = YOLO('weights/20240312_yolov8x_urchinDetector_best.pt')

# train the model
model.train(data='data/urchin.yaml', 
            epochs=1000, 
            imgsz=640,
            workers=10,
            cache=True,
            batch=-1,
            patience = 100,
            pretrained = True,
            #plots = True,
            #Speed opt
            amp= True,

            project = "/home/wardlewo/Reggie/ultralytics_output/"
            
            #output_dir="/home/wardlewo/Reggie/ultralytics/runs/detection/"
            )

print('done')