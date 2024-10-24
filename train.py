#This code trains a model based on the customised dataset

from ultralytics import YOLO

obb_model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
import yaml
with open('final_version/data.yaml', 'r') as file:
  data = yaml.safe_load(file)
data['train'] = "train/images"
data['val'] = "valid/images"
with open('final_version/data.yaml', 'w') as file:
  yaml.dump(data, file, default_flow_style=False)
  
# Train the model
results = obb_model.train(data='final_version/data.yaml', epochs=50, imgsz=640,batch=16)

metrics = obb_model.val(data='final_version/data.yaml')  # no arguments needed, dataset and settings remembered

#shape algorithm -> train11 and train112 -> first try (epochs = 20, imgsz=640, batch=16)
#shape algorithm -> train5 and train52 -> second try (epochs = 30, imgsz=640, batch=20) - got worst
#shape algorithm -> train6 and train62 -> third try (epochs = 40, imgsz=640, batch=16) 
#shape algorithm -> train7 and train72 -> fourth try (epochs = 10, imgsz=640, batch=16) - got better (best)
#shape algorithm -> train12 and train122 -> fifth try (epochs = 5, imgsz=640, batch=16) - bad
#shape algorithm -> train13 and train132 -> sixth try (epochs = 15, imgsz=640, batch=16) 
#shape algorithm -> train14 and train142 -> seventh try (epochs = 11, imgsz=640, batch=16) - slightly better
#shape algorithm -> train16 and train162 -> eighth try (added more dataset from original one)

#lineandcurve algorithm -> train 17 and train 172 -> first try 
#lineand curve algorithm -> train19 and train 192 -> second try (increase 211 images)
#lineandcurve algorithm -> train20 and train202 -> third try 
#train21 and train212 
#train3 and train31
#train4 and train41
#train23 and train232
