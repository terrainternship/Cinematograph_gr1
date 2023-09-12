from ultralytics import YOLO
import torch


def Track(file):
    # Load the model.
    model = YOLO('models/smokeroboflow_yolov8s_640_best.pt')
    # model.to('cuda')
    rez=model.predict(file, save=False, conf=0.45)
    print('______________________________')


    out =[i+1 for i,r in enumerate (rez) if len(r.boxes.cls)>0]
    return out


print('CUDA -',torch.cuda.is_available())
