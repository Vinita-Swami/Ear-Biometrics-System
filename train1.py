import os

from ultralytics import YOLO


model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='/content/gdrive/My Drive/EB/config.yaml', epochs=1, imgsz=640)
