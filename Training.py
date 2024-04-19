import os
import json
import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO

#Load model
yolo_yaml_path = r'C:\Users\Admin\Desktop\LearningAI\CarDetection\yolo_data\data.yml'
model = YOLO('yolov8s.yaml').load('yolov8s.pt')

#Set up some important parametters
epochs = 15 # The number of trainging model iterated
imgsz = 640
batch_size = 8
patience = 5
lr = 0.0005

result = model.train(
    data = yolo_yaml_path,
    epochs = epochs,
    imgsz = imgsz,
    batch = batch_size,
    lr0 = lr,
    patience = patience,
    project = 'model',
    name = 'yolov8/detect/train'
)