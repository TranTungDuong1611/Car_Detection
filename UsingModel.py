import os
from ultralytics import YOLO

model_path = r'\CarDetection\model\yolov8\detect\train\weights\best.pt'
test_img_path = r'\CarDetection\Test'

conf_thres = 0.8
model = YOLO(model_path)
#result = model.predict(test_img_path, save=True)
for image in os.listdir(test_img_path):
    img_path = os.path.join(test_img_path, image)
    result = model.predict(img_path, save=True)
