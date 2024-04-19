#Using model from YOLOv8
from ultralytics import YOLO

model_path = 'C:\Users\Admin\Desktop\LearningAI\CarDetection\model\yolov8\detect\train\weights\best.pt'
model = YOLO(model_path)

metrics = model.val(
    project = 'model',
    name = 'yolov8/detect/val'
)