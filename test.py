from ultralytics import YOLO
import torch

# Load a model
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
# Train the model with 2 GPUs
results = model.train(data='data.yaml', epochs=10, imgsz=640, device='mps', name='yolov8_results', plots=True)