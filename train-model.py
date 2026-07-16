from ultralytics import YOLO

# Load a pretrained model (recommended start)
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.

# Train
model.train(
    data="data.yaml",
    # if using cpu, this will take a long time. You can reduce the number of epochs to speed up training, but it may affect accuracy.
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,   # GPU 0 (use "cpu" if no GPU)
    workers=4,
    patience=0,
    project="runs/train",
    name="exp"
)