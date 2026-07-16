from ultralytics import YOLO
import cv2
import time
from collections import deque

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

prev_time = time.time()
fps_history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # timing
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time

    fps = 1.0 / dt if dt > 0 else 0
    fps_history.append(fps)

    smooth_fps = sum(fps_history) / len(fps_history)

    results = model(frame)[0]
    annotated = results.plot()

    cv2.putText(
        annotated,
        f"FPS: {smooth_fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("YOLO", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()