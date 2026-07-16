from ultralytics import YOLO
import cv2

model = YOLO("runs/train/exp/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf = 0.05)[0]
    annotated = results.plot()

    cv2.imshow("YOLO", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or q
        break

cap.release()
cv2.destroyAllWindows()