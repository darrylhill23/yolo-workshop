import os
import cv2
import shutil
from ultralytics import YOLO

# --------------------
# Paths
# --------------------
CANDIDATES = "candidates"
IMG_OUT = "dataset/images/train"
LBL_OUT = "dataset/labels/train"

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# --------------------
# Model
# --------------------
model = YOLO("runs/train/exp/weights/best.pt")

# --------------------
# Load images
# --------------------
images = [
    f for f in os.listdir(CANDIDATES)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# --------------------
# Loop
# --------------------
for img_name in images:
    img_path = os.path.join(CANDIDATES, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w = img.shape[:2]

    # --------------------
    # Inference
    # --------------------
    results = model.predict(img, conf=0.05, verbose=False, device=0)[0]
    boxes = results.boxes

    # --------------------
    # YOLO format labels
    # --------------------
    labels = []

    if boxes is not None and len(boxes) > 0:
        cls_list = boxes.cls.tolist()
        xyxy_list = boxes.xyxy.tolist()

        for cls, (x1, y1, x2, y2) in zip(cls_list, xyxy_list):
            cls = int(cls)

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            labels.append(f"{cls} {xc} {yc} {bw} {bh}")

        # --------------------
        # Visualization (YOLO built-in)
        # --------------------
        annotated = results.plot()

        cv2.imshow("review", annotated)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        # --------------------
        # Decision
        # --------------------
        if key == ord('y'):
            # move image
            new_img_path = os.path.join(IMG_OUT, img_name)
            shutil.move(img_path, new_img_path)

            # write label file (even if empty = valid negative sample)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(LBL_OUT, label_name)

            with open(label_path, "w") as f:
                f.write("\n".join(labels))

            print(f"[ACCEPTED] {img_name}")

        elif key == ord('n'):
            print(f"[SKIPPED] {img_name}")
            continue

        else:
            print("[INVALID KEY] Use y or n. Skipping.")